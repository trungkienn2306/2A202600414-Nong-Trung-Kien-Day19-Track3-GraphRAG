"""Step 5 — Full benchmark: GraphRAG vs Flat RAG on 20 questions.

Usage:
    python scripts/05_benchmark.py --question-set test
    python scripts/05_benchmark.py --question-set full
    python scripts/05_benchmark.py --questions-file data/multi-hops-question/multi-hops-question.json
    python scripts/05_benchmark.py --question-set test --force-reindex
    python scripts/05_benchmark.py --question-set test --seeds 3 --run-name seeds3_test
    python scripts/05_benchmark.py --dry-run   # print questions only, don't call APIs

Output:
    benchmark_results.csv
    report/benchmark_runs/<run_name>/benchmark_details.csv
    report/benchmark_runs/<run_name>/summary.json
    report/benchmark_runs/<run_name>/run_metadata.json
    Console: accuracy/latency/cost summary table
"""
import sys
import argparse
import functools
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.graphrag as graphrag_mod
import src.flatrag  as flatrag_mod
from src.build_graph  import load_node_embeddings
from src.fetch_corpus import load_chunks
from src.benchmark    import (
    run,
    print_summary,
    build_summary,
    load_questions,
    save_run_artifacts,
    FULL_QUESTIONS_FILE,
    TEST_QUESTIONS_FILE,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run GraphRAG vs Flat RAG benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print questions without calling APIs")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seed nodes for GraphRAG (default: 3)")
    parser.add_argument(
        "--question-set",
        choices=["test", "full"],
        default="test",
        help="Use test subset or full benchmark set (default: test)",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Optional JSON path override for questions file",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuilding persistent Chroma index instead of reusing cache",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for report artifacts (default: auto timestamp)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run low-cost smoke subset (fixed hard IDs) before full sweep",
    )
    parser.add_argument(
        "--max-latency-multiplier",
        type=float,
        default=1.5,
        help="Efficiency gate: max GraphRAG/FlatRAG avg latency ratio (default: 1.5)",
    )
    parser.add_argument(
        "--max-cost-multiplier",
        type=float,
        default=1.5,
        help="Efficiency gate: max GraphRAG/FlatRAG total cost ratio (default: 1.5)",
    )
    args = parser.parse_args()

    if args.questions_file:
        questions = load_questions(args.questions_file)
        question_label = f"custom file: {args.questions_file}"
    elif args.question_set == "full":
        questions = load_questions(FULL_QUESTIONS_FILE)
        question_label = "full (20 multi-hop)"
    else:
        questions = load_questions(TEST_QUESTIONS_FILE)
        question_label = "test subset"

    if args.smoke:
        smoke_ids = {2, 3, 4, 6, 11, 12, 13, 16}
        smoke_questions = [q for q in questions if q.get("id") in smoke_ids]
        if smoke_questions:
            questions = smoke_questions
            question_label = f"{question_label} + smoke(8)"

    print("=" * 55)
    print("  Step 5: Benchmark (GraphRAG vs Flat RAG)")
    print("=" * 55)
    print(f"Question set: {question_label}")

    if args.dry_run:
        print(f"\n{len(questions)} benchmark questions:\n")
        for qa in questions:
            print(f"  Q{qa['id']:02d} [{qa['type']}] {qa['question']}")
        return

    # Load data
    print("\nLoading node embeddings...")
    try:
        node_embeddings = load_node_embeddings()
    except Exception as exc:
        logging.exception("Failed loading node embeddings: %s", exc)
        sys.exit(1)
    print(f"  {len(node_embeddings)} nodes")

    print("\nBuilding Flat RAG index...")
    try:
        chunks = load_chunks()
        flat_col = flatrag_mod.build_index(chunks, force_rebuild=args.force_reindex)
    except Exception as exc:
        logging.exception("Failed preparing FlatRAG index: %s", exc)
        sys.exit(1)

    # Bind query functions to their runtime state
    gr_fn = functools.partial(graphrag_mod.query, node_embeddings=node_embeddings, top_k_seeds=args.seeds)
    fr_fn = functools.partial(flatrag_mod.query,  collection=flat_col)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    runs_root = Path(__file__).parent.parent / "report" / "benchmark_runs"
    run_output_dir = runs_root / run_name

    print(f"\nRunning {len(questions)} questions...\n")
    try:
        df = run(gr_fn, fr_fn, questions=questions, output_dir=run_output_dir)
    except Exception as exc:
        logging.exception("Benchmark run failed: %s", exc)
        sys.exit(1)
    print_summary(df)
    summary = build_summary(
        df,
        run_name=run_name,
        question_set_label=question_label,
        seeds=args.seeds,
    )
    metadata = {
        "run_name": run_name,
        "question_set": question_label,
        "question_count": len(questions),
        "seeds": args.seeds,
        "force_reindex": bool(args.force_reindex),
        "questions_file": args.questions_file,
    }
    save_run_artifacts(output_dir=run_output_dir, summary=summary, metadata=metadata)

    metrics = summary["metrics"]
    gr_acc = metrics["overall_accuracy"]["graphrag"]
    fr_acc = metrics["overall_accuracy"]["flatrag"]
    gr_lat = metrics["avg_latency_seconds"]["graphrag"]
    fr_lat = metrics["avg_latency_seconds"]["flatrag"] or 1e-9
    gr_cost = metrics["query_cost_usd"]["graphrag"]
    fr_cost = metrics["query_cost_usd"]["flatrag"] or 1e-9

    latency_ratio = gr_lat / fr_lat
    cost_ratio = gr_cost / fr_cost
    gate_accuracy_ok = gr_acc >= fr_acc
    gate_efficiency_ok = (
        latency_ratio <= args.max_latency_multiplier
        and cost_ratio <= args.max_cost_multiplier
    )
    print(
        f"\nGate check => accuracy_ok={gate_accuracy_ok}, efficiency_ok={gate_efficiency_ok} "
        f"(latency_ratio={latency_ratio:.2f}, cost_ratio={cost_ratio:.2f})"
    )
    if args.smoke and not gate_accuracy_ok:
        print("[STOP] Smoke gate failed on accuracy. Tune pipeline before full run.")
        sys.exit(2)
    if args.smoke and gate_accuracy_ok and not gate_efficiency_ok:
        print("[STOP] Smoke gate failed on efficiency bounds. Tune before full run.")
        sys.exit(3)

    print("\nFull results: benchmark_results.csv")
    print(f"Run artifacts: {run_output_dir}")


if __name__ == "__main__":
    main()
