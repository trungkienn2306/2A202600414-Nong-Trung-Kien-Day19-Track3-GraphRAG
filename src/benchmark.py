"""Step 6 — Benchmark runner: GraphRAG vs Flat RAG."""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, ROOT

client = OpenAI(api_key=OPENAI_API_KEY)

QUESTION_DIR = ROOT / "data" / "multi-hops-question"
FULL_QUESTIONS_FILE = QUESTION_DIR / "multi-hops-question.json"
TEST_QUESTIONS_FILE = QUESTION_DIR / "test-multi-hops-question.json"

INPUT_PRICE = 0.15 / 1_000_000   # gpt-4o-mini input token
OUTPUT_PRICE = 0.60 / 1_000_000  # gpt-4o-mini output token
EMBED_PRICE = 0.02 / 1_000_000   # text-embedding-3-small token


def load_questions(path: Path | str = FULL_QUESTIONS_FILE) -> list[dict]:
    """Load benchmark questions from JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Questions file must contain a JSON array")
    required = {"id", "type", "question", "answer"}
    for idx, q in enumerate(payload, start=1):
        missing = required.difference(q.keys())
        if missing:
            raise ValueError(f"Question #{idx} missing keys: {sorted(missing)}")
    return payload


BENCHMARK_QUESTIONS: list[dict] = load_questions(FULL_QUESTIONS_FILE)


# ── LLM Judge ─────────────────────────────────────────────────

def judge(
    question: str,
    predicted: str,
    ground_truth: str,
    question_meta: dict | None = None,
) -> tuple[bool, str, int, int, list[str]]:
    """LLM-as-judge (strict): returns (correct, reason, prompt_tokens, completion_tokens, missing_facts)."""
    hops_hint = ""
    chain_hint = ""
    if question_meta:
        hops = question_meta.get("hops")
        chain = question_meta.get("chain")
        if isinstance(hops, list) and hops:
            hops_hint = "Expected hop domains/entities: " + ", ".join(str(h) for h in hops) + "\n"
        if isinstance(chain, str) and chain.strip():
            chain_hint = "Expected reasoning chain: " + chain.strip() + "\n"

    msg = (
        "You are a strict benchmark judge for multi-hop QA.\n"
        + "Decide if the predicted answer is correct against the ground truth.\n"
        + "Rules:\n"
        + "1) The answer must include the core target entity/entities from the ground truth.\n"
        + "2) Partial/hedged answers that omit required entities should be false.\n"
        + "3) For multi-hop questions, answer must not collapse to only one hop.\n\n"
        + "Question: " + question + "\n"
        + "Ground truth: " + ground_truth + "\n"
        + hops_hint
        + chain_hint
        + "Predicted: " + predicted + "\n\n"
        + "Return JSON with this schema exactly:\n"
        + "{\"correct\": true/false, \"reason\": \"one sentence\", \"missing_facts\": [\"...\"]}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": msg}],
        response_format={"type": "json_object"},
        max_tokens=120,
        temperature=0,
    )
    r = json.loads(resp.choices[0].message.content)
    return (
        bool(r.get("correct", False)),
        r.get("reason", ""),
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
        [str(x) for x in r.get("missing_facts", [])] if isinstance(r.get("missing_facts", []), list) else [],
    )


def estimate_embed_tokens(text: str) -> int:
    """Approximate embedding token usage for cost estimates."""
    return max(1, len(text) // 4)


def _query_cost(
    prompt_tokens: int,
    completion_tokens: int,
    judge_prompt_tokens: int,
    judge_completion_tokens: int,
    embed_tokens: int,
) -> float:
    return (
        prompt_tokens * INPUT_PRICE
        + completion_tokens * OUTPUT_PRICE
        + judge_prompt_tokens * INPUT_PRICE
        + judge_completion_tokens * OUTPUT_PRICE
        + embed_tokens * EMBED_PRICE
    )


# ── Runner ─────────────────────────────────────────────────────

def run(
    graphrag_fn: Callable,
    flatrag_fn:  Callable,
    questions:   list[dict] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run full benchmark. graphrag_fn / flatrag_fn each accept (question: str) -> dict."""
    if questions is None:
        questions = BENCHMARK_QUESTIONS

    results = []
    print(f"Running {len(questions)} questions...\n" + "-" * 75)

    for qa in questions:
        q  = qa["question"]
        gt = qa["answer"]

        gr = graphrag_fn(q)
        time.sleep(0.4)
        fr = flatrag_fn(q)
        time.sleep(0.4)

        gr_ok, gr_reason, gr_judge_in, gr_judge_out, gr_missing = judge(q, gr["answer"], gt, qa)
        fr_ok, fr_reason, fr_judge_in, fr_judge_out, fr_missing = judge(q, fr["answer"], gt, qa)
        time.sleep(0.3)

        embed_tokens = estimate_embed_tokens(q)
        gr_cost = _query_cost(
            prompt_tokens=gr.get("prompt_tokens", 0),
            completion_tokens=gr.get("completion_tokens", 0),
            judge_prompt_tokens=gr_judge_in,
            judge_completion_tokens=gr_judge_out,
            embed_tokens=embed_tokens,
        )
        fr_cost = _query_cost(
            prompt_tokens=fr.get("prompt_tokens", 0),
            completion_tokens=fr.get("completion_tokens", 0),
            judge_prompt_tokens=fr_judge_in,
            judge_completion_tokens=fr_judge_out,
            embed_tokens=embed_tokens,
        )

        results.append({
            "id":                qa["id"],
            "type":              qa["type"],
            "question":          q,
            "ground_truth":      gt,
            "hops":              ", ".join(qa.get("hops", [])) if isinstance(qa.get("hops"), list) else "",
            "chain":             qa.get("chain", "") if isinstance(qa.get("chain", ""), str) else "",
            "graphrag_answer":   gr["answer"],
            "flatrag_answer":    fr["answer"],
            "graphrag_correct":  gr_ok,
            "flatrag_correct":   fr_ok,
            "graphrag_latency":  gr.get("latency", 0),
            "flatrag_latency":   fr.get("latency", 0),
            "graphrag_tokens":   gr.get("tokens", 0),
            "flatrag_tokens":    fr.get("tokens", 0),
            "graphrag_prompt_tokens": gr.get("prompt_tokens", 0),
            "graphrag_completion_tokens": gr.get("completion_tokens", 0),
            "flatrag_prompt_tokens": fr.get("prompt_tokens", 0),
            "flatrag_completion_tokens": fr.get("completion_tokens", 0),
            "graphrag_judge_prompt_tokens": gr_judge_in,
            "graphrag_judge_completion_tokens": gr_judge_out,
            "flatrag_judge_prompt_tokens": fr_judge_in,
            "flatrag_judge_completion_tokens": fr_judge_out,
            "embedding_tokens_estimate": embed_tokens,
            "graphrag_cost": round(gr_cost, 6),
            "flatrag_cost": round(fr_cost, 6),
            "graphrag_seeds":    ", ".join(gr.get("seeds", [])),
            "graphrag_subgraph": gr.get("subgraph_size", 0),
            "graphrag_reason":   gr_reason,
            "flatrag_reason":    fr_reason,
            "graphrag_missing_facts": "; ".join(gr_missing),
            "flatrag_missing_facts": "; ".join(fr_missing),
        })

        g = "PASS" if gr_ok else "FAIL"
        f = "PASS" if fr_ok else "FAIL"
        print(f"Q{qa['id']:02d} [{qa['type'][:5]}] GraphRAG={g}  FlatRAG={f} | {q[:55]}...")

    df  = pd.DataFrame(results)
    out = ROOT / "benchmark_results.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] Results saved -> {out}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        details_csv = output_dir / "benchmark_details.csv"
        df.to_csv(details_csv, index=False)
        print(f"[OK] Details CSV saved -> {details_csv}")
    return df


# ── Summary printer ────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    total  = len(df)
    df_mh  = df[df["type"] == "multi-hop"]
    df_s   = df[df["type"] == "simple"]

    gr_acc    = df["graphrag_correct"].mean()     * 100
    fr_acc    = df["flatrag_correct"].mean()      * 100
    gr_mh_acc = df_mh["graphrag_correct"].mean()  * 100
    fr_mh_acc = df_mh["flatrag_correct"].mean()   * 100
    gr_s_acc  = df_s["graphrag_correct"].mean()   * 100
    fr_s_acc  = df_s["flatrag_correct"].mean()    * 100

    gr_cost = df["graphrag_cost"].sum()
    fr_cost = df["flatrag_cost"].sum()

    print("\n" + "=" * 62)
    print("          BENCHMARK RESULTS")
    print("=" * 62)
    print(f"{'Metric':<32} {'GraphRAG':>10} {'FlatRAG':>9} {'Delta':>8}")
    print("-" * 62)
    print(f"{'Overall accuracy':<32} {gr_acc:>9.1f}% {fr_acc:>8.1f}% {gr_acc-fr_acc:>+7.1f}%")
    print(f"{'Multi-hop accuracy':<32} {gr_mh_acc:>9.1f}% {fr_mh_acc:>8.1f}% {gr_mh_acc-fr_mh_acc:>+7.1f}%")
    print(f"{'Simple Q accuracy':<32} {gr_s_acc:>9.1f}% {fr_s_acc:>8.1f}% {gr_s_acc-fr_s_acc:>+7.1f}%")
    print(f"{'Avg latency (s)':<32} {df['graphrag_latency'].mean():>10.2f} {df['flatrag_latency'].mean():>9.2f}")
    print(f"{'Query cost ($)':<32} {gr_cost:>10.4f} {fr_cost:>9.4f}")
    print("=" * 62)

    wins = df[(df["graphrag_correct"]) & (~df["flatrag_correct"])]
    print(f"\nGraphRAG wins (PASS, FlatRAG FAIL): {len(wins)}")
    for _, r in wins.iterrows():
        print(f"  Q{r['id']:02d} [{r['type']}] {r['question'][:60]}...")

    losses = df[(~df["graphrag_correct"]) & (df["flatrag_correct"])]
    print(f"\nFlatRAG wins  (PASS, GraphRAG FAIL): {len(losses)}")
    for _, r in losses.iterrows():
        print(f"  Q{r['id']:02d} [{r['type']}] {r['question'][:60]}...")


def build_summary(
    df: pd.DataFrame,
    *,
    run_name: str,
    question_set_label: str,
    seeds: int,
) -> dict:
    """Build summary payload for report artifacts."""
    df_mh = df[df["type"] == "multi-hop"]
    df_s = df[df["type"] == "simple"]
    graphrag_acc = float(df["graphrag_correct"].mean() * 100) if len(df) else 0.0
    flatrag_acc = float(df["flatrag_correct"].mean() * 100) if len(df) else 0.0
    graphrag_mh_acc = float(df_mh["graphrag_correct"].mean() * 100) if len(df_mh) else 0.0
    flatrag_mh_acc = float(df_mh["flatrag_correct"].mean() * 100) if len(df_mh) else 0.0
    graphrag_simple_acc = float(df_s["graphrag_correct"].mean() * 100) if len(df_s) else 0.0
    flatrag_simple_acc = float(df_s["flatrag_correct"].mean() * 100) if len(df_s) else 0.0

    wins = df[(df["graphrag_correct"]) & (~df["flatrag_correct"])]
    losses = df[(~df["graphrag_correct"]) & (df["flatrag_correct"])]

    return {
        "run_name": run_name,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "question_set": question_set_label,
        "seeds": seeds,
        "question_count": int(len(df)),
        "metrics": {
            "overall_accuracy": {
                "graphrag": round(graphrag_acc, 2),
                "flatrag": round(flatrag_acc, 2),
                "delta": round(graphrag_acc - flatrag_acc, 2),
            },
            "multi_hop_accuracy": {
                "graphrag": round(graphrag_mh_acc, 2),
                "flatrag": round(flatrag_mh_acc, 2),
                "delta": round(graphrag_mh_acc - flatrag_mh_acc, 2),
            },
            "simple_accuracy": {
                "graphrag": round(graphrag_simple_acc, 2),
                "flatrag": round(flatrag_simple_acc, 2),
                "delta": round(graphrag_simple_acc - flatrag_simple_acc, 2),
            },
            "avg_latency_seconds": {
                "graphrag": round(float(df["graphrag_latency"].mean()), 3),
                "flatrag": round(float(df["flatrag_latency"].mean()), 3),
            },
            "query_cost_usd": {
                "graphrag": round(float(df["graphrag_cost"].sum()), 6),
                "flatrag": round(float(df["flatrag_cost"].sum()), 6),
            },
        },
        "head_to_head": {
            "graphrag_wins_count": int(len(wins)),
            "flatrag_wins_count": int(len(losses)),
            "graphrag_wins_ids": [int(v) for v in wins["id"].tolist()],
            "flatrag_wins_ids": [int(v) for v in losses["id"].tolist()],
        },
    }


def save_run_artifacts(
    *,
    output_dir: Path,
    summary: dict,
    metadata: dict | None = None,
) -> None:
    """Persist JSON summary/metadata files for a benchmark run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if metadata is not None:
        metadata_path = output_dir / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
