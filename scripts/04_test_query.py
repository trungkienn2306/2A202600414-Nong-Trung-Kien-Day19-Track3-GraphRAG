"""Step 4 — Interactive test: compare GraphRAG vs Flat RAG on sample queries.

Usage:
    python scripts/04_test_query.py
    python scripts/04_test_query.py --question "Who founded Anthropic?"
"""
import sys
import argparse
import functools
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.graphrag as graphrag_mod
import src.flatrag  as flatrag_mod
from src.build_graph   import load_node_embeddings
from src.fetch_corpus  import load_chunks

DEFAULT_QUESTIONS = [
    "Which AI companies were co-founded by former Google employees?",
    "What is the relationship between OpenAI and Microsoft?",
    "Who founded Anthropic and where did they come from?",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Test GraphRAG vs Flat RAG")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to test (default: run 3 sample questions)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seed nodes for GraphRAG (default: 3)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 4: Interactive Query Test")
    print("=" * 55)

    # Load persisted data
    print("Loading node embeddings...")
    try:
        node_embeddings = load_node_embeddings()
    except Exception as exc:
        logging.exception("Failed loading node embeddings: %s", exc)
        sys.exit(1)
    print(f"  {len(node_embeddings)} node embeddings loaded")

    print("Building Flat RAG index (ChromaDB)...")
    try:
        chunks = load_chunks()
        flat_col = flatrag_mod.build_index(chunks)
    except Exception as exc:
        logging.exception("Failed building FlatRAG index: %s", exc)
        sys.exit(1)

    # Bind arguments
    gr_fn = functools.partial(graphrag_mod.query, node_embeddings=node_embeddings, top_k_seeds=args.seeds)
    fr_fn = functools.partial(flatrag_mod.query,  collection=flat_col)

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    for q in questions:
        print("\n" + "=" * 70)
        print(f"  Q: {q}")
        print("=" * 70)

        try:
            gr = gr_fn(q)
            fr = fr_fn(q)
        except Exception as exc:
            logging.exception("Query failed for question '%s': %s", q, exc)
            sys.exit(1)

        print(f"\n[GraphRAG]  seeds={gr['seeds']}  subgraph={gr['subgraph_size']} triples  latency={gr['latency']}s")
        print(f"  {gr['answer']}")

        print(f"\n[Flat RAG]  chunks={fr['chunks']}  latency={fr['latency']}s")
        print(f"  {fr['answer']}")


if __name__ == "__main__":
    main()
