"""Step 2 — Extract (subject, predicate, object) triples via LLM.

Usage:
    python scripts/02_extract.py
    python scripts/02_extract.py --skip-if-exists

Outputs:
    data/processed/triples.json
    data/processed/extraction_stats.json
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROCESSED
from src.extract_triples import extract_all, load_triples, load_stats
from src.fetch_corpus import load_chunks


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Extract knowledge triples")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip extraction if triples.json already exists")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 2: Triple Extraction (LLM-based NER)")
    print("=" * 55)

    triples_file = DATA_PROCESSED / "triples.json"

    if args.skip_if_exists and triples_file.exists():
        print("[SKIP] triples.json exists — loading cached triples")
        triples = load_triples()
        stats   = load_stats()
        print(f"  Loaded {len(triples)} triples")
        print(f"  Total tokens: {stats['total_tokens']:,}")
    else:
        chunks = load_chunks()
        print(f"Extracting triples from {len(chunks)} chunks...\n")
        try:
            triples, stats = extract_all(chunks)
        except Exception as exc:
            logging.exception("Extraction failed. Stopping run: %s", exc)
            sys.exit(1)

    print("\n--- Summary ---")
    print(f"  Total triples : {len(triples)}")
    print(f"  Total tokens  : {stats['total_tokens']:,}")
    print(f"  Est. cost     : ${stats['total_tokens'] * 0.15 / 1_000_000:.4f}")
    print(f"  Saved         : {triples_file}")
    print("\nTip: inspect data/processed/triples.json to see extracted triples")


if __name__ == "__main__":
    main()
