"""Step 1 — Fetch 10 Wikipedia articles and chunk them.

Usage:
    python scripts/01_fetch.py
    python scripts/01_fetch.py --skip-if-exists    # re-use cached data/raw/

Outputs:
    data/raw/<company>.json    (one readable JSON per company)
    data/processed/chunks.json
"""
import sys
import argparse
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_RAW, DATA_PROCESSED
from src.fetch_corpus import fetch_all, chunk_corpus, load_corpus, load_chunks, COMPANIES


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Fetch Wikipedia corpus")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip fetching if data/raw/ already has files")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 1: Fetch Wikipedia Corpus")
    print("=" * 55)

    raw_files = list(DATA_RAW.glob("*.json"))

    if args.skip_if_exists and len(raw_files) >= len(COMPANIES):
        print(f"[SKIP] Found {len(raw_files)} files in data/raw/ — loading cached corpus")
        corpus = load_corpus()
    else:
        print(f"Fetching {len(COMPANIES)} Wikipedia articles...")
        try:
            corpus = fetch_all()
        except Exception as exc:
            logging.exception("Fetch failed. Stopping run: %s", exc)
            sys.exit(1)

    chunks_file = DATA_PROCESSED / "chunks.json"
    if args.skip_if_exists and chunks_file.exists():
        print(f"[SKIP] chunks.json exists — loading cached chunks")
        chunks = load_chunks()
    else:
        print("\nChunking articles...")
        try:
            chunks = chunk_corpus(corpus)
        except Exception as exc:
            logging.exception("Chunking failed. Stopping run: %s", exc)
            sys.exit(1)

    print("\n--- Summary ---")
    print(f"  Articles : {len(corpus)}")
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Data raw : {DATA_RAW}")
    print(f"  Chunks   : {DATA_PROCESSED / 'chunks.json'}")
    print("\nTip: inspect any article in data/raw/<company>.json")


if __name__ == "__main__":
    main()
