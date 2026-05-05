"""Step 1 — Fetch Wikipedia articles and chunk them.

Saves to:
  data/raw/<company_slug>.json          (one file per company, inspectable)
  data/processed/chunks.json            (all chunks for Flat RAG)
"""
import json
import time
import re

import wikipedia
import tiktoken

from src.config import (
    DATA_RAW, DATA_PROCESSED,
    COMPANIES, CHUNK_MAX_TOKENS, CHUNK_OVERLAP,
)


# ── Utilities ──────────────────────────────────────────────────

def slug(company: str) -> str:
    """Convert company name to safe filename slug."""
    return re.sub(r"[^a-z0-9_]", "", company.lower().replace(" ", "_"))


# ── Fetching ───────────────────────────────────────────────────

def fetch_article(company: str) -> dict | None:
    """Fetch a single Wikipedia article, trying auto-suggest if exact match fails."""
    for auto_suggest in [False, True]:
        try:
            page = wikipedia.page(company, auto_suggest=auto_suggest)
            return {
                "company": company,
                "title":   page.title,
                "text":    page.content,
                "url":     page.url,
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Pick first unambiguous option from the list
            for option in e.options[:3]:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    return {
                        "company": company,
                        "title":   page.title,
                        "text":    page.content,
                        "url":     page.url,
                    }
                except Exception:
                    continue
        except wikipedia.exceptions.PageError:
            if not auto_suggest:
                continue  # retry with auto_suggest=True
    return None


def fetch_all(companies: list[str] = COMPANIES) -> list[dict]:
    """Fetch all company articles. Each one is saved to data/raw/ as JSON."""
    wikipedia.set_lang("en")
    corpus = []

    for company in companies:
        doc = fetch_article(company)
        if doc:
            out = DATA_RAW / f"{slug(company)}.json"
            out.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [OK] {doc['title']:<40} -> {out.name}  ({len(doc['text']):,} chars)")
            corpus.append(doc)
        else:
            print(f"  [FAIL] {company}")
        time.sleep(0.5)   # be polite to Wikipedia

    print(f"\nFetched: {len(corpus)}/{len(companies)} articles")
    return corpus


def load_corpus() -> list[dict]:
    """Load corpus from data/raw/ (skips re-fetching)."""
    corpus = []
    for f in sorted(DATA_RAW.glob("*.json")):
        corpus.append(json.loads(f.read_text(encoding="utf-8")))
    return corpus


# ── Chunking ───────────────────────────────────────────────────

def chunk_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks


def chunk_corpus(corpus: list[dict]) -> list[dict]:
    """Chunk every article and save to data/processed/chunks.json."""
    all_chunks = []
    for doc in corpus:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "id":          f"{slug(doc['company'])}_{i}",
                "company":     doc["company"],
                "chunk_index": i,
                "text":        chunk,
            })

    out = DATA_PROCESSED / "chunks.json"
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_chunks)} chunks -> {out}")
    return all_chunks


def load_chunks() -> list[dict]:
    return json.loads((DATA_PROCESSED / "chunks.json").read_text(encoding="utf-8"))
