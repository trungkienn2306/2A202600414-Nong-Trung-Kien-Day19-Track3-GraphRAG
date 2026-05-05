"""Step 5 — Flat RAG baseline using persistent ChromaDB + OpenAI embeddings."""
import time

import chromadb
from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, EMBED_MODEL, DATA_CHROMA

client = OpenAI(api_key=OPENAI_API_KEY)
_chroma = chromadb.PersistentClient(path=str(DATA_CHROMA))


# ── Helpers ────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return resp.data[0].embedding


# ── Index building ─────────────────────────────────────────────

def build_index(
    chunks: list[dict],
    batch_size: int = 20,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """Build or reuse ChromaDB index. Returns the collection."""
    existing_collection = None
    try:
        existing_collection = _chroma.get_collection("flat_rag")
    except Exception:
        existing_collection = None

    if force_rebuild and existing_collection is not None:
        _chroma.delete_collection("flat_rag")
        existing_collection = None

    if existing_collection is not None and existing_collection.count() == len(chunks):
        print(
            f"[OK] Reusing persistent ChromaDB index "
            f"({existing_collection.count()} docs) at {DATA_CHROMA}"
        )
        return existing_collection

    if existing_collection is not None:
        _chroma.delete_collection("flat_rag")

    col = _chroma.create_collection("flat_rag")
    print(f"Indexing {len(chunks)} chunks into persistent ChromaDB at {DATA_CHROMA}...")

    for i in range(0, len(chunks), batch_size):
        batch  = chunks[i : i + batch_size]
        texts  = [c["text"] for c in batch]
        resp   = client.embeddings.create(input=texts, model=EMBED_MODEL)
        embeds = [r.embedding for r in resp.data]

        col.add(
            ids        = [c["id"]      for c in batch],
            embeddings = embeds,
            documents  = texts,
            metadatas  = [{"company": c["company"]} for c in batch],
        )
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(0.1)

    print(f"[OK] ChromaDB: {col.count()} documents indexed")
    return col


# ── Query ──────────────────────────────────────────────────────

def query(
    question: str,
    collection: chromadb.Collection,
    top_k: int = 5,
) -> dict:
    """Flat RAG: embed -> top-k retrieval -> LLM answer."""
    t0    = time.time()
    q_emb = get_embedding(question)

    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs    = results["documents"][0]
    context = "\n\n---\n\n".join(docs)

    messages = [
        {
            "role": "system",
            "content": "Answer questions based on the provided context. Be concise and factual.",
        },
        {
            "role": "user",
            "content": "Context:\n" + context[:4000] + "\n\nQuestion: " + question,
        },
    ]
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=300, temperature=0
    )

    return {
        "answer":  resp.choices[0].message.content,
        "chunks":  len(docs),
        "tokens":  resp.usage.total_tokens,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "latency": round(time.time() - t0, 3),
    }
