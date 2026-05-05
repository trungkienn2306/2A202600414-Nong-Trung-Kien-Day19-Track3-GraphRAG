"""Step 4 — GraphRAG retrieval pipeline.

  query
    -> embed (text-embedding-3-small)
    -> cosine similarity -> top-K seed nodes
    -> BFS 2-hop via Cypher
    -> textualize subgraph
    -> GPT-4o-mini generates answer
"""
import time
import re

import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase

from src.config import (
    OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_MODEL, EMBED_MODEL,
)

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Utilities ──────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return resp.data[0].embedding


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (norm + 1e-9))


GENERIC_ENTITY_PATTERNS = (
    "other",
    "some ",
    "various",
    "first employees",
    "team",
    "people",
    "institutions",
    "companies",
    "organization",
    "organizations",
    "services",
    "abilities",
    "mission",
    "goal",
    "goals",
)

RELATION_WEIGHTS = {
    "FOUNDED_BY": 1.3,
    "LEFT_FOR": 1.25,
    "JOINED": 1.2,
    "POSITION_HELD": 1.2,
    "ACQUIRED": 1.15,
    "MERGED_WITH": 1.15,
    "INVESTED_IN": 1.15,
    "PARTNERED_WITH": 1.1,
    "REBRANDED_TO": 1.2,
}

LOW_CONFIDENCE_MARKERS = (
    "insufficient",
    "not enough context",
    "cannot determine",
    "can't determine",
    "unclear",
    "không đủ",
    "không thể",
)


def _canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_generic_entity(name: str) -> bool:
    lowered = name.lower().strip()
    if len(lowered) < 3:
        return True
    if lowered.isdigit():
        return True
    if lowered in {"none", "unknown", "n/a"}:
        return True
    return any(pat in lowered for pat in GENERIC_ENTITY_PATTERNS)


def _select_seeds(candidates: list[str], top_k: int) -> list[str]:
    selected: list[str] = []
    seen = set()

    # Pass 1: prefer non-generic seeds
    for name in candidates:
        key = _canonical_key(name)
        if key in seen or _is_generic_entity(name):
            continue
        seen.add(key)
        selected.append(name)
        if len(selected) >= top_k:
            return selected

    # Pass 2 fallback: allow generic seeds only if needed
    for name in candidates:
        key = _canonical_key(name)
        if key in seen:
            continue
        seen.add(key)
        selected.append(name)
        if len(selected) >= top_k:
            break
    return selected


def _seed_quality_bonus(name: str) -> float:
    has_year = any(ch.isdigit() for ch in name)
    title_case_tokens = [t for t in name.split() if t[:1].isupper()]
    if has_year:
        return 0.15
    if len(title_case_tokens) >= 2:
        return 0.1
    return 0.0


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _lexical_overlap_score(question: str, candidate: str) -> float:
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(candidate)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(c_tokens))
    return overlap / max(1, len(c_tokens))


# ── Seed finding ───────────────────────────────────────────────

def find_seed_nodes(
    question: str,
    query_emb: list[float],
    node_embeddings: dict[str, list[float]],
    top_k: int = 3,
) -> list[str]:
    """Find top-k seed nodes.
    Tries Neo4j vector index first; falls back to Python cosine similarity.
    """
    # Try Neo4j vector index (Neo4j 5.11+)
    try:
        with driver.session() as s:
            results = s.run(
                "CALL db.index.vector.queryNodes('entity_embedding', $k, $emb) "
                "YIELD node, score RETURN node.name AS name, score",
                k=max(top_k * 8, 24), emb=query_emb,
            ).data()
        if results:
            reranked = sorted(
                results,
                key=lambda r: (
                    float(r.get("score", 0.0))
                    + 0.35 * _lexical_overlap_score(question, r["name"])
                    + _seed_quality_bonus(r["name"])
                ),
                reverse=True,
            )
            ordered = [r["name"] for r in reranked]
            return _select_seeds(ordered, top_k)
    except Exception:
        pass

    # Python cosine similarity fallback
    scored = [
        (
            name,
            cosine_sim(query_emb, emb)
            + 0.35 * _lexical_overlap_score(question, name)
            + _seed_quality_bonus(name),
        )
        for name, emb in node_embeddings.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    ordered = [name for name, _ in scored[: max(top_k * 3, top_k)]]
    return _select_seeds(ordered, top_k)


# ── BFS traversal ──────────────────────────────────────────────

def _fact_score(rel_type: str, hop: int) -> float:
    rel_weight = RELATION_WEIGHTS.get(rel_type, 1.0)
    hop_penalty = 1.0 if hop == 1 else 0.87
    return rel_weight * hop_penalty


def bfs_subgraph(seed_names: list[str], max_triples: int = 80) -> list[str]:
    """BFS 2-hop Cypher traversal from seed nodes.
    Returns a deduplicated list of human-readable triple strings.
    """
    scored_facts: list[tuple[float, str]] = []

    with driver.session() as s:
        for seed in seed_names:
            # 1-hop outgoing
            out_1 = s.run(
                "MATCH (a:Entity {name: $seed})-[r:RELATION]->(b:Entity) "
                "RETURN a.name AS s, r.type AS p, b.name AS o LIMIT 25",
                seed=seed,
            ).data()
            # 1-hop incoming
            in_1 = s.run(
                "MATCH (a:Entity)-[r:RELATION]->(b:Entity {name: $seed}) "
                "RETURN a.name AS s, r.type AS p, b.name AS o LIMIT 15",
                seed=seed,
            ).data()
            # 2-hop outgoing
            out_2 = s.run(
                "MATCH (a:Entity {name: $seed})-[r1:RELATION]->(m:Entity)"
                "-[r2:RELATION]->(b:Entity) "
                "RETURN a.name AS s, r1.type AS p1, m.name AS m, r2.type AS p2, b.name AS o "
                "LIMIT 20",
                seed=seed,
            ).data()

            for r in out_1 + in_1:
                fact = f"{r['s']} --[{r['p']}]--> {r['o']}"
                scored_facts.append((_fact_score(r["p"], hop=1), fact))
            for r in out_2:
                fact = f"{r['s']} --[{r['p1']}]--> {r['m']} --[{r['p2']}]--> {r['o']}"
                score = (_fact_score(r["p1"], hop=1) + _fact_score(r["p2"], hop=2)) / 2
                scored_facts.append((score, fact))

    # Deduplicate by fact; keep best score
    best_by_fact: dict[str, float] = {}
    for score, fact in scored_facts:
        prev = best_by_fact.get(fact)
        if prev is None or score > prev:
            best_by_fact[fact] = score

    ranked = sorted(best_by_fact.items(), key=lambda kv: kv[1], reverse=True)
    return [fact for fact, _ in ranked[:max_triples]]


def _format_fact_blocks(facts: list[str]) -> str:
    if not facts:
        return "(no facts)"
    lines = []
    for i, fact in enumerate(facts, start=1):
        lines.append(f"[F{i:02d}] {fact}")
    return "\n".join(lines)


def _is_low_confidence_answer(answer: str) -> bool:
    lowered = answer.lower()
    if len(answer.strip()) < 18:
        return True
    return any(marker in lowered for marker in LOW_CONFIDENCE_MARKERS)


# ── Main query function ─────────────────────────────────────────

def query(
    question: str,
    node_embeddings: dict[str, list[float]],
    top_k_seeds: int = 3,
) -> dict:
    """Full GraphRAG pipeline. Returns answer dict with metadata."""
    t0 = time.time()

    q_emb    = get_embedding(question)
    seeds    = find_seed_nodes(question, q_emb, node_embeddings, top_k=top_k_seeds)
    subgraph = bfs_subgraph(seeds)

    context = (
        "Seed entities: " + ", ".join(seeds) + "\n\n"
        "Knowledge graph fact blocks:\n" + _format_fact_blocks(subgraph)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer questions based on the knowledge graph context. "
                "Return final answer with required target entity names and one short reasoning chain. "
                "Do not omit key entities if evidence exists."
            ),
        },
        {
            "role": "user",
            "content": "Context:\n" + context + "\n\nQuestion: " + question,
        },
    ]
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=300, temperature=0
    )
    answer = resp.choices[0].message.content
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens

    # Lightweight bounded retry: only for low-confidence answers.
    if _is_low_confidence_answer(answer):
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Given the same context, provide the most specific final answer possible. "
                    "Must include target entities when available."
                ),
            },
            {"role": "user", "content": "Context:\n" + context + "\n\nQuestion: " + question},
        ]
        retry_resp = client.chat.completions.create(
            model=LLM_MODEL, messages=retry_messages, max_tokens=220, temperature=0
        )
        retry_answer = retry_resp.choices[0].message.content
        prompt_tokens += retry_resp.usage.prompt_tokens
        completion_tokens += retry_resp.usage.completion_tokens
        if len(retry_answer.strip()) >= len(answer.strip()):
            answer = retry_answer

    return {
        "answer":        answer,
        "seeds":         seeds,
        "subgraph_size": len(subgraph),
        "tokens":        prompt_tokens + completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency":       round(time.time() - t0, 3),
    }
