"""Step 3 — Build Neo4j knowledge graph + node embeddings.

Saves to:
  data/processed/node_embeddings.json   (for Python-side seed search)
  Neo4j: Entity nodes + RELATION edges + .embedding property
"""
import json
import logging
import time

import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase

from src.config import (
    OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBED_MODEL, DATA_PROCESSED,
)

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return resp.data[0].embedding


def check_connection() -> bool:
    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        log.error("Neo4j unreachable: %s", e)
        log.info("Run: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return False


# ── Graph construction ─────────────────────────────────────────

def clear_graph() -> None:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    log.info("Graph cleared")


def _load_batch(tx, batch: list[dict]) -> None:
    for t in batch:
        tx.run(
            "MERGE (s:Entity {name: $subject}) "
            "ON CREATE SET s.source = $source "
            "MERGE (o:Entity {name: $object}) "
            "MERGE (s)-[r:RELATION {type: $predicate}]->(o)",
            subject=t["subject"], object=t["object"],
            predicate=t["predicate"], source=t["source"],
        )


def load_triples_to_neo4j(triples: list[dict], batch_size: int = 50) -> tuple[int, int]:
    """Batch-insert triples. Returns (node_count, edge_count)."""
    log.info("Loading %d triples into Neo4j...", len(triples))
    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]
        try:
            with driver.session() as s:
                s.execute_write(_load_batch, batch)
        except Exception as exc:
            log.exception(
                "Failed loading triple batch ending at %d/%d: %s",
                min(i + batch_size, len(triples)),
                len(triples),
                exc,
            )
            raise
        log.info("Loaded %d/%d triples", min(i + batch_size, len(triples)), len(triples))

    with driver.session() as s:
        nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edges = s.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]

    log.info("Graph loaded: %d nodes, %d edges", nodes, edges)
    return nodes, edges


# ── Embeddings ─────────────────────────────────────────────────

def _store_emb(tx, name: str, emb: list[float]) -> None:
    tx.run(
        "MATCH (n:Entity {name: $name}) SET n.embedding = $emb",
        name=name, emb=emb,
    )


def generate_and_store_embeddings() -> dict[str, list[float]]:
    """Embed every node name, store in Neo4j and return as {name: embedding}."""
    with driver.session() as s:
        node_names = [r["name"] for r in s.run("MATCH (n:Entity) RETURN n.name AS name").data()]

    log.info("Generating embeddings for %d nodes...", len(node_names))
    node_embeddings: dict[str, list[float]] = {}

    for i, name in enumerate(node_names):
        log.info("Embedding node %d/%d: %s", i + 1, len(node_names), name)
        try:
            emb = get_embedding(name)
        except Exception as exc:
            log.exception("Failed embedding node '%s': %s", name, exc)
            raise
        node_embeddings[name] = emb
        try:
            with driver.session() as s:
                s.execute_write(_store_emb, name, emb)
        except Exception as exc:
            log.exception("Failed storing embedding for node '%s': %s", name, exc)
            raise
        if (i + 1) % 25 == 0 or (i + 1) == len(node_names):
            log.info("Embedded %d/%d nodes", i + 1, len(node_names))
        time.sleep(0.02)

    # Cache to disk for reuse across sessions
    out = DATA_PROCESSED / "node_embeddings.json"
    out.write_text(json.dumps(node_embeddings, indent=2), encoding="utf-8")
    log.info("Embeddings saved -> %s", out.name)
    return node_embeddings


# ── Vector index ───────────────────────────────────────────────

def create_vector_index() -> None:
    """Create Neo4j vector index (requires Neo4j 5.11+)."""
    query = (
        "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
        "FOR (n:Entity) ON (n.embedding) "
        "OPTIONS {indexConfig: {"
        "`vector.dimensions`: 1536, "
        "`vector.similarity_function`: 'cosine'"
        "}}"
    )
    try:
        with driver.session() as s:
            s.run(query)
        log.info("Vector index created (Neo4j 5.11+)")
    except Exception as e:
        log.warning("Vector index skipped (Python cosine fallback active): %s", e)


# ── Public entry point ─────────────────────────────────────────

def build_full_graph(triples: list[dict]) -> dict[str, list[float]]:
    """Full pipeline: clear -> load -> embed -> index."""
    if not check_connection():
        raise RuntimeError("Neo4j not reachable")
    clear_graph()
    load_triples_to_neo4j(triples)
    node_embeddings = generate_and_store_embeddings()
    create_vector_index()
    return node_embeddings


# ── Load helpers ───────────────────────────────────────────────

def load_node_embeddings() -> dict[str, list[float]]:
    return json.loads((DATA_PROCESSED / "node_embeddings.json").read_text(encoding="utf-8"))


def get_graph_stats() -> dict:
    with driver.session() as s:
        nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edges = s.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
        by_company = s.run(
            "MATCH (n:Entity) RETURN n.source AS company, count(n) AS nodes ORDER BY nodes DESC"
        ).data()
    return {"nodes": nodes, "edges": edges, "by_company": by_company}
