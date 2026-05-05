"""Step 3 — Load triples into Neo4j + generate node embeddings.

Usage:
    python scripts/03_build_graph.py
    python scripts/03_build_graph.py --skip-embeddings   # reload graph only, reuse embeddings

Outputs:
    Neo4j: Entity nodes + RELATION edges + .embedding property
    data/processed/node_embeddings.json

Visualization:
    Neo4j Browser: http://localhost:7474
    Run: MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROCESSED
from src.build_graph import (
    check_connection, clear_graph,
    load_triples_to_neo4j, generate_and_store_embeddings,
    create_vector_index, load_node_embeddings, get_graph_stats,
)
from src.extract_triples import load_triples


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Build Neo4j knowledge graph")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip re-embedding nodes (reuse data/processed/node_embeddings.json)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 3: Build Knowledge Graph in Neo4j")
    print("=" * 55)

    if not check_connection():
        sys.exit(1)

    triples = load_triples()
    print(f"Loaded {len(triples)} triples from triples.json\n")

    try:
        clear_graph()
        nodes, edges = load_triples_to_neo4j(triples)
    except Exception as exc:
        logging.exception("Failed while loading triples into Neo4j: %s", exc)
        sys.exit(1)

    emb_file = DATA_PROCESSED / "node_embeddings.json"
    if args.skip_embeddings and emb_file.exists():
        print("\n[SKIP] Reusing existing node_embeddings.json")
        node_embeddings = load_node_embeddings()
        # Re-store embeddings into Neo4j (graph was just cleared)
        from src.build_graph import _store_emb
        from neo4j import GraphDatabase
        from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        for idx, (name, emb) in enumerate(node_embeddings.items(), start=1):
            try:
                with driver.session() as s:
                    s.execute_write(_store_emb, name, emb)
            except Exception as exc:
                logging.exception("Failed restoring cached embedding %d for '%s': %s", idx, name, exc)
                sys.exit(1)
        print(f"[OK] Re-stored {len(node_embeddings)} cached embeddings into Neo4j")
    else:
        print()
        try:
            node_embeddings = generate_and_store_embeddings()
        except Exception as exc:
            logging.exception("Failed during embedding generation: %s", exc)
            sys.exit(1)

    create_vector_index()

    stats = get_graph_stats()
    print("\n--- Graph Summary ---")
    print(f"  Nodes : {stats['nodes']}")
    print(f"  Edges : {stats['edges']}")
    print(f"  By company:")
    for row in stats["by_company"]:
        print(f"    {row['company']}: {row['nodes']} nodes")

    print("\n--- Visualization ---")
    print("  Neo4j Browser  : http://localhost:7474")
    print("  Full graph     : MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100")
    print("  OpenAI subgraph: MATCH p=(n:Entity {name:'OpenAI'})-[*1..2]-(m) RETURN p LIMIT 50")
    print("\n  Python viz     : python visualizations/visualize_graph.py")
    print("  Entity subgraph: python visualizations/visualize_graph.py OpenAI")


if __name__ == "__main__":
    main()
