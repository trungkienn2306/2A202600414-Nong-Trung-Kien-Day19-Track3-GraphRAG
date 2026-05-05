"""NetworkX + Matplotlib visualization of the knowledge graph.

Outputs PNG files to the visualizations/ folder.
Also prints useful Cypher queries for Neo4j Browser.

Usage:
    python visualizations/visualize_graph.py              # full graph overview
    python visualizations/visualize_graph.py OpenAI       # OpenAI subgraph (2-hop)
    python visualizations/visualize_graph.py Anthropic 3  # Anthropic 3-hop subgraph
"""
import sys
import json
from pathlib import Path

# Allow running from project root or from visualizations/ folder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves PNG without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import DATA_PROCESSED, VISUALIZATIONS

# ── Company colour palette ─────────────────────────────────────
COMPANY_COLORS: dict[str, str] = {
    "OpenAI":              "#10a37f",
    "Anthropic (company)": "#e06b3f",
    "Google DeepMind":     "#4285f4",
    "Meta AI":             "#0668e1",
    "xAI (company)":       "#555555",
    "Mistral AI":          "#ff7f00",
    "Cohere (company)":    "#2196f3",
    "Inflection AI":       "#9c27b0",
    "Stability AI":        "#f44336",
    "Hugging Face":        "#c8a300",
}
DEFAULT_COLOR = "#aaaaaa"


# ── Graph building ─────────────────────────────────────────────

def build_nx_graph(triples: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in triples:
        G.add_edge(
            t["subject"], t["object"],
            label=t["predicate"], source=t["source"],
        )
        # Tag node with source company (first occurrence wins)
        if "source" not in G.nodes[t["subject"]]:
            G.nodes[t["subject"]]["source"] = t["source"]
        if "source" not in G.nodes[t["object"]]:
            G.nodes[t["object"]]["source"] = t["source"]
    return G


def node_colors(G: nx.DiGraph) -> list[str]:
    return [
        COMPANY_COLORS.get(G.nodes[n].get("source", ""), DEFAULT_COLOR)
        for n in G.nodes()
    ]


# ── Full graph overview ────────────────────────────────────────

def visualize_full(triples: list[dict], max_nodes: int = 80) -> Path:
    """Draw the top-N nodes by degree and save as PNG."""
    G = build_nx_graph(triples)

    # Keep most-connected nodes for readability
    top = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    G_sub = G.subgraph(top).copy()

    fig, ax = plt.subplots(figsize=(22, 17))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(G_sub, k=2.2, seed=42, iterations=60)

    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax,
        node_color=node_colors(G_sub),
        node_size=900, alpha=0.92,
    )
    nx.draw_networkx_labels(
        G_sub, pos, ax=ax,
        font_size=6.5, font_color="white", font_weight="bold",
    )
    nx.draw_networkx_edges(
        G_sub, pos, ax=ax,
        edge_color="#888888", arrows=True,
        arrowsize=14, width=0.8, alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )

    # Legend
    patches = [
        mpatches.Patch(color=color, label=co.replace(" (company)", "").replace(" AI", " AI"))
        for co, color in COMPANY_COLORS.items()
    ]
    ax.legend(
        handles=patches, loc="upper left",
        fontsize=8, framealpha=0.3,
        labelcolor="white", facecolor="#1a1a2e",
    )

    ax.set_title(
        f"GraphRAG Knowledge Graph  —  Top {len(G_sub)} nodes by degree  "
        f"({G.number_of_nodes()} total nodes, {G.number_of_edges()} edges)",
        fontsize=13, color="white", pad=14,
    )
    ax.axis("off")
    plt.tight_layout()

    out = VISUALIZATIONS / "graph_full.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Full graph saved -> {out}")
    return out


# ── Entity subgraph ────────────────────────────────────────────

def visualize_subgraph(entity: str, triples: list[dict], hops: int = 2) -> Path | None:
    """Draw an N-hop subgraph centred on a single entity."""
    G = build_nx_graph(triples)

    if entity not in G:
        # Fuzzy match
        candidates = [n for n in G.nodes() if entity.lower() in n.lower()]
        if not candidates:
            print(f"[WARN] Entity '{entity}' not found. Available nodes (sample):")
            for n in list(G.nodes())[:20]:
                print(f"  {n}")
            return None
        entity = candidates[0]
        print(f"[INFO] Using closest match: '{entity}'")

    # BFS expansion
    neighbourhood = {entity}
    frontier = {entity}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.predecessors(n))
            next_frontier.update(G.successors(n))
        neighbourhood.update(next_frontier)
        frontier = next_frontier

    G_sub = G.subgraph(neighbourhood).copy()

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(G_sub, k=3.0, seed=42)

    colors = [
        "#ff4444" if n == entity
        else COMPANY_COLORS.get(G_sub.nodes[n].get("source", ""), DEFAULT_COLOR)
        for n in G_sub.nodes()
    ]
    sizes  = [2500 if n == entity else 900 for n in G_sub.nodes()]

    nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.92)
    nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        G_sub, pos, ax=ax, edge_color="#aaaaaa",
        arrows=True, arrowsize=16, width=1.0, alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )
    edge_labels = {(u, v): d["label"] for u, v, d in G_sub.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G_sub, pos, edge_labels=edge_labels,
        ax=ax, font_size=6.5, font_color="#dddddd",
        bbox=dict(boxstyle="round,pad=0.2", fc="#2a2a4a", alpha=0.7),
    )

    ax.set_title(
        f"Subgraph: '{entity}'  ({hops}-hop,  {len(G_sub)} nodes, {G_sub.number_of_edges()} edges)",
        fontsize=12, color="white", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    safe = entity.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    out  = VISUALIZATIONS / f"subgraph_{safe}_{hops}hop.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Subgraph saved -> {out}")
    return out


# ── Neo4j Cypher cheat-sheet ───────────────────────────────────

def print_neo4j_queries() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║  Neo4j Browser  →  http://localhost:7474                 ║
╚══════════════════════════════════════════════════════════╝

▸ Full graph (limit 100):
  MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100

▸ Subgraph around OpenAI (2-hop):
  MATCH p=(n:Entity {name:'OpenAI'})-[*1..2]-(m)
  RETURN p LIMIT 80

▸ All relationships FROM a company:
  MATCH (n:Entity {name:'Anthropic (company)'})-[r]->(m)
  RETURN n.name, r.type, m.name

▸ Who invested in whom:
  MATCH (s:Entity)-[r:RELATION {type:'INVESTED_IN'}]->(o)
  RETURN s.name AS investor, o.name AS investee

▸ Co-founder chains:
  MATCH (person)-[:RELATION {type:'FOUNDED_BY'}]-(company)
  RETURN company.name, person.name

▸ Highlight answer nodes (change color in Browser):
  MATCH (n:Entity) WHERE n.name IN ['OpenAI','Anthropic (company)']
  SET n:AnswerNode
  RETURN n
""")


# ── Main ───────────────────────────────────────────────────────

def main():
    triples_path = DATA_PROCESSED / "triples.json"
    if not triples_path.exists():
        print("[ERROR] data/processed/triples.json not found.")
        print("  Run: python scripts/02_extract.py")
        sys.exit(1)

    triples = json.loads(triples_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(triples)} triples from {triples_path.name}")

    entity = sys.argv[1] if len(sys.argv) > 1 else None
    hops   = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Always generate full overview
    visualize_full(triples)

    # Entity subgraph if requested
    if entity:
        visualize_subgraph(entity, triples, hops=hops)
    else:
        # Default: show OpenAI subgraph as example
        visualize_subgraph("OpenAI", triples, hops=2)

    print_neo4j_queries()
    print(f"\nImages saved to: {VISUALIZATIONS}")


if __name__ == "__main__":
    main()
