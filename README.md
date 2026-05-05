# Lab 19: GraphRAG on Tech Company Corpus

Build a Knowledge Graph from 10 Wikipedia AI-company articles, implement GraphRAG
multi-hop retrieval, and benchmark it against Flat RAG on 20 questions.

## Project Structure

```text
Day19-GraphRAG/
│
├── data/                          ← auto-generated (gitignored)
│   ├── raw/                       ← one JSON per Wikipedia article  ← inspect here
│   │   ├── openai.json
│   │   ├── anthropic_company.json
│   │   └── ...
│   └── processed/
│       ├── chunks.json            ← chunked text for Flat RAG
│       ├── triples.json           ← extracted (subject, predicate, object) triples
│       ├── node_embeddings.json   ← node embedding cache (reused across sessions)
│       └── extraction_stats.json  ← token usage / cost per company
│
├── src/                           ← core Python modules
│   ├── config.py                  ← paths, env vars, constants
│   ├── fetch_corpus.py            ← Wikipedia fetch + chunking
│   ├── extract_triples.py         ← LLM-based NER (GPT-4o-mini)
│   ├── build_graph.py             ← Neo4j graph + embeddings
│   ├── graphrag.py                ← GraphRAG retrieval pipeline
│   ├── flatrag.py                 ← Flat RAG baseline (ChromaDB)
│   └── benchmark.py               ← 20-question benchmark + LLM judge
│
├── scripts/                       ← runnable from terminal (sequential steps)
│   ├── 01_fetch.py
│   ├── 02_extract.py
│   ├── 03_build_graph.py
│   ├── 04_test_query.py
│   └── 05_benchmark.py
│
├── visualizations/
│   ├── visualize_graph.py         ← NetworkX → PNG output
│   ├── graph_full.png             ← auto-generated
│   └── subgraph_*.png             ← auto-generated
│
├── report/
│   ├── README.md                  ← benchmark-report workflow + commands
│   ├── benchmark_report.md        ← official markdown report template
│   └── benchmark_runs/            ← per-run CSV/JSON artifacts
│
├── graphrag_lab19.ipynb           ← lightweight notebook orchestrator + EDA
├── benchmark_results.csv          ← auto-generated
├── requirements.txt
├── .env                           ← your API keys (gitignored)
└── .env.example
```

## Prerequisites

### 1. Start Neo4j

```bash
docker compose up -d neo4j
```

Neo4j Browser UI: [http://localhost:7474](http://localhost:7474)

You can still run Neo4j manually without compose:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Run Steps (Terminal)

```bash
# Step 1 — Fetch 10 Wikipedia articles (saved to data/raw/)
python scripts/01_fetch.py

# Inspect the data!
cat data/raw/openai.json | python -m json.tool | head -50

# Step 2 — Extract knowledge triples via LLM
python scripts/02_extract.py

# Inspect triples
cat data/processed/triples.json | python -m json.tool | head -80

# Step 3 — Build Neo4j graph + node embeddings
python scripts/03_build_graph.py
# -> open http://localhost:7474 and run: MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100

# Step 4 — Test queries interactively
python scripts/04_test_query.py
python scripts/04_test_query.py -q "Who founded Anthropic?"

# Step 5 — Benchmark
# Uses persistent ChromaDB cache at data/chroma (reuses index between runs)
python scripts/05_benchmark.py --question-set test --seeds 2 --run-name seeds2_test
python scripts/05_benchmark.py --question-set test --seeds 3 --run-name seeds3_test
python scripts/05_benchmark.py --question-set test --seeds 5 --run-name seeds5_test
python scripts/05_benchmark.py --question-set full --seeds 3 --run-name seeds3_full
python scripts/05_benchmark.py --dry-run   # preview questions only
python scripts/05_benchmark.py --question-set test --force-reindex  # rebuild Chroma index

# Optional: one-command orchestrator with realtime logs + fail-fast
python scripts/06_run_test_first.py --skip-fetch --seeds 2,3,5
```

Run artifacts (for official report) are saved to:

- `report/benchmark_runs/<run_name>/benchmark_details.csv`
- `report/benchmark_runs/<run_name>/summary.json`
- `report/benchmark_runs/<run_name>/run_metadata.json`

---

## Visualization

### Neo4j Browser (interactive, best for graph exploration)

Open [http://localhost:7474](http://localhost:7474) and try these Cypher queries:

```cypher
-- Full graph overview
MATCH (n:Entity)-[r]->(m) RETURN n,r,m LIMIT 100

-- OpenAI 2-hop subgraph
MATCH p=(n:Entity {name:'OpenAI'})-[*1..2]-(m) RETURN p LIMIT 80

-- Investment relationships
MATCH (s:Entity)-[r:RELATION {type:'INVESTED_IN'}]->(o)
RETURN s.name AS investor, o.name AS investee

-- Founders
MATCH (company)-[r:RELATION {type:'FOUNDED_BY'}]->(person)
RETURN company.name, person.name
```

### NetworkX PNG (static, good for reports)

```bash
# Full graph overview (top 80 nodes)
python visualizations/visualize_graph.py

# Entity subgraph
python visualizations/visualize_graph.py OpenAI
python visualizations/visualize_graph.py "Anthropic (company)" 3
python visualizations/visualize_graph.py "Sam Altman" 2
```

Outputs are saved to `visualizations/graph_full.png` and `visualizations/subgraph_*.png`.

### Notebook (inline EDA + results)

```bash
jupyter notebook graphrag_lab19.ipynb
```

The notebook is a thin orchestrator — it calls `src/` modules and displays
inline PNG visualizations and coloured benchmark tables.

---

## Benchmark Metrics

| Metric | Description |
| --- | --- |
| `graphrag_correct` | LLM judge: does GraphRAG answer match ground truth? |
| `flatrag_correct` | LLM judge: does Flat RAG answer match ground truth? |
| `graphrag_latency` | Seconds per query (embed + BFS + LLM) |
| `flatrag_latency` | Seconds per query (embed + retrieval + LLM) |
| `graphrag_tokens` | GPT-4o-mini tokens per query |
| `graphrag_seeds` | Which entities were used as BFS starting points |

---

## Companies in Corpus

OpenAI · Anthropic · Google DeepMind · Meta AI · xAI · Mistral AI ·
Cohere · Inflection AI · Stability AI · Hugging Face
