"""Central configuration — all paths, constants, and env vars."""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CHROMA    = ROOT / "data" / "chroma"
VISUALIZATIONS = ROOT / "visualizations"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_CHROMA, VISUALIZATIONS):
    _d.mkdir(parents=True, exist_ok=True)

# ── Environment ────────────────────────────────────────────────
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Model names ────────────────────────────────────────────────
LLM_MODEL   = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536

# ── Corpus ─────────────────────────────────────────────────────
COMPANIES = [
    "OpenAI",
    "Anthropic (company)",
    "Google DeepMind",
    "Meta AI",
    "xAI (company)",
    "Mistral AI",
    "Cohere (company)",
    "Inflection AI",
    "Stability AI",
    "Hugging Face",
]

# ── Chunking ───────────────────────────────────────────────────
CHUNK_MAX_TOKENS   = 500
CHUNK_OVERLAP      = 50
CHUNKS_PER_COMPANY = 3   # how many chunks to send to LLM for triple extraction
