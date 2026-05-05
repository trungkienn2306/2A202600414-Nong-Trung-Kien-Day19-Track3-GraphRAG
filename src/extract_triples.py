"""Step 2 — LLM-based NER: extract (subject, predicate, object) triples.

Saves to:
  data/processed/triples.json
  data/processed/extraction_stats.json
"""
import json
import logging
import time

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY, LLM_MODEL,
    DATA_PROCESSED, CHUNKS_PER_COMPANY,
)

client = OpenAI(api_key=OPENAI_API_KEY)
log = logging.getLogger(__name__)

PREDICATE_ALIASES = {
    "FOUNDED": "FOUNDED_BY",
    "CO_FOUNDED": "FOUNDED_BY",
    "COFOUNDED": "FOUNDED_BY",
    "LEFT_FOR": "LEFT_FOR",
    "JOINED": "JOINED",
    "JOINED_AS": "POSITION_HELD",
    "CEO": "POSITION_HELD",
    "VP_OF_RESEARCH_AT": "POSITION_HELD",
    "POSITION_HELD_AT": "POSITION_HELD",
    "WORKED_AT": "EMPLOYEE_OF",
    "EMPLOYED_BY": "EMPLOYEE_OF",
    "ACQUIRED_BY": "ACQUIRED",
    "MERGED_INTO": "MERGED_WITH",
    "REBRANDED_AS": "REBRANDED_TO",
}

CANONICAL_ENTITY_ALIASES = {
    "fair": "Meta AI",
    "facebook ai research": "Meta AI",
    "google brain + deepmind": "Google DeepMind",
    "google brain and deepmind": "Google DeepMind",
    "google deep mind": "Google DeepMind",
}


def _canonical_entity(name: str) -> str:
    compact = " ".join(name.strip().split())
    lowered = compact.lower()
    if lowered in CANONICAL_ENTITY_ALIASES:
        return CANONICAL_ENTITY_ALIASES[lowered]
    return compact


def _canonical_predicate(pred: str) -> str:
    p = pred.strip().upper().replace(" ", "_").replace("-", "_")
    return PREDICATE_ALIASES.get(p, p)


# ── Core extraction ────────────────────────────────────────────

def extract_triples_from_chunk(text: str, company_name: str) -> tuple[list[dict], int]:
    """Call GPT-4o-mini to extract triples from one text chunk."""
    user_msg = (
        "Extract named entities and their relationships from the text below about "
        + company_name + ".\n\n"
        "Return a JSON object with key 'triples' containing an array.\n"
        "Each element must have exactly three string fields:\n"
        "  subject   - the source entity name\n"
        "  predicate - the relationship type in UPPERCASE\n"
        "  object    - the target entity name\n\n"
        "Allowed predicates (use the closest fit):\n"
        "  FOUNDED_BY, LEFT_FOR, JOINED, POSITION_HELD, CEO_OF, INVESTED_IN, DEVELOPED, ACQUIRED, LOCATED_IN,\n"
        "  PARTNERED_WITH, EMPLOYEE_OF, COMPETES_WITH, RELEASED, FUNDED_BY,\n"
        "  MERGED_WITH, PART_OF, CREATED_BY, BOARD_MEMBER_OF, REBRANDED_TO\n\n"
        "Rules:\n"
        "  - Only extract facts clearly stated in the text\n"
        "  - Max 25 triples per chunk\n"
        "  - Entity names must be proper nouns (people, companies, products, places, years)\n\n"
        "Text:\n" + text[:2500]
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert knowledge graph builder that extracts structured triples from text.",
            },
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result  = json.loads(response.choices[0].message.content)
    triples = result.get("triples", [])
    tokens  = response.usage.total_tokens
    return triples, tokens


def _deduplicate(raw: list[dict], source: str) -> list[dict]:
    """Normalise and deduplicate; attach source company."""
    seen, result = set(), []
    for t in raw:
        s = _canonical_entity(str(t.get("subject",   "")).strip())
        p = _canonical_predicate(str(t.get("predicate", "")).strip())
        o = _canonical_entity(str(t.get("object",    "")).strip())
        if s and p and o and len(s) < 120 and len(o) < 120:
            key = (s.lower(), p, o.lower())
            if key not in seen:
                seen.add(key)
                result.append({"subject": s, "predicate": p, "object": o, "source": source})
    return result


# ── Pipeline ───────────────────────────────────────────────────

def extract_all(chunks: list[dict]) -> tuple[list[dict], dict]:
    """Run extraction over all companies; returns (triples, stats)."""
    all_triples = []
    stats       = {"total_tokens": 0, "per_company": {}}

    companies = list(dict.fromkeys(c["company"] for c in chunks))  # preserve order

    for company in companies:
        company_chunks = [
            c for c in chunks if c["company"] == company
        ][:CHUNKS_PER_COMPANY]

        raw_triples, tokens_used = [], 0
        for idx, chunk in enumerate(company_chunks, start=1):
            log.info("[%s] extracting chunk %d/%d", company, idx, len(company_chunks))
            try:
                triples, tokens = extract_triples_from_chunk(chunk["text"], company)
            except Exception as exc:
                log.exception("[%s] failed on chunk %d/%d: %s", company, idx, len(company_chunks), exc)
                raise
            raw_triples.extend(triples)
            tokens_used += tokens
            time.sleep(0.3)   # rate-limit guard

        deduped = _deduplicate(raw_triples, company)
        all_triples.extend(deduped)

        stats["total_tokens"] += tokens_used
        stats["per_company"][company] = {
            "raw":       len(raw_triples),
            "unique":    len(deduped),
            "tokens":    tokens_used,
            "cost_usd":  round(tokens_used * 0.15 / 1_000_000, 6),
        }
        log.info(
            "%s: %3d raw -> %3d unique (%s tokens)",
            company,
            len(raw_triples),
            len(deduped),
            f"{tokens_used:,}",
        )

    # ── Persist ────────────────────────────────────────────────
    triples_out = DATA_PROCESSED / "triples.json"
    triples_out.write_text(
        json.dumps(all_triples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats_out = DATA_PROCESSED / "extraction_stats.json"
    stats_out.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_cost = stats["total_tokens"] * 0.15 / 1_000_000
    log.info("Total triples: %d", len(all_triples))
    log.info("Total tokens: %s", f"{stats['total_tokens']:,}")
    log.info("Est. cost: $%.4f (gpt-4o-mini)", total_cost)
    log.info("Saved -> %s", triples_out)
    return all_triples, stats


def load_triples() -> list[dict]:
    return json.loads((DATA_PROCESSED / "triples.json").read_text(encoding="utf-8"))


def load_stats() -> dict:
    return json.loads((DATA_PROCESSED / "extraction_stats.json").read_text(encoding="utf-8"))
