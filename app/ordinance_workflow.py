"""
Shared ordinance scoring workflow used by the Streamlit app.

Keeps defaults, criteria loading/sorting, PDF→chunks, scoring, and report
shape in one place so tests can cover orchestration without running Streamlit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from app.scorer import OrdinanceScorer
from app.utils import chunk_text, extract_text_from_pdf

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CRITERIA_PATH = Path("app/criteria.json")


def load_criteria_bundle(
    criteria_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Load criteria JSON and return (raw dict, list sorted by serial_number).

    The sorted list is what OrdinanceScorer and the UI use for stable order.
    """
    path = criteria_path or DEFAULT_CRITERIA_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    criteria_list = sorted(
        data["criteria"],
        key=lambda c: c.get("serial_number", 0),
    )
    return data, criteria_list


def extract_text_and_chunk(
    file_obj: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[str, list[str]]:
    """
    PDF → raw text → chunks using the same parameters as the Streamlit app.

    Returns:
        (raw_text, chunks). If extraction is empty/whitespace-only, chunks is [].
    """
    raw_text = extract_text_from_pdf(file_obj)
    if not raw_text.strip():
        return raw_text, []
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    return raw_text, chunks


def run_scoring(
    chunks: list[str],
    criteria_list: list[dict[str, Any]],
    top_k: int,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """
    Embed chunks and criterion descriptions, then score (OrdinanceScorer.score).
    """
    scorer = OrdinanceScorer(criteria=criteria_list, model_name=model_name)
    doc_embeddings = scorer.embed_texts(chunks)
    crit_embeddings = scorer.embed_texts([c["description"] for c in criteria_list])
    return scorer.score(
        doc_chunks=chunks,
        doc_embeddings=doc_embeddings,
        crit_embeddings=crit_embeddings,
        top_k=top_k,
    )


def build_score_report(
    results: dict[str, Any],
    *,
    num_chunks: int,
    top_k: int,
    timestamp_utc_iso: str | None = None,
) -> dict[str, Any]:
    """
    JSON structure for the Streamlit download (and tests).

    Meta keys must stay stable for API consumers.
    """
    if timestamp_utc_iso is None:
        timestamp_utc_iso = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    return {
        "meta": {
            "timestamp": timestamp_utc_iso,
            "overall_score": results["overall_score"],
            "num_chunks": num_chunks,
            "model": DEFAULT_MODEL_NAME,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "top_k": top_k,
        },
        "criteria_results": results["criteria_results"],
    }
