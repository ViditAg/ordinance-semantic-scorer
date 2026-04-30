"""Dataclasses for scoring use-case inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ScoringRequest:
    """Parameters for one scoring run (independent of document ingestion)."""

    top_k: int = 1


@dataclass(frozen=True)
class ScoringResult:
    """Outcome of embedding + scoring a fixed list of document chunks."""

    chunks: List[str]
    score_payload: Dict[str, Any]
    """
    Same shape as :meth:`app.scorer.OrdinanceScorer.score` return value:
    ``overall_score`` and ``criteria_results``.
    """
