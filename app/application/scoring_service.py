"""
Ordinance scoring **use case** — one entry point for “text → chunks → embed → score”.

``OrdinanceScorer`` remains the embedding + matrix engine; this service wires a
:class:`~app.domain.ports.Chunker` and exposes a stable API for adapters
(Streamlit, CLI, notebooks) without duplicating orchestration logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.domain.models import ScoringRequest, ScoringResult
from app.domain.ports import Chunker, TextSource
from app.scorer import OrdinanceScorer


class OrdinanceScoringService:
    """
    Application service for semantic ordinance scoring.

    Construct with rubric criteria, model id, and a :class:`~app.domain.ports.Chunker`.
    Call :meth:`score_document` when you have a :class:`~app.domain.ports.TextSource`,
    or :meth:`score_chunks` when chunks are already computed (e.g. UI showed a preview).
    """

    def __init__(
        self,
        criteria: List[Dict[str, Any]],
        model_name: str,
        chunker: Chunker,
    ) -> None:
        self._criteria = criteria
        self._chunker = chunker
        self._scorer = OrdinanceScorer(criteria=criteria, model_name=model_name)

    def score_document(self, source: TextSource, request: ScoringRequest) -> ScoringResult:
        """
        Load text from ``source``, chunk with the configured chunker, then embed and score.
        """
        raw = source.load_plain_text()
        if not raw or not str(raw).strip():
            raise ValueError("empty document")
        chunks = self._chunker.chunk(str(raw).strip())
        return self.score_chunks(chunks, request)

    def score_chunks(self, chunks: List[str], request: ScoringRequest) -> ScoringResult:
        """
        Embed and score **precomputed** chunks (must be non-empty).

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("no chunks to score")

        doc_embeddings = self._scorer.embed_texts(chunks)
        crit_texts = [c["description"] for c in self._criteria]
        crit_embeddings = self._scorer.embed_texts(crit_texts)
        payload: Dict[str, Any] = self._scorer.score(
            doc_chunks=chunks,
            doc_embeddings=doc_embeddings,
            crit_embeddings=crit_embeddings,
            top_k=request.top_k,
        )
        return ScoringResult(chunks=chunks, score_payload=payload)
