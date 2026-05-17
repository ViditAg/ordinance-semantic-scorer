"""
Semantic scoring for ordinance text against weighted criteria.

This module ties together:

* **Embeddings** — local Sentence-Transformers models (cached per model name).
* **Scoring** — per-criterion, each **probe** (short rubric phrase; falls back to
  ``title`` only if ``probes`` is missing or empty) gets max cosine similarity to document
  chunks; probe scores are averaged (mapped 0–100 per probe, then mean). Chunks
  are ranked for evidence by max similarity over probes at that chunk.

**Mathematical sketch**

For each criterion, let probe vectors be :math:`\\mathbf{p}_j` (unit-normalized)
and chunk vectors :math:`\\mathbf{d}_i`. For each probe :math:`j`, define
:math:`m_j = \\max_i \\cos(\\mathbf{p}_j, \\mathbf{d}_i)`. The criterion score is
the mean of :math:`100 \\times \\max(0, m_j)` over probes (one probe ⇒ same as
the legacy single-vector max). The **overall score** is the weighted average of
criterion scores using renormalized ``weight`` values.

**Typical usage**

.. code-block:: python

    scorer = OrdinanceScorer(criteria, model_name="all-MiniLM-L6-v2")
    doc_embs = scorer.embed_texts(chunks)
    crit_probe_embs = scorer.embed_criteria_probes()
    result = scorer.score(
        doc_chunks=chunks,
        doc_embeddings=doc_embs,
        crit_probe_embeddings=crit_probe_embs,
        top_k=2,
    )

**Testing note**

:class:`OrdinanceScorer` can be constructed and :meth:`score` can be called with
hand-crafted embedding lists so unit tests never need to download model weights.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from app.defaults import DEFAULT_SENTENCE_TRANSFORMER_MODEL
from app.utils import similarity_to_score


def criterion_probe_texts(criterion: Dict[str, Any]) -> List[str]:
    """
    Return the list of embeddable probe strings for one rubric row.

    Uses ``criterion["probes"]`` when it is a non-empty list of strings (after
    stripping empties). If there are no usable probes, falls back to a single
    embedding of the criterion ``title`` so a malformed row still runs.
    """
    probes = criterion.get("probes")
    if isinstance(probes, list) and len(probes) > 0:
        out = [str(p).strip() for p in probes if str(p).strip()]
        if out:
            return out
    title = (criterion.get("title") or "criterion").strip()
    return [title if title else "criterion"]


def criterion_short_preview(criterion: Dict[str, Any], max_len: int = 200) -> str:
    """Compact line for APIs: explicit ``short``, else truncated joined probes."""
    s = criterion.get("short")
    if isinstance(s, str) and s.strip():
        return s.strip()[:max_len]
    parts = criterion_probe_texts(criterion)
    joined = "; ".join(parts)
    return joined[:max_len] if len(joined) <= max_len else joined[: max_len - 1] + "…"


def criterion_probe_counts(criteria: List[Dict[str, Any]]) -> List[int]:
    """Number of probes per criterion, aligned with ``criteria`` order."""
    return [len(criterion_probe_texts(c)) for c in criteria]


def unpack_probe_embeddings(
    flat_embeddings: List[List[float]],
    probe_counts: List[int],
) -> List[List[List[float]]]:
    """
    Split a single batched embedding list into per-criterion probe groups.

    ``sum(probe_counts)`` must equal ``len(flat_embeddings)``.
    """
    out: List[List[List[float]]] = []
    i = 0
    for n in probe_counts:
        out.append(flat_embeddings[i : i + n])
        i += n
    if i != len(flat_embeddings):
        raise ValueError(
            "flat_embeddings length does not match sum(probe_counts): "
            f"{len(flat_embeddings)} vs consumed {i}"
        )
    return out


@lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str):
    """
    Load (or reuse) a ``SentenceTransformer`` model by Hugging Face identifier.

    **Caching**

    ``lru_cache`` memoizes the last few distinct ``model_name`` strings used in
    this process. That matters for Streamlit reruns and notebooks where users flip
    between a small set of models: we avoid reloading weights from disk each time.

    **Args:**
        model_name: Any id accepted by ``sentence_transformers.SentenceTransformer``,
            e.g. ``"all-MiniLM-L6-v2"``.

    **Returns:**
        A loaded ``SentenceTransformer`` instance ready for ``.encode``.

    **Implementation detail:**
        The import is deferred inside this function so that merely importing
        ``app.scorer`` in lightweight environments does not require
        ``sentence_transformers`` unless embeddings are actually computed.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class OrdinanceScorer:
    """
    Embed ordinance and criterion text locally, then score by semantic similarity.

    **Criteria dict shape**

    Each criterion is a ``dict`` with:

    * ``title`` — short label for UI / reports.
    * ``probes`` — list of short strings; each is embedded separately and the
      criterion score is the **mean** of per-probe max-similarity scores (0–100).
      If missing or empty, ``title`` is embedded as a single probe.
    * ``short`` — optional trimmed line for compact tables; otherwise derived from
      joined probes.
    * ``weight`` — optional relative importance (defaults to ``1.0`` if omitted);
      values renormalize **globally** across all criteria for the overall score.
    * ``pillar`` — optional grouping label for UIs (ignored by scoring math).

    **Overall score**

    Per-criterion scores live on a 0–100 scale. The overall score is
    :math:`\\sum_k w_k \\, \\text{score}_k` where ``w_k`` is each criterion's
    ``weight`` renormalized so the weights sum to 1.

    **Lazy model loading**

    The transformer is not loaded in ``__init__``. It is instantiated on first
    access to :attr:`model` / first call to :meth:`embed_texts`, which keeps
    import and test startup fast.
    """

    def __init__(
        self,
        criteria: List[Dict[str, Any]],
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    ) -> None:
        """
        Store the rubric and prepare normalized weights for aggregation.

        **Args:**
            criteria: Ordered list of criterion dicts (see class docstring).
            model_name: Sentence-Transformers checkpoint used when embedding
                (defaults to :data:`~app.defaults.DEFAULT_SENTENCE_TRANSFORMER_MODEL`).

        **Side effects:**
            Computes ``self.weights`` — positive floats summing to 1 (unless the
            total raw weight is 0, in which case all normalized weights are 0;
            see unit tests for that edge case).
        """
        self.criteria = criteria
        self.model_name = model_name
        self._model = None  # Populated lazily; see :property:`model`.

        # Renormalize explicit weights so overall_score is a weighted average.
        weights = [c.get("weight", 1.0) for c in criteria]
        total = sum(weights) or 1.0
        self.weights = [w / total for w in weights]

    @property
    def model(self):
        """
        Return the cached ``SentenceTransformer`` for this instance.

        **Why a property?**

        Callers use ``self.model.encode(...)`` without caring about lazy init.
        The first access triggers ``_get_sentence_transformer``; later accesses
        reuse ``self._model``.
        """
        if self._model is None:
            self._model = _get_sentence_transformer(self.model_name)
        return self._model

    def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Encode a batch of strings into L2-normalized dense vectors.

        **Normalization**

        ``normalize_embeddings=True`` ensures cosine similarity equals the dot
        product, which simplifies the scoring path to a single matrix multiply.

        **Args:**
            texts: Strings to embed in batch (order preserved).

        **Returns:**
            Python list of float lists (one inner list per input string). Empty
            input returns ``[]`` without touching the model.

        **Note:**
            Embeddings are converted from NumPy rows to nested lists for JSON
            serialization and for callers that prefer plain Python scalars.
        """
        if not texts:
            return []
        embs = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embs]

    def embed_criteria_probes(self) -> List[List[List[float]]]:
        """
        Embed all criterion probes in one batch, grouped per rubric row.

        Returns:
            List aligned with ``self.criteria``; each element is a non-empty list
            of probe embedding vectors (each vector is a ``list[float]``).
        """
        texts: List[str] = []
        for c in self.criteria:
            texts.extend(criterion_probe_texts(c))
        flat = self.embed_texts(texts)
        counts = criterion_probe_counts(self.criteria)
        return unpack_probe_embeddings(flat, counts)

    def score(
        self,
        doc_chunks: List[str],
        doc_embeddings: List[List[float]],
        crit_probe_embeddings: List[List[List[float]]],
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """
        Score pre-embedded document chunks against pre-embedded criterion probes.

        **Per criterion**

        1. Stack probe embeddings ``P`` (shape ``n_probes × dim``), document ``D``
           (``n_chunks × dim``). With L2-normalized rows, ``S = D @ P.T`` holds
           cosine similarities ``S[i, j]``.
        2. Per probe ``j``, ``m_j = max_i S[i, j]``. Criterion ``score`` is the
           mean of :func:`~app.utils.similarity_to_score` applied to each ``m_j``.
        3. Per chunk ``i``, ``aggregate_i = max_j S[i, j]`` ranks evidence excerpts
           (best chunk for *any* probe).
        4. ``raw_similarity`` is the mean of ``m_j`` (average best cosine per probe).

        **Args:**
            doc_chunks: Plain-text segments aligned index-wise with ``doc_embeddings``.
            doc_embeddings: Shape ``(num_chunks, dim)`` as nested lists.
            crit_probe_embeddings: Same length as ``self.criteria``; each entry is a
                non-empty list of probe vectors for that criterion.
            top_k: Number of evidence snippets to retain per criterion (>= 1).

        **Returns:**
            Dict with:

            * ``overall_score`` — float in ``[0, 100]`` (weighted mean).
            * ``criteria_results`` — list of dicts with keys including ``title``,
              ``short``, ``score``, ``raw_similarity``, ``probe_count``,
              ``top_excerpts``, ``top_scores``, ``weight``, and ``pillar`` when the
              rubric defines it.

        **Preconditions:**

        ``len(doc_chunks) == len(doc_embeddings)`` and
        ``len(crit_probe_embeddings) == len(self.criteria)`` should hold; the code
        does not assert this explicitly for performance, but violations will yield
        incorrect indexing or shape errors inside NumPy.
        """
        D = np.array(doc_embeddings, dtype=float)

        criteria_results: List[Dict[str, Any]] = []
        per_scores: List[float] = []

        for idx, crit in enumerate(self.criteria):
            probes = crit_probe_embeddings[idx]
            if not probes:
                raise ValueError(f"criterion {idx} has no probe embeddings")
            P = np.array(probes, dtype=float)
            # Cosine similarity matrix: chunks × probes (rows unit-normalized).
            sims = D @ P.T
            per_probe_max = np.max(sims, axis=0)
            chunk_aggregate = np.max(sims, axis=1)

            top_idx = chunk_aggregate.argsort()[::-1][:top_k]
            top_excerpts = [doc_chunks[i] for i in top_idx]
            top_scores = chunk_aggregate[top_idx].tolist()

            probe_scores = [similarity_to_score(float(x)) for x in per_probe_max]
            score = float(sum(probe_scores) / len(probe_scores))
            raw_sim = float(np.mean(per_probe_max))
            per_scores.append(score)

            row: Dict[str, Any] = {
                "title": crit.get("title", f"Criterion {idx+1}"),
                "short": criterion_short_preview(crit),
                "score": score,
                "raw_similarity": raw_sim,
                "probe_count": len(probes),
                "top_excerpts": top_excerpts,
                "top_scores": top_scores,
                "weight": crit.get("weight", 1.0),
            }
            if "pillar" in crit and crit["pillar"] is not None:
                row["pillar"] = crit["pillar"]
            criteria_results.append(row)

        overall = 0.0
        for w, s in zip(self.weights, per_scores):
            overall += w * s

        return {
            "overall_score": overall,
            "criteria_results": criteria_results,
        }
