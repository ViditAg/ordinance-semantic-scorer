"""
Semantic scoring for ordinance text against weighted criteria.

This module ties together:

* **Embeddings** — local Sentence-Transformers models (cached per model name).
* **Scoring** — per-criterion max cosine similarity between criterion vectors
  and document-chunk vectors, mapped to 0–100 and combined with normalized
  weights.

**Mathematical sketch**

For each criterion :math:`c` with unit-normalized embedding :math:`\\mathbf{c}`
and document chunks with embeddings :math:`\\mathbf{d}_i`, we compute cosine
similarities :math:`s_i = \\cos(\\mathbf{c}, \\mathbf{d}_i)`. The criterion score
is :math:`100 \\times \\max(0, \\max_i s_i)` (negative similarity treated as 0).
The **overall score** is the weighted average of criterion scores using weights
from JSON, renormalized to sum to 1.

**Typical usage**

.. code-block:: python

    scorer = OrdinanceScorer(criteria, model_name="all-MiniLM-L6-v2")
    doc_embs = scorer.embed_texts(chunks)
    crit_embs = scorer.embed_texts([c["description"] for c in criteria])
    result = scorer.score(
        doc_chunks=chunks,
        doc_embeddings=doc_embs,
        crit_embeddings=crit_embs,
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
from app.utils import cosine_similarities_array, similarity_to_score


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
    * ``description`` — prose the model embeds (the semantic "probe" for the rubric).
    * ``weight`` — optional relative importance (defaults to ``1.0`` if omitted).
    * ``short`` — optional one-line summary for tables; falls back to a trimmed
      ``description``.

    **Overall score**

    Per-criterion scores live on a 0–100 scale. The overall score is
    :math:`\\sum_k w_k \\, \\text{score}_k` where ``w_k`` comes from JSON weights
    divided by their sum (so weights behave like a convex combination).

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

    def score(
        self,
        doc_chunks: List[str],
        doc_embeddings: List[List[float]],
        crit_embeddings: List[List[float]],
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """
        Score pre-embedded document chunks against pre-embedded criteria.

        **Per criterion**

        1. Take embedding row ``C[idx]`` for that criterion.
        2. Compute cosine similarity against every document row in ``D``.
        3. Record ``raw_similarity = max(sims)`` and map it to ``score`` via
           :func:`~app.utils.similarity_to_score`.
        4. Sort chunk indices descending by similarity; keep the top ``top_k``
           for human review (excerpts + parallel ``top_scores`` list).

        **Args:**
            doc_chunks: Plain-text segments aligned index-wise with ``doc_embeddings``.
            doc_embeddings: Shape ``(num_chunks, dim)`` as nested lists.
            crit_embeddings: Shape ``(num_criteria, dim)``, same order as ``self.criteria``.
            top_k: Number of evidence snippets to retain per criterion (>= 1).

        **Returns:**
            Dict with:

            * ``overall_score`` — float in ``[0, 100]`` (weighted mean).
            * ``criteria_results`` — list of dicts with keys including ``title``,
              ``short``, ``score``, ``raw_similarity``, ``top_excerpts``,
              ``top_scores``, ``weight``.

        **Preconditions:**

        ``len(doc_chunks) == len(doc_embeddings)`` and
        ``len(crit_embeddings) == len(self.criteria)`` should hold; the code does
        not assert this explicitly for performance, but violations will yield
        incorrect indexing or shape errors inside NumPy.
        """
        # Shape (n_chunks, dim) and (n_crit, dim) for vectorized dot products.
        D = np.array(doc_embeddings)
        C = np.array(crit_embeddings)

        criteria_results: List[Dict[str, Any]] = []
        per_scores: List[float] = []

        for idx, crit in enumerate(self.criteria):
            c_emb = C[idx]
            sims = cosine_similarities_array(c_emb, D)

            # argsort ascending → reverse for descending similarity order.
            top_idx = sims.argsort()[::-1][:top_k]
            top_excerpts = [doc_chunks[i] for i in top_idx]
            top_scores = sims[top_idx].tolist()

            best_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
            score = similarity_to_score(best_sim)
            per_scores.append(score)

            criteria_results.append(
                {
                    "title": crit.get("title", f"Criterion {idx+1}"),
                    "short": crit.get("short", crit.get("description", "")[:200]),
                    "score": score,
                    "raw_similarity": best_sim,
                    "top_excerpts": top_excerpts,
                    "top_scores": top_scores,
                    "weight": crit.get("weight", 1.0),
                }
            )

        overall = 0.0
        for w, s in zip(self.weights, per_scores):
            overall += w * s

        return {
            "overall_score": overall,
            "criteria_results": criteria_results,
        }
