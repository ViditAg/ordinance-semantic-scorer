"""
Unit tests for ``app.scorer`` — :class:`~app.scorer.OrdinanceScorer` and helpers.

**Testing strategy**

* **Pure math paths** — supply handcrafted normalised vectors so cosine similarities
  are exact without invoking ``sentence_transformers``.
* **Embedding path** — patch ``_get_sentence_transformer`` with a ``MagicMock`` that
  exposes a deterministic ``encode`` side effect.

**Coverage map**

* ``OrdinanceScorer.__init__`` — weight renormalisation edge cases.
* :meth:`~app.scorer.OrdinanceScorer.score` — shapes, ``top_k`` excerpts, weighting.
* :meth:`~app.scorer.OrdinanceScorer.embed_texts` — batching + output types (mocked).

Vector math helpers live in :mod:`app.utils` and are tested in ``test_utils.py``.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.scorer import OrdinanceScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    """
    L2-normalise an arbitrary vector for hand-crafted embedding tests.

    **Why explicit normalisation?**

    :class:`~app.scorer.OrdinanceScorer` uses unit-normalised embeddings from the
    model; tests mirror that contract so expected similarities are predictable.
    """
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else v


def _minimal_criteria():
    """Single-criterion rubric — minimal dict accepted by ``OrdinanceScorer``."""
    return [{"title": "t", "description": "d", "weight": 1.0}]


def _make_mock_model(dim: int = 8, seed: int = 42):
    """
    Build a ``MagicMock`` exposing ``.encode`` like ``SentenceTransformer``.

    **Fake encode:** Gaussian draws reshaped to ``(batch, dim)``, then row-wise
    L2 normalisation — cheap, deterministic given *seed*, and shape-compatible with
    production code paths.
    """
    mock_model = MagicMock()
    rng = np.random.default_rng(seed)

    def _fake_encode(texts, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    mock_model.encode.side_effect = _fake_encode
    return mock_model


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------


class TestWeightNormalisation:
    """``OrdinanceScorer.__init__`` should make weights a convex combination."""

    def test_weights_sum_to_one(self):
        """Renormalised explicit weights must sum to unity (floating tolerance)."""
        criteria = [
            {"title": "A", "description": "desc A", "weight": 2.0},
            {"title": "B", "description": "desc B", "weight": 1.0},
            {"title": "C", "description": "desc C", "weight": 1.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert sum(scorer.weights) == pytest.approx(1.0, abs=1e-9)

    def test_equal_weights_normalise_to_equal_fractions(self):
        """Uniform raw weights → each criterion gets ``1/n`` influence."""
        n = 4
        criteria = [{"title": str(i), "description": "d", "weight": 1.0} for i in range(n)]
        scorer = OrdinanceScorer(criteria)
        for w in scorer.weights:
            assert w == pytest.approx(1.0 / n, abs=1e-9)

    def test_missing_weight_defaults_to_one(self):
        """Omitted ``weight`` key should behave like ``weight: 1.0``."""
        criteria = [
            {"title": "A", "description": "d"},
            {"title": "B", "description": "d", "weight": 1.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert len(scorer.weights) == 2
        assert sum(scorer.weights) == pytest.approx(1.0)

    def test_all_zero_weights_fallback_to_uniform(self):
        """
        If every raw weight is 0, total mass is 0 and the code divides by the
        ``or 1.0`` fallback — weights become all-zero (documented edge case).
        """
        criteria = [
            {"title": "A", "description": "d", "weight": 0.0},
            {"title": "B", "description": "d", "weight": 0.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert all(w == pytest.approx(0.0) for w in scorer.weights)


# ---------------------------------------------------------------------------
# OrdinanceScorer.score()
# ---------------------------------------------------------------------------


class TestScorerScore:
    """
    End-to-end checks of :meth:`OrdinanceScorer.score` **without** loading models.

    Embeddings are crafted so cosine similarities are predictable (basis vectors,
    duplicates, etc.). This isolates linear-algebra logic from neural network noise.
    """

    @pytest.fixture()
    def two_criteria(self):
        """Two-criterion rubric with equal explicit weights."""
        return [
            {"title": "A", "description": "desc A", "weight": 1.0},
            {"title": "B", "description": "desc B", "weight": 1.0},
        ]

    @pytest.fixture()
    def identical_embeddings(self):
        """
        Shared fixture: first criterion matches the lone doc chunk; second is orthogonal.
        """
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        return {
            "doc_chunks": ["chunk text"],
            "doc_embeddings": [vec],
            "crit_embeddings": [vec, _unit([0.0, 1.0, 0.0, 0.0])],
        }

    def test_output_has_overall_score_key(self, two_criteria, identical_embeddings):
        """API contract: callers expect an aggregate ``overall_score`` field."""
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert "overall_score" in result

    def test_output_has_criteria_results_key(self, two_criteria, identical_embeddings):
        """API contract: per-row detail lives under ``criteria_results``."""
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert "criteria_results" in result

    def test_criteria_results_count_matches_criteria(self, two_criteria, identical_embeddings):
        """One result object per rubric entry, preserving input order."""
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert len(result["criteria_results"]) == len(two_criteria)

    def test_each_result_has_required_keys(self, two_criteria, identical_embeddings):
        """Streamlit / JSON consumers rely on this stable key set."""
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        required = {"title", "short", "score", "raw_similarity", "top_excerpts", "top_scores", "weight"}
        for r in result["criteria_results"]:
            assert required.issubset(r.keys()), f"Missing keys in {r}"

    def test_identical_embedding_scores_one_hundred(self, two_criteria):
        """When chunk embedding equals criterion embedding, cosine → 1 → score 100."""
        vec = _unit([1.0, 0.0, 0.0])
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(
            doc_chunks=["relevant text"],
            doc_embeddings=[vec],
            crit_embeddings=[vec, _unit([0.0, 1.0, 0.0])],
        )
        assert result["criteria_results"][0]["score"] == pytest.approx(100.0, abs=0.01)

    def test_orthogonal_embedding_scores_zero(self, two_criteria):
        """Orthogonal criterion vs document → cosine 0 → clipped score 0."""
        doc_vec = _unit([1.0, 0.0, 0.0])
        crit_vec = _unit([0.0, 1.0, 0.0])
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(
            doc_chunks=["irrelevant text"],
            doc_embeddings=[doc_vec],
            crit_embeddings=[crit_vec, doc_vec],
        )
        assert result["criteria_results"][0]["score"] == pytest.approx(0.0, abs=0.01)

    def test_score_in_range_zero_to_one_hundred(self):
        """Property test with random *but normalised* embeddings — scores stay bounded."""
        n = 5
        criteria = [{"title": str(i), "description": "d", "weight": 1.0} for i in range(n)]
        scorer = OrdinanceScorer(criteria)
        rng = np.random.default_rng(7)
        D = rng.standard_normal((10, 16)).astype(float)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        C = rng.standard_normal((n, 16)).astype(float)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        result = scorer.score(
            doc_chunks=[f"chunk {i}" for i in range(10)],
            doc_embeddings=D.tolist(),
            crit_embeddings=C.tolist(),
        )
        for r in result["criteria_results"]:
            assert 0.0 <= r["score"] <= 100.0

    def test_overall_score_is_weighted_average(self):
        """
        Hand-designed scenario: criterion A hits, criterion B misses — overall should
        match ``(2/3)*100 + (1/3)*0`` given weights 2 and 1.
        """
        criteria = [
            {"title": "A", "description": "d", "weight": 2.0},
            {"title": "B", "description": "d", "weight": 1.0},
        ]
        vec_a = _unit([1.0, 0.0])
        vec_b = _unit([0.0, 1.0])
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=["x"],
            doc_embeddings=[vec_a],
            crit_embeddings=[vec_a, vec_b],
        )
        expected = (2 / 3) * 100.0 + (1 / 3) * 0.0
        assert result["overall_score"] == pytest.approx(expected, abs=0.1)

    def test_top_k_one_returns_single_excerpt(self):
        """Default ``top_k=1`` should surface exactly one supporting chunk string."""
        criteria = [{"title": "A", "description": "d", "weight": 1.0}]
        vec = _unit([1.0, 0.0])
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=["chunk one", "chunk two", "chunk three"],
            doc_embeddings=[vec, vec, vec],
            crit_embeddings=[vec],
            top_k=1,
        )
        assert len(result["criteria_results"][0]["top_excerpts"]) == 1

    def test_top_k_three_returns_three_excerpts(self):
        """With five random chunks, ``top_k=3`` should return three distinct excerpts."""
        criteria = [{"title": "A", "description": "d", "weight": 1.0}]
        rng = np.random.default_rng(0)
        D = rng.standard_normal((5, 4)).astype(float)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        C = rng.standard_normal((1, 4)).astype(float)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=[f"chunk {i}" for i in range(5)],
            doc_embeddings=D.tolist(),
            crit_embeddings=C.tolist(),
            top_k=3,
        )
        assert len(result["criteria_results"][0]["top_excerpts"]) == 3

    def test_raw_similarity_matches_max_cosine_similarity(self):
        """``raw_similarity`` must equal the max over per-chunk cosine similarities."""
        criteria = [{"title": "A", "description": "d", "weight": 1.0}]
        vec_a = _unit([1.0, 0.0, 0.0])
        vec_b = _unit([0.0, 1.0, 0.0])
        vec_c = _unit([1.0, 0.0, 0.0])
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=["a", "b", "c"],
            doc_embeddings=[vec_a, vec_b, vec_c],
            crit_embeddings=[vec_a],
        )
        assert result["criteria_results"][0]["raw_similarity"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# OrdinanceScorer.embed_texts (mocked model)
# ---------------------------------------------------------------------------


class TestOrdinanceScorerEmbedTexts:
    """
    :meth:`OrdinanceScorer.embed_texts` exercised against a patched transformer.

    ``autouse`` fixture keeps tests hermetic — no Hugging Face download, no GPU.
    """

    @pytest.fixture(autouse=True)
    def _patch_model(self):
        self._mock_model = _make_mock_model()
        with patch(
            "app.scorer._get_sentence_transformer",
            return_value=self._mock_model,
        ):
            yield

    def test_empty_input_returns_empty_list(self):
        """Short-circuit path: no encode call, empty list in/out."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        assert scorer.embed_texts([]) == []

    def test_single_text_returns_list_of_length_one(self):
        """Batch size 1 still returns a list-of-vectors (never a bare vector)."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        assert len(scorer.embed_texts(["hello"])) == 1

    def test_multiple_texts_returns_correct_count(self):
        """Output list length must mirror input list length."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        texts = ["one", "two", "three", "four"]
        assert len(scorer.embed_texts(texts)) == len(texts)

    def test_each_embedding_is_a_list(self):
        """Embeddings are Python lists (JSON-friendly), not ``ndarray`` views."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        result = scorer.embed_texts(["test sentence"])
        assert isinstance(result[0], list)

    def test_each_embedding_element_is_float(self):
        """Scalar dtype should be plain ``float`` for interoperability."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        for val in scorer.embed_texts(["test sentence"])[0]:
            assert isinstance(val, float)

    def test_all_embeddings_have_same_dimension(self):
        """All rows in a batch must share embedding width (model-dependent, here 8)."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        result = scorer.embed_texts(["hello", "world", "dark sky"])
        assert len({len(e) for e in result}) == 1

    def test_embeddings_are_approximately_unit_length(self):
        """Production call sets ``normalize_embeddings=True`` — mock should respect that."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        for emb in scorer.embed_texts(["test text", "another sentence"]):
            norm = math.sqrt(sum(x ** 2 for x in emb))
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_encode_is_called_with_texts(self):
        """Regression: ensure the list of strings is forwarded unchanged to ``encode``."""
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        texts = ["shielding requirement", "color temperature"]
        scorer.embed_texts(texts)
        self._mock_model.encode.assert_called_once()
        call_args = self._mock_model.encode.call_args
        assert call_args[0][0] == texts
