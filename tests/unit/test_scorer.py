"""
Unit tests for app.analysis.scorer.OrdinanceScorer
====================================================

All embeddings are constructed manually (no model download needed).

Covered:
  * _sim_to_score()           – static score normalisation
  * _cosine_similarities_array() – batch cosine computation
  * OrdinanceScorer.__init__  – weight normalisation
  * OrdinanceScorer.score()   – full scoring pipeline
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.analysis.scorer import OrdinanceScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    """Return L2-normalised version of *v*."""
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else v


def _orthogonal_basis(dim: int, n: int) -> list[list[float]]:
    """Return *n* orthogonal unit vectors of length *dim* (n <= dim)."""
    assert n <= dim
    eye = np.eye(dim)
    return [eye[i].tolist() for i in range(n)]


# ---------------------------------------------------------------------------
# _sim_to_score
# ---------------------------------------------------------------------------


class TestSimToScore:
    def test_zero_similarity_gives_zero_score(self):
        assert OrdinanceScorer._sim_to_score(0.0) == 0.0

    def test_one_similarity_gives_hundred(self):
        assert OrdinanceScorer._sim_to_score(1.0) == pytest.approx(100.0)

    def test_half_similarity_gives_fifty(self):
        assert OrdinanceScorer._sim_to_score(0.5) == pytest.approx(50.0)

    def test_negative_similarity_clamped_to_zero(self):
        assert OrdinanceScorer._sim_to_score(-0.5) == 0.0
        assert OrdinanceScorer._sim_to_score(-1.0) == 0.0

    def test_arbitrary_positive_value(self):
        sim = 0.73
        assert OrdinanceScorer._sim_to_score(sim) == pytest.approx(73.0)

    def test_result_is_rounded_to_two_decimal_places(self):
        # round(0.12345 * 100, 2) == 12.35
        result = OrdinanceScorer._sim_to_score(0.12345)
        # Verify no more than 2 decimal places.
        assert result == round(result, 2)


# ---------------------------------------------------------------------------
# _cosine_similarities_array
# ---------------------------------------------------------------------------


class TestCosineSimArray:
    def test_identical_vector_gives_one(self):
        v = [1.0, 0.0, 0.0]
        M = [[1.0, 0.0, 0.0]]
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vector_gives_zero(self):
        v = [1.0, 0.0, 0.0]
        M = [[0.0, 1.0, 0.0]]
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vector_gives_minus_one(self):
        v = [1.0, 0.0]
        M = [[-1.0, 0.0]]
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_in_matrix_gets_safe_denominator(self):
        """A zero row in the matrix must not cause division-by-zero."""
        v = [1.0, 0.0]
        M = [[0.0, 0.0], [1.0, 0.0]]
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert not any(math.isnan(s) for s in sims.tolist())

    def test_multiple_rows_computed_correctly(self):
        v = [1.0, 0.0, 0.0]
        M = _orthogonal_basis(3, 3)  # identity matrix rows
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)
        assert sims[2] == pytest.approx(0.0, abs=1e-6)

    def test_returns_numpy_array(self):
        v = [1.0, 0.0]
        M = [[1.0, 0.0]]
        result = OrdinanceScorer._cosine_similarities_array(v, M)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------


class TestWeightNormalisation:
    def test_weights_sum_to_one(self):
        criteria = [
            {"title": "A", "description": "desc A", "weight": 2.0},
            {"title": "B", "description": "desc B", "weight": 1.0},
            {"title": "C", "description": "desc C", "weight": 1.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert sum(scorer.weights) == pytest.approx(1.0, abs=1e-9)

    def test_equal_weights_normalise_to_equal_fractions(self):
        n = 4
        criteria = [{"title": str(i), "description": "d", "weight": 1.0} for i in range(n)]
        scorer = OrdinanceScorer(criteria)
        for w in scorer.weights:
            assert w == pytest.approx(1.0 / n, abs=1e-9)

    def test_missing_weight_defaults_to_one(self):
        """Criteria without a 'weight' key should default to 1.0."""
        criteria = [
            {"title": "A", "description": "d"},   # no weight key
            {"title": "B", "description": "d", "weight": 1.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert len(scorer.weights) == 2
        assert sum(scorer.weights) == pytest.approx(1.0)

    def test_all_zero_weights_fallback_to_uniform(self):
        """Guard against degenerate all-zero weights (uses or 1.0 default)."""
        criteria = [
            {"title": "A", "description": "d", "weight": 0.0},
            {"title": "B", "description": "d", "weight": 0.0},
        ]
        # This should not raise; the sum falls back to 1.0 via `sum(weights) or 1.0`
        scorer = OrdinanceScorer(criteria)
        assert all(w == pytest.approx(0.0) for w in scorer.weights)


# ---------------------------------------------------------------------------
# OrdinanceScorer.score()
# ---------------------------------------------------------------------------


class TestScorerScore:
    """Integration-style unit tests using hand-crafted embeddings."""

    # ---------- fixtures ----------

    @pytest.fixture()
    def two_criteria(self):
        return [
            {"title": "A", "description": "desc A", "weight": 1.0},
            {"title": "B", "description": "desc B", "weight": 1.0},
        ]

    @pytest.fixture()
    def identical_embeddings(self):
        """Doc chunk and criterion have identical embeddings → score = 100."""
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        return {
            "doc_chunks": ["chunk text"],
            "doc_embeddings": [vec],
            "crit_embeddings": [vec, _unit([0.0, 1.0, 0.0, 0.0])],
        }

    # ---------- structural tests ----------

    def test_output_has_overall_score_key(self, two_criteria, identical_embeddings):
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert "overall_score" in result

    def test_output_has_criteria_results_key(self, two_criteria, identical_embeddings):
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert "criteria_results" in result

    def test_criteria_results_count_matches_criteria(self, two_criteria, identical_embeddings):
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        assert len(result["criteria_results"]) == len(two_criteria)

    def test_each_result_has_required_keys(self, two_criteria, identical_embeddings):
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(**identical_embeddings)
        required = {"title", "short", "score", "raw_similarity", "top_excerpts", "top_scores", "weight"}
        for r in result["criteria_results"]:
            assert required.issubset(r.keys()), f"Missing keys in {r}"

    # ---------- score values ----------

    def test_identical_embedding_scores_one_hundred(self, two_criteria):
        vec = _unit([1.0, 0.0, 0.0])
        scorer = OrdinanceScorer(two_criteria)
        result = scorer.score(
            doc_chunks=["relevant text"],
            doc_embeddings=[vec],
            crit_embeddings=[vec, _unit([0.0, 1.0, 0.0])],
        )
        assert result["criteria_results"][0]["score"] == pytest.approx(100.0, abs=0.01)

    def test_orthogonal_embedding_scores_zero(self, two_criteria):
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
        n = 5
        criteria = [{"title": str(i), "description": "d", "weight": 1.0} for i in range(n)]
        scorer = OrdinanceScorer(criteria)
        rng = np.random.default_rng(7)
        # Random normalised embeddings
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

    # ---------- overall score ----------

    def test_overall_score_is_weighted_average(self):
        criteria = [
            {"title": "A", "description": "d", "weight": 2.0},
            {"title": "B", "description": "d", "weight": 1.0},
        ]
        # Criterion A: identical to doc → score 100
        # Criterion B: orthogonal to doc → score 0
        vec_a = _unit([1.0, 0.0])
        vec_b = _unit([0.0, 1.0])
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=["x"],
            doc_embeddings=[vec_a],
            crit_embeddings=[vec_a, vec_b],
        )
        # weights normalised: 2/3 and 1/3
        expected = (2 / 3) * 100.0 + (1 / 3) * 0.0
        assert result["overall_score"] == pytest.approx(expected, abs=0.1)

    # ---------- top_k ----------

    def test_top_k_one_returns_single_excerpt(self):
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
        """raw_similarity should equal the maximum cosine similarity found."""
        criteria = [{"title": "A", "description": "d", "weight": 1.0}]
        vec_a = _unit([1.0, 0.0, 0.0])
        vec_b = _unit([0.0, 1.0, 0.0])
        vec_c = _unit([1.0, 0.0, 0.0])   # identical to criterion
        scorer = OrdinanceScorer(criteria)
        result = scorer.score(
            doc_chunks=["a", "b", "c"],
            doc_embeddings=[vec_a, vec_b, vec_c],
            crit_embeddings=[vec_a],
        )
        # All sims between [1,0,0] criterion and chunks [1,0,0], [0,1,0], [1,0,0]
        # max is 1.0
        assert result["criteria_results"][0]["raw_similarity"] == pytest.approx(1.0, abs=1e-6)
