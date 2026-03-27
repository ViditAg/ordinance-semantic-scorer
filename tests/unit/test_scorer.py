"""
Unit tests for app.scorer (OrdinanceScorer and static helpers).

Embeddings in scoring tests are hand-crafted (no model download).

Covered:
  * _sim_to_score() — static score normalisation
  * _cosine_similarities_array() — batch cosine (used inside score())
  * OrdinanceScorer.__init__ — weight normalisation
  * OrdinanceScorer.score() — full scoring pipeline
  * OrdinanceScorer.embed_texts() — SentenceTransformer mocked
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
    """Return L2-normalised version of *v*."""
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else v


def _orthogonal_basis(dim: int, n: int) -> list[list[float]]:
    """Return *n* orthogonal unit vectors of length *dim* (n <= dim)."""
    assert n <= dim
    eye = np.eye(dim)
    return [eye[i].tolist() for i in range(n)]


def _minimal_criteria():
    return [{"title": "t", "description": "d", "weight": 1.0}]


def _make_mock_model(dim: int = 8, seed: int = 42):
    """Return a MagicMock that mimics SentenceTransformer.encode."""
    mock_model = MagicMock()
    rng = np.random.default_rng(seed)

    def _fake_encode(texts, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    mock_model.encode.side_effect = _fake_encode
    return mock_model


def _manual_row_cosines(vec: list[float], matrix: list[list[float]]) -> list[float]:
    """Reference cosine(vec, row) per row — mirrors _cosine_similarities_array."""
    v = np.array(vec, dtype=float)
    vn = np.linalg.norm(v)
    out = []
    for row in matrix:
        r = np.array(row, dtype=float)
        rn = np.linalg.norm(r)
        denom = vn * rn
        if denom == 0:
            out.append(float("nan"))
        else:
            out.append(float(np.dot(v, r) / denom))
    return out


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
        result = OrdinanceScorer._sim_to_score(0.12345)
        assert result == round(result, 2)


# ---------------------------------------------------------------------------
# _cosine_similarities_array (batch cosine — scoring hot path)
# ---------------------------------------------------------------------------


class TestCosineSimilaritiesArray:
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
        M = _orthogonal_basis(3, 3)
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)
        assert sims[2] == pytest.approx(0.0, abs=1e-6)

    def test_returns_numpy_array(self):
        v = [1.0, 0.0]
        M = [[1.0, 0.0]]
        result = OrdinanceScorer._cosine_similarities_array(v, M)
        assert isinstance(result, np.ndarray)

    def test_matches_per_row_cosine_formula_nonzero_rows(self):
        """Batch output equals cosine(vec, row) for ordinary nonzero rows."""
        v = _unit([1.0, 2.0, -0.5])
        M = [
            _unit([0.1, -3.0, 2.0]),
            _unit([2.0, 2.0, 2.0]),
        ]
        sims = OrdinanceScorer._cosine_similarities_array(v, M)
        expected = _manual_row_cosines(v, M)
        for got, exp in zip(sims.tolist(), expected):
            assert got == pytest.approx(exp, abs=1e-6)


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
        criteria = [
            {"title": "A", "description": "d"},
            {"title": "B", "description": "d", "weight": 1.0},
        ]
        scorer = OrdinanceScorer(criteria)
        assert len(scorer.weights) == 2
        assert sum(scorer.weights) == pytest.approx(1.0)

    def test_all_zero_weights_fallback_to_uniform(self):
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
    """Integration-style unit tests using hand-crafted embeddings."""

    @pytest.fixture()
    def two_criteria(self):
        return [
            {"title": "A", "description": "desc A", "weight": 1.0},
            {"title": "B", "description": "desc B", "weight": 1.0},
        ]

    @pytest.fixture()
    def identical_embeddings(self):
        vec = _unit([1.0, 0.0, 0.0, 0.0])
        return {
            "doc_chunks": ["chunk text"],
            "doc_embeddings": [vec],
            "crit_embeddings": [vec, _unit([0.0, 1.0, 0.0, 0.0])],
        }

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
    @pytest.fixture(autouse=True)
    def _patch_model(self):
        self._mock_model = _make_mock_model()
        with patch(
            "app.scorer._get_sentence_transformer",
            return_value=self._mock_model,
        ):
            yield

    def test_empty_input_returns_empty_list(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        assert scorer.embed_texts([]) == []

    def test_single_text_returns_list_of_length_one(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        assert len(scorer.embed_texts(["hello"])) == 1

    def test_multiple_texts_returns_correct_count(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        texts = ["one", "two", "three", "four"]
        assert len(scorer.embed_texts(texts)) == len(texts)

    def test_each_embedding_is_a_list(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        result = scorer.embed_texts(["test sentence"])
        assert isinstance(result[0], list)

    def test_each_embedding_element_is_float(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        for val in scorer.embed_texts(["test sentence"])[0]:
            assert isinstance(val, float)

    def test_all_embeddings_have_same_dimension(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        result = scorer.embed_texts(["hello", "world", "dark sky"])
        assert len({len(e) for e in result}) == 1

    def test_embeddings_are_approximately_unit_length(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        for emb in scorer.embed_texts(["test text", "another sentence"]):
            norm = math.sqrt(sum(x ** 2 for x in emb))
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_encode_is_called_with_texts(self):
        scorer = OrdinanceScorer(_minimal_criteria(), model_name="mock-model")
        texts = ["shielding requirement", "color temperature"]
        scorer.embed_texts(texts)
        self._mock_model.encode.assert_called_once()
        call_args = self._mock_model.encode.call_args
        assert call_args[0][0] == texts
