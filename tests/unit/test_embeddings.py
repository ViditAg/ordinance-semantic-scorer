"""
Unit tests for app.analysis.embeddings
=======================================

Covered:
  * cosine_similarity() – pure math, no mocking needed
  * EmbeddingProvider.embed_texts() – SentenceTransformer is mocked so that
    no model download is required
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analysis.embeddings import EmbeddingProvider, cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity – unit tests (no external dependencies)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the standalone cosine_similarity(a, b) helper."""

    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-7)

    def test_opposite_vectors_return_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-7)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-7)

    def test_zero_vector_a_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors_return_zero(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_value(self):
        # cos( 45° ) = sqrt(2)/2 ≈ 0.7071
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)

    def test_result_is_float(self):
        result = cosine_similarity([1.0, 2.0], [3.0, 4.0])
        assert isinstance(result, float)

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_scaling_invariance(self):
        """Scaling a vector should not change the similarity."""
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        assert cosine_similarity(a, b) == pytest.approx(
            cosine_similarity([x * 100 for x in a], b), abs=1e-6
        )


# ---------------------------------------------------------------------------
# EmbeddingProvider – mocked SentenceTransformer
# ---------------------------------------------------------------------------


def _make_mock_model(dim: int = 8, seed: int = 42):
    """Return a MagicMock that mimics SentenceTransformer.encode."""
    mock_model = MagicMock()
    rng = np.random.default_rng(seed)

    def _fake_encode(texts, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)  # L2-normalised

    mock_model.encode.side_effect = _fake_encode
    return mock_model


class TestEmbeddingProvider:

    @pytest.fixture(autouse=True)
    def _patch_model(self):
        """Patch the cached model loader so no real download happens."""
        self._mock_model = _make_mock_model()
        with patch(
            "app.analysis.embeddings._get_sentence_transformer",
            return_value=self._mock_model,
        ):
            yield

    # ------------------------------------------------------------------
    # empty / trivial inputs
    # ------------------------------------------------------------------

    def test_empty_input_returns_empty_list(self):
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts([])
        assert result == []

    # ------------------------------------------------------------------
    # return-value shape and type
    # ------------------------------------------------------------------

    def test_single_text_returns_list_of_length_one(self):
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts(["hello"])
        assert len(result) == 1

    def test_multiple_texts_returns_correct_count(self):
        provider = EmbeddingProvider(model_name="mock-model")
        texts = ["one", "two", "three", "four"]
        result = provider.embed_texts(texts)
        assert len(result) == len(texts)

    def test_each_embedding_is_a_list(self):
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts(["test sentence"])
        assert isinstance(result[0], list)

    def test_each_embedding_element_is_float(self):
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts(["test sentence"])
        for val in result[0]:
            assert isinstance(val, float)

    def test_all_embeddings_have_same_dimension(self):
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts(["hello", "world", "dark sky"])
        dims = {len(e) for e in result}
        assert len(dims) == 1, "All embeddings must have the same dimension"

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------

    def test_embeddings_are_approximately_unit_length(self):
        """The model is configured with normalize_embeddings=True, so L2
        norm of each embedding should be very close to 1.0."""
        provider = EmbeddingProvider(model_name="mock-model")
        result = provider.embed_texts(["test text", "another sentence"])
        for emb in result:
            norm = math.sqrt(sum(x ** 2 for x in emb))
            assert norm == pytest.approx(1.0, abs=1e-5), (
                f"Embedding not normalised – L2 norm = {norm}"
            )

    # ------------------------------------------------------------------
    # encode is actually called
    # ------------------------------------------------------------------

    def test_encode_is_called_with_texts(self):
        provider = EmbeddingProvider(model_name="mock-model")
        texts = ["shielding requirement", "color temperature"]
        provider.embed_texts(texts)
        self._mock_model.encode.assert_called_once()
        call_args = self._mock_model.encode.call_args
        assert call_args[0][0] == texts or call_args[1].get("texts") == texts or \
            list(call_args[0][0]) == texts
