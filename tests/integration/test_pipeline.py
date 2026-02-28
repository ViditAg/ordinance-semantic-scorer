"""
Integration tests for the full ordinance-scoring pipeline
==========================================================

These tests wire together the real application modules
(pdf_parser → text_splitter → EmbeddingProvider → OrdinanceScorer)
while replacing only the SentenceTransformer model and pdfplumber so that
no internet access or file I/O is required.

They verify:
  1. criteria.json is loadable and well-formed
  2. The text-splitting and scoring pipeline produces consistent, sensible output
  3. Key invariants hold end-to-end (score count, score range, required keys …)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analysis.embeddings import EmbeddingProvider
from app.analysis.scorer import OrdinanceScorer
from app.utils.text_splitter import chunk_text

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_PATH = ROOT / "app" / "data" / "criteria.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_CRITERIA_COUNT = 29
EXPECTED_CRITERIA_KEYS = {"title", "short", "description", "weight"}


def _fake_encode_factory(dim: int = 16, seed: int = 0):
    """
    Return an encode function that produces deterministic normalised vectors
    of dimension *dim*.  Each call advances the RNG so different batches get
    different (but reproducible) embeddings.
    """
    rng = np.random.default_rng(seed)

    def _encode(texts, show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    return _encode


def _make_mock_model(dim: int = 16, seed: int = 0) -> MagicMock:
    m = MagicMock()
    m.encode.side_effect = _fake_encode_factory(dim, seed)
    return m


# ---------------------------------------------------------------------------
# 1. criteria.json integrity
# ---------------------------------------------------------------------------


class TestCriteriaJson:
    def test_file_exists(self):
        assert CRITERIA_PATH.exists(), f"criteria.json not found at {CRITERIA_PATH}"

    def test_file_is_valid_json(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_top_level_key_is_criteria(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        assert "criteria" in data, "Top-level key 'criteria' missing from criteria.json"

    def test_criteria_is_a_list(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        assert isinstance(data["criteria"], list)

    def test_expected_criterion_count(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        assert len(data["criteria"]) == EXPECTED_CRITERIA_COUNT, (
            f"Expected {EXPECTED_CRITERIA_COUNT} criteria, "
            f"got {len(data['criteria'])}"
        )

    def test_each_criterion_has_required_keys(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        for idx, crit in enumerate(data["criteria"]):
            missing = EXPECTED_CRITERIA_KEYS - crit.keys()
            assert not missing, (
                f"Criterion #{idx + 1} is missing keys: {missing}"
            )

    def test_all_weights_are_positive(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        for crit in data["criteria"]:
            assert crit["weight"] > 0, (
                f"Criterion '{crit['title']}' has non-positive weight {crit['weight']}"
            )

    def test_serial_numbers_are_unique(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        serials = [c.get("serial_number") for c in data["criteria"] if "serial_number" in c]
        assert len(serials) == len(set(serials)), "Duplicate serial_numbers in criteria.json"

    def test_no_blank_descriptions(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        for crit in data["criteria"]:
            assert crit["description"].strip(), (
                f"Criterion '{crit['title']}' has an empty description"
            )


# ---------------------------------------------------------------------------
# 2. text_splitter → scorer (no embeddings needed)
# ---------------------------------------------------------------------------


class TestChunkingPipeline:
    SAMPLE_TEXT = (
        "All outdoor lighting shall be fully shielded and downcast. "
        "Fixtures must comply with BUG rating B0 U0 G0. "
        "Color temperature must not exceed 3000K for all new installations. "
    ) * 30  # ~3 000 chars to force multiple chunks

    def test_chunking_produces_chunks(self):
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=400, overlap=50)
        assert len(chunks) > 1

    def test_chunk_count_consistent_across_calls(self):
        chunks_a = chunk_text(self.SAMPLE_TEXT, chunk_size=400, overlap=50)
        chunks_b = chunk_text(self.SAMPLE_TEXT, chunk_size=400, overlap=50)
        assert len(chunks_a) == len(chunks_b)

    def test_all_chunks_non_empty(self):
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=400, overlap=50)
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# 3. End-to-end pipeline (mocked model)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """
    Wires: text → chunk_text → EmbeddingProvider.embed_texts → OrdinanceScorer.score
    The SentenceTransformer is replaced with a deterministic fake.
    """

    SAMPLE_TEXT = (
        "Section 1: Purpose. This ordinance protects dark skies. "
        "Section 2: Scope. Applies to all outdoor artificial lighting. "
        "Section 3: Shielding. All fixtures shall be fully shielded. "
        "Section 4: Color temperature shall not exceed 2200K. "
        "Section 5: Uplighting is prohibited. "
    ) * 20

    @pytest.fixture(autouse=True)
    def _patch_model(self):
        self._mock_model = _make_mock_model(dim=16, seed=99)
        with patch(
            "app.analysis.embeddings._get_sentence_transformer",
            return_value=self._mock_model,
        ):
            yield

    @pytest.fixture()
    def criteria(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        return data["criteria"]

    @pytest.fixture()
    def pipeline_result(self, criteria):
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=500, overlap=50)
        provider = EmbeddingProvider(model_name="mock-model")
        doc_embeddings = provider.embed_texts(chunks)
        crit_embeddings = provider.embed_texts([c["description"] for c in criteria])
        scorer = OrdinanceScorer(criteria=criteria)
        return scorer.score(
            doc_chunks=chunks,
            doc_embeddings=doc_embeddings,
            crit_embeddings=crit_embeddings,
            top_k=2,
        )

    # --- structural assertions ---

    def test_result_has_overall_score(self, pipeline_result):
        assert "overall_score" in pipeline_result

    def test_result_has_criteria_results(self, pipeline_result):
        assert "criteria_results" in pipeline_result

    def test_criteria_results_count(self, pipeline_result):
        assert len(pipeline_result["criteria_results"]) == EXPECTED_CRITERIA_COUNT

    def test_overall_score_in_range(self, pipeline_result):
        score = pipeline_result["overall_score"]
        assert 0.0 <= score <= 100.0, f"Overall score {score} out of [0, 100]"

    def test_per_criterion_scores_in_range(self, pipeline_result):
        for r in pipeline_result["criteria_results"]:
            assert 0.0 <= r["score"] <= 100.0, (
                f"Criterion '{r['title']}' has out-of-range score {r['score']}"
            )

    def test_each_result_has_required_keys(self, pipeline_result):
        required = {"title", "short", "score", "raw_similarity",
                    "top_excerpts", "top_scores", "weight"}
        for r in pipeline_result["criteria_results"]:
            assert required.issubset(r.keys())

    def test_top_k_excerpts_count(self, pipeline_result):
        for r in pipeline_result["criteria_results"]:
            assert len(r["top_excerpts"]) <= 2

    def test_top_excerpts_are_strings(self, pipeline_result):
        for r in pipeline_result["criteria_results"]:
            for excerpt in r["top_excerpts"]:
                assert isinstance(excerpt, str)

    def test_doc_embeddings_count_equals_chunk_count(self):
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=500, overlap=50)
        provider = EmbeddingProvider(model_name="mock-model")
        doc_embeddings = provider.embed_texts(chunks)
        assert len(doc_embeddings) == len(chunks), (
            "Number of embeddings must match number of chunks"
        )

    def test_crit_embeddings_count_equals_criteria_count(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        criteria = data["criteria"]
        provider = EmbeddingProvider(model_name="mock-model")
        crit_embeddings = provider.embed_texts([c["description"] for c in criteria])
        assert len(crit_embeddings) == len(criteria)

    def test_result_is_deterministic(self):
        """Running the pipeline twice on the same text must yield the same scores."""
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        criteria = data["criteria"]
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=500, overlap=50)
        provider = EmbeddingProvider(model_name="mock-model")

        # Embed once and reuse: model mock is stateful, so we capture embeddings.
        doc_embs = provider.embed_texts(chunks)
        crit_embs = provider.embed_texts([c["description"] for c in criteria])

        scorer = OrdinanceScorer(criteria=criteria)
        r1 = scorer.score(doc_chunks=chunks, doc_embeddings=doc_embs,
                          crit_embeddings=crit_embs, top_k=1)
        r2 = scorer.score(doc_chunks=chunks, doc_embeddings=doc_embs,
                          crit_embeddings=crit_embs, top_k=1)
        assert r1["overall_score"] == r2["overall_score"]
        for a, b in zip(r1["criteria_results"], r2["criteria_results"]):
            assert a["score"] == b["score"]


# ---------------------------------------------------------------------------
# 4. scripts/score_samples.py helpers (no CLI invocation needed)
# ---------------------------------------------------------------------------


class TestScriptHelpers:
    """Tests for the helper functions used by the CLI script."""

    def test_load_criteria_from_json(self):
        from scripts.score_samples import _load_criteria
        criteria = _load_criteria(CRITERIA_PATH)
        assert len(criteria) == EXPECTED_CRITERIA_COUNT

    def test_load_criteria_raises_on_bad_format(self, tmp_path):
        from scripts.score_samples import _load_criteria
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps([{"title": "X"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="criteria"):
            _load_criteria(bad_file)

    def test_load_text_from_txt_file(self, tmp_path):
        from scripts.score_samples import _load_text
        txt = tmp_path / "ordinance.txt"
        txt.write_text("All lights shall be shielded.", encoding="utf-8")
        assert _load_text(txt) == "All lights shall be shielded."

    def test_load_text_unsupported_extension_raises(self, tmp_path):
        from scripts.score_samples import _load_text
        bad = tmp_path / "file.docx"
        bad.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_text(bad)
