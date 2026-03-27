"""
Integration tests for the full ordinance-scoring pipeline
==========================================================

These tests wire together the real application modules
(pdf_parser → text_splitter → OrdinanceScorer embed + score)
while replacing only the SentenceTransformer model and pdfplumber so that
no internet access or file I/O is required.

They verify:
  1. criteria.json is loadable and well-formed
  2. The text-splitting and scoring pipeline produces consistent, sensible output
  3. Key invariants hold end-to-end (score count, score range, required keys …)
  4. The real sample PDF can be parsed and produces non-empty text (smoke test)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.scorer import OrdinanceScorer
from app.utils import chunk_text

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_PATH = ROOT / "app" / "criteria.json"
SAMPLE_PDF = ROOT / "sample" / "updated_dark_skies_outdoor_lighting_ordinance.pdf"

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
    Wires: text → chunk_text → OrdinanceScorer.embed_texts → OrdinanceScorer.score
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
            "app.scorer._get_sentence_transformer",
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
        scorer = OrdinanceScorer(criteria=criteria, model_name="mock-model")
        doc_embeddings = scorer.embed_texts(chunks)
        crit_embeddings = scorer.embed_texts([c["description"] for c in criteria])
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

    def test_doc_embeddings_count_equals_chunk_count(self, criteria):
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=500, overlap=50)
        scorer = OrdinanceScorer(criteria=criteria, model_name="mock-model")
        doc_embeddings = scorer.embed_texts(chunks)
        assert len(doc_embeddings) == len(chunks), (
            "Number of embeddings must match number of chunks"
        )

    def test_crit_embeddings_count_equals_criteria_count(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        criteria = data["criteria"]
        scorer = OrdinanceScorer(criteria=criteria, model_name="mock-model")
        crit_embeddings = scorer.embed_texts([c["description"] for c in criteria])
        assert len(crit_embeddings) == len(criteria)

    def test_result_is_deterministic(self):
        """Running the pipeline twice on the same text must yield the same scores."""
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        criteria = data["criteria"]
        chunks = chunk_text(self.SAMPLE_TEXT, chunk_size=500, overlap=50)
        scorer = OrdinanceScorer(criteria=criteria, model_name="mock-model")

        # Embed once and reuse: model mock is stateful, so we capture embeddings.
        doc_embs = scorer.embed_texts(chunks)
        crit_embs = scorer.embed_texts([c["description"] for c in criteria])
        r1 = scorer.score(doc_chunks=chunks, doc_embeddings=doc_embs,
                          crit_embeddings=crit_embs, top_k=1)
        r2 = scorer.score(doc_chunks=chunks, doc_embeddings=doc_embs,
                          crit_embeddings=crit_embs, top_k=1)
        assert r1["overall_score"] == r2["overall_score"]
        for a, b in zip(r1["criteria_results"], r2["criteria_results"]):
            assert a["score"] == b["score"]


# ---------------------------------------------------------------------------
# 4. Real sample PDF smoke tests (no mocking — uses actual pdfplumber)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_PDF.exists(),
    reason=f"Sample PDF not found at {SAMPLE_PDF}",
)
class TestRealSamplePdf:
    """
    Smoke tests against the real ordinance PDF in sample/.
    These use pdfplumber for genuine text extraction — no mocking.
    They do NOT run the embedding model; text extraction and chunking only.
    """

    @pytest.fixture(scope="class")
    def extracted_text(self):
        from app.utils import extract_text_from_pdf
        with SAMPLE_PDF.open("rb") as fh:
            return extract_text_from_pdf(fh)

    @pytest.fixture(scope="class")
    def chunks(self, extracted_text):
        return chunk_text(extracted_text, chunk_size=2000, overlap=200)

    # --- extraction ---

    def test_pdf_extraction_returns_non_empty_string(self, extracted_text):
        assert isinstance(extracted_text, str)
        assert len(extracted_text.strip()) > 0, "No text extracted from sample PDF"

    def test_pdf_extraction_contains_expected_keywords(self, extracted_text):
        """The ordinance should mention core dark-sky concepts."""
        text_lower = extracted_text.lower()
        keywords = ["lighting", "shielded", "outdoor"]
        for kw in keywords:
            assert kw in text_lower, f"Expected keyword '{kw}' not found in extracted text"

    def test_pdf_extraction_has_reasonable_length(self, extracted_text):
        """A real ordinance PDF should produce at least 1 000 characters."""
        assert len(extracted_text) >= 1_000, (
            f"Extracted text suspiciously short: {len(extracted_text)} chars"
        )

    # --- chunking ---

    def test_chunking_produces_multiple_chunks(self, chunks):
        assert len(chunks) > 1, "Expected multiple chunks from a real ordinance PDF"

    def test_all_chunks_are_non_empty_strings(self, chunks):
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, str) and chunk.strip(), (
                f"Chunk {i} is empty or not a string"
            )

    def test_all_chunks_within_size_limit(self, chunks):
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 2000, f"Chunk {i} exceeds chunk_size: {len(chunk)} chars"
