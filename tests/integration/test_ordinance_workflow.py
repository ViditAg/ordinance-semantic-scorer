"""
Integration tests for app.ordinance_workflow — full product integration (no Streamlit).

Single place for:
  * criteria.json integrity and load_criteria_bundle() ordering
  * chunking (utils + workflow defaults)
  * end-to-end scoring via run_scoring() with mocked SentenceTransformer
  * build_score_report() contract
  * real sample PDF → extract_text_and_chunk() smoke (pdfplumber)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import ordinance_workflow as ow
from app.scorer import OrdinanceScorer
from app.utils import chunk_text

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_PATH = ROOT / "app" / "criteria.json"
SAMPLE_PDF = ROOT / "sample" / "updated_dark_skies_outdoor_lighting_ordinance.pdf"

EXPECTED_CRITERIA_COUNT = 29
EXPECTED_CRITERIA_KEYS = {"title", "short", "description", "weight"}

REPORT_META_KEYS = {
    "timestamp",
    "overall_score",
    "num_chunks",
    "model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
}

LONG_SAMPLE_TEXT = (
    "All outdoor lighting shall be fully shielded and downcast. "
    "Fixtures must comply with BUG rating B0 U0 G0. "
    "Color temperature must not exceed 3000K for all new installations. "
) * 30

SCORING_SAMPLE_TEXT = (
    "Section 1: Purpose. This ordinance protects dark skies. "
    "Section 2: Scope. Applies to all outdoor artificial lighting. "
    "Section 3: Shielding. All fixtures shall be fully shielded. "
    "Section 4: Color temperature shall not exceed 2200K. "
    "Section 5: Uplighting is prohibited. "
) * 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_encode_factory(dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)

    def _encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    return _encode


def _make_mock_model(dim: int = 16, seed: int = 0) -> MagicMock:
    m = MagicMock()
    m.encode.side_effect = _fake_encode_factory(dim, seed)
    return m


# ---------------------------------------------------------------------------
# criteria.json on disk
# ---------------------------------------------------------------------------


class TestCriteriaJsonFile:
    def test_file_exists(self):
        assert CRITERIA_PATH.exists(), f"criteria.json not found at {CRITERIA_PATH}"

    def test_valid_structure_and_rubric_keys(self):
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and "criteria" in data
        assert isinstance(data["criteria"], list)
        assert len(data["criteria"]) == EXPECTED_CRITERIA_COUNT
        for idx, crit in enumerate(data["criteria"]):
            missing = EXPECTED_CRITERIA_KEYS - crit.keys()
            assert not missing, f"Criterion #{idx + 1} missing keys: {missing}"
            assert crit["weight"] > 0
            assert crit["description"].strip()
        serials = [c.get("serial_number") for c in data["criteria"] if "serial_number" in c]
        assert len(serials) == len(set(serials))


# ---------------------------------------------------------------------------
# load_criteria_bundle + defaults
# ---------------------------------------------------------------------------


class TestWorkflowBundleAndDefaults:
    def test_chunk_defaults_match_utils(self):
        assert ow.DEFAULT_CHUNK_SIZE == 2000
        assert ow.DEFAULT_CHUNK_OVERLAP == 200

    def test_model_default_matches_scorer(self):
        s = OrdinanceScorer([{"title": "t", "description": "d", "weight": 1.0}])
        assert ow.DEFAULT_MODEL_NAME == s.model_name

    def test_bundle_loads_sorted_by_serial(self):
        data, criteria_list = ow.load_criteria_bundle(CRITERIA_PATH)
        assert "criteria" in data
        assert len(criteria_list) == EXPECTED_CRITERIA_COUNT
        serials = [c["serial_number"] for c in criteria_list]
        assert serials == sorted(serials)


# ---------------------------------------------------------------------------
# chunk_text integration (utils)
# ---------------------------------------------------------------------------


class TestChunkingIntegration:
    def test_multiple_chunks_non_empty_stable(self):
        chunks = chunk_text(LONG_SAMPLE_TEXT, chunk_size=400, overlap=50)
        assert len(chunks) > 1
        assert chunk_text(LONG_SAMPLE_TEXT, chunk_size=400, overlap=50) == chunks
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# extract_text_and_chunk (workflow)
# ---------------------------------------------------------------------------


class TestExtractTextAndChunk:
    def test_empty_extraction_yields_no_chunks(self):
        with patch("app.ordinance_workflow.extract_text_from_pdf", return_value="  \n  "):
            raw, chunks = ow.extract_text_and_chunk(io.BytesIO(b"%PDF"))
        assert raw.strip() == ""
        assert chunks == []

    def test_non_empty_text_matches_chunk_text_with_workflow_defaults(self):
        long_text = ("All outdoor lighting shall be shielded. " * 200)
        with patch(
            "app.ordinance_workflow.extract_text_from_pdf",
            return_value=long_text,
        ):
            raw, chunks = ow.extract_text_and_chunk(io.BytesIO(b"%PDF"))
        assert raw == long_text
        assert len(chunks) >= 2
        _, ref = chunk_text(
            long_text,
            chunk_size=ow.DEFAULT_CHUNK_SIZE,
            overlap=ow.DEFAULT_CHUNK_OVERLAP,
        )
        assert chunks == ref


# ---------------------------------------------------------------------------
# run_scoring end-to-end (mocked model)
# ---------------------------------------------------------------------------


class TestRunScoringIntegration:
    @pytest.fixture(autouse=True)
    def _patch_model(self):
        with patch(
            "app.scorer._get_sentence_transformer",
            return_value=_make_mock_model(dim=16, seed=99),
        ):
            yield

    @pytest.fixture()
    def criteria_list(self):
        _, lst = ow.load_criteria_bundle(CRITERIA_PATH)
        return lst

    @pytest.fixture()
    def pipeline_result(self, criteria_list):
        chunks = chunk_text(SCORING_SAMPLE_TEXT, chunk_size=500, overlap=50)
        return ow.run_scoring(chunks, criteria_list, top_k=2)

    def test_result_structure_and_ranges(self, pipeline_result):
        assert "overall_score" in pipeline_result
        assert len(pipeline_result["criteria_results"]) == EXPECTED_CRITERIA_COUNT
        assert 0.0 <= pipeline_result["overall_score"] <= 100.0
        required = {
            "title", "short", "score", "raw_similarity",
            "top_excerpts", "top_scores", "weight",
        }
        for r in pipeline_result["criteria_results"]:
            assert required.issubset(r.keys())
            assert 0.0 <= r["score"] <= 100.0
            assert len(r["top_excerpts"]) <= 2
            assert all(isinstance(e, str) for e in r["top_excerpts"])

    def test_embedding_counts(self, criteria_list):
        chunks = chunk_text(SCORING_SAMPLE_TEXT, chunk_size=500, overlap=50)
        scorer = OrdinanceScorer(criteria=criteria_list, model_name="mock")
        assert len(scorer.embed_texts(chunks)) == len(chunks)
        assert len(scorer.embed_texts([c["description"] for c in criteria_list])) == len(
            criteria_list
        )

    def test_score_idempotent_for_same_embeddings(self, criteria_list):
        chunks = chunk_text(SCORING_SAMPLE_TEXT, chunk_size=500, overlap=50)
        scorer = OrdinanceScorer(criteria=criteria_list, model_name="mock")
        doc_embs = scorer.embed_texts(chunks)
        crit_embs = scorer.embed_texts([c["description"] for c in criteria_list])
        r1 = scorer.score(chunks, doc_embs, crit_embs, top_k=1)
        r2 = scorer.score(chunks, doc_embs, crit_embs, top_k=1)
        assert r1["overall_score"] == r2["overall_score"]


# ---------------------------------------------------------------------------
# build_score_report
# ---------------------------------------------------------------------------


class TestBuildScoreReport:
    def test_meta_contract(self):
        report = ow.build_score_report(
            {"overall_score": 55.5, "criteria_results": []},
            num_chunks=12,
            top_k=3,
            timestamp_utc_iso="2026-01-01T00:00:00.000Z",
        )
        assert REPORT_META_KEYS == set(report["meta"].keys())
        assert report["meta"]["model"] == ow.DEFAULT_MODEL_NAME
        assert report["meta"]["chunk_size"] == ow.DEFAULT_CHUNK_SIZE
        assert report["meta"]["chunk_overlap"] == ow.DEFAULT_CHUNK_OVERLAP
        json.dumps(report)


# ---------------------------------------------------------------------------
# Real PDF smoke (optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason=f"No sample PDF at {SAMPLE_PDF}")
class TestRealSamplePdfWorkflow:
    def test_extract_and_chunk_smoke(self):
        with SAMPLE_PDF.open("rb") as fh:
            raw, chunks = ow.extract_text_and_chunk(fh)
        assert isinstance(raw, str) and len(raw.strip()) > 0
        for kw in ("lighting", "shielded", "outdoor"):
            assert kw in raw.lower()
        assert len(raw) >= 1_000
        assert len(chunks) > 1
        assert all(isinstance(c, str) and c.strip() for c in chunks)
        for c in chunks:
            assert len(c) <= ow.DEFAULT_CHUNK_SIZE
