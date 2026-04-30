"""
Corpus-level integration checks for optional benchmark PDFs.

PDF paths come from ``tests/fixtures/ordinances/manifest.json`` (repo-root-relative
``path``). Missing files are skipped so CI stays green until you commit ordinances.

**Layers**

1. **Extraction** — real ``pdfplumber`` path; ``assess_extraction_quality`` flags scans.
2. **Chunking** — same defaults as production-ish tests in ``test_pipeline``.
3. **Scoring** — mocked SentenceTransformer for fast CI; optional ``slow`` tests with
   real models when ``ORDINANCE_RUN_SCORING_BENCHMARK=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.adapters.chunking_char import FixedCharacterChunker
from app.application.scoring_service import OrdinanceScoringService
from app.defaults import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_SENTENCE_TRANSFORMER_MODEL
from app.domain.models import ScoringRequest
from app.extraction_quality import assess_extraction_quality
from app.utils import extract_text_from_pdf

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "ordinances" / "manifest.json"


def _load_manifest():
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data["documents"]


def _doc_path(entry: dict) -> Path:
    return ROOT / entry["path"]


def _fake_encode_factory(dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)

    def _encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    return _encode


def _make_mock_model(dim: int = 16, seed: int = 0) -> MagicMock:
    m = MagicMock()
    m.encode.side_effect = _fake_encode_factory(dim, seed)
    return m


@pytest.fixture(scope="module")
def criteria_list():
    crit_path = ROOT / "app" / "criteria.json"
    return json.loads(crit_path.read_text(encoding="utf-8"))["criteria"]


class TestManifestDocumentsExtraction:
    @pytest.mark.parametrize("entry", _load_manifest(), ids=lambda e: e["id"])
    def test_extraction_quality_when_pdf_present(self, entry):
        path = _doc_path(entry)
        if not path.exists():
            pytest.skip(f"PDF not present: {path}")

        raw = extract_text_from_pdf(path.open("rb"))
        q = assess_extraction_quality(
            raw,
            min_chars=int(entry.get("min_chars", 500)),
        )
        assert q.char_count >= int(entry["min_chars"]), (
            f"{entry['id']}: extracted only {q.char_count} chars "
            f"(manifest min_chars={entry['min_chars']}). "
            "If this is a scan, add OCR or a text-layer PDF."
        )
        assert q.looks_usable, (
            f"{entry['id']}: extraction looks unusable "
            f"(alpha_ratio={q.alpha_ratio}, word_like={q.word_like_count}). "
            "Suspect image-only PDF or corrupt extract."
        )

        text_lower = raw.lower()
        for kw in entry.get("keywords", []):
            assert kw.lower() in text_lower, (
                f"{entry['id']}: expected keyword {kw!r} not found — "
                "wrong file or extraction failed mid-document."
            )


class TestManifestDocumentsScoringMocked:
    @pytest.mark.parametrize("entry", _load_manifest(), ids=lambda e: e["id"])
    def test_end_to_end_score_when_pdf_present(self, entry, criteria_list):
        path = _doc_path(entry)
        if not path.exists():
            pytest.skip(f"PDF not present: {path}")

        raw = extract_text_from_pdf(path.open("rb"))
        if not assess_extraction_quality(raw, min_chars=int(entry["min_chars"])).looks_usable:
            pytest.skip(f"{entry['id']}: extraction quality check failed")

        chunker = FixedCharacterChunker(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        chunks = chunker.chunk(raw)
        assert len(chunks) >= 1

        mock = _make_mock_model(dim=16, seed=hash(entry["id"]) % 2**32)
        with patch("app.scorer._get_sentence_transformer", return_value=mock):
            svc = OrdinanceScoringService(
                criteria=criteria_list,
                model_name="mock-for-integration",
                chunker=chunker,
            )
            result = svc.score_chunks(chunks, ScoringRequest(top_k=1))

        payload = result.score_payload
        assert "overall_score" in payload
        assert 0.0 <= payload["overall_score"] <= 100.0
        assert len(payload["criteria_results"]) == len(criteria_list)


@pytest.mark.slow
class TestManifestDocumentsScoringRealModel:
    """Set ``ORDINANCE_RUN_SCORING_BENCHMARK=1`` locally to regression-test real scores."""

    def test_benchmark_minimum_scores_when_enabled(self, criteria_list):
        if os.environ.get("ORDINANCE_RUN_SCORING_BENCHMARK") != "1":
            pytest.skip("Set ORDINANCE_RUN_SCORING_BENCHMARK=1 to run real-model benchmarks")

        chunker = FixedCharacterChunker(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        svc = OrdinanceScoringService(
            criteria=criteria_list,
            model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            chunker=chunker,
        )

        for entry in _load_manifest():
            path = _doc_path(entry)
            if not path.exists():
                continue
            min_score = entry.get("benchmark_min_overall_score")
            if min_score is None:
                continue

            raw = extract_text_from_pdf(path.open("rb"))
            if not assess_extraction_quality(raw, min_chars=int(entry["min_chars"])).looks_usable:
                pytest.fail(f"{entry['id']}: unusable extraction; fix PDF before benchmarking")

            result = svc.score_chunks(chunker.chunk(raw), ScoringRequest(top_k=1))
            got = result.score_payload["overall_score"]
            assert got >= float(min_score), (
                f"{entry['id']}: overall_score {got:.1f} below benchmark {min_score} "
                f"for model {DEFAULT_SENTENCE_TRANSFORMER_MODEL}. "
                "Tune manifest thresholds after rubric/model lock-in."
            )
