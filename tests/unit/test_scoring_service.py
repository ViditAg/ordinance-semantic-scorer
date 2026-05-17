"""Unit tests for :class:`~app.application.scoring_service.OrdinanceScoringService`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.adapters.chunking_char import FixedCharacterChunker
from app.adapters.text_plain import PlainTextSource
from app.application.scoring_service import OrdinanceScoringService
from app.domain.models import ScoringRequest


def _minimal_criteria():
    return [{"title": "t", "probes": ["d"], "weight": 1.0}]


def test_score_chunks_empty_raises():
    svc = OrdinanceScoringService(
        criteria=_minimal_criteria(),
        model_name="dummy-model",
        chunker=FixedCharacterChunker(100, 10),
    )
    with pytest.raises(ValueError, match="no chunks"):
        svc.score_chunks([], ScoringRequest(top_k=1))


@patch("app.application.scoring_service.OrdinanceScorer")
def test_score_chunks_delegates_to_scorer(mock_scorer_cls):
    mock_scorer = MagicMock()
    mock_scorer.embed_texts.side_effect = [
        [[1.0, 0.0]],
    ]
    mock_scorer.embed_criteria_probes.return_value = [[[0.0, 1.0]]]
    mock_scorer.score.return_value = {"overall_score": 50.0, "criteria_results": []}
    mock_scorer_cls.return_value = mock_scorer

    svc = OrdinanceScoringService(
        criteria=_minimal_criteria(),
        model_name="m",
        chunker=FixedCharacterChunker(10, 2),
    )
    out = svc.score_chunks(["hello"], ScoringRequest(top_k=2))

    assert out.chunks == ["hello"]
    assert out.score_payload["overall_score"] == 50.0
    assert mock_scorer.embed_texts.call_count == 1
    mock_scorer.embed_criteria_probes.assert_called_once()
    mock_scorer.score.assert_called_once()


@patch("app.application.scoring_service.OrdinanceScorer")
def test_score_document_loads_and_chunks(mock_scorer_cls):
    mock_scorer = MagicMock()
    mock_scorer.embed_texts.side_effect = [
        [[1.0, 0.0], [0.0, 1.0]],
    ]
    mock_scorer.embed_criteria_probes.return_value = [[[0.0, 1.0]]]
    mock_scorer.score.return_value = {"overall_score": 10.0, "criteria_results": []}
    mock_scorer_cls.return_value = mock_scorer

    chunker = FixedCharacterChunker(chunk_size=4, overlap=0)
    svc = OrdinanceScoringService(
        criteria=_minimal_criteria(),
        model_name="m",
        chunker=chunker,
    )
    out = svc.score_document(PlainTextSource("abcdefgh"), ScoringRequest(top_k=1))

    assert len(out.chunks) == 2
    mock_scorer.score.assert_called_once()


def test_score_document_empty_raises():
    svc = OrdinanceScoringService(
        criteria=_minimal_criteria(),
        model_name="m",
        chunker=FixedCharacterChunker(10, 2),
    )
    with pytest.raises(ValueError, match="empty document"):
        svc.score_document(PlainTextSource("   "), ScoringRequest(top_k=1))
