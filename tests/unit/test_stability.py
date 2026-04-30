"""Unit tests for ``app.stability`` (chunk sweep helpers)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.stability import (
    annotate_neighbor_stability,
    composite_rank_score,
    run_chunk_sweep,
    valid_chunk_pairs,
)


class TestValidChunkPairs:
    def test_drops_overlap_ge_size(self):
        pairs = valid_chunk_pairs([500, 1000], [100, 500, 600])
        assert (500, 100) in pairs
        assert (1000, 100) in pairs
        assert (500, 500) not in pairs
        assert (500, 600) not in pairs

    def test_cartesian_count_when_all_valid(self):
        pairs = valid_chunk_pairs([1000, 2000], [50, 100])
        assert len(pairs) == 4


class TestAnnotateNeighborStability:
    def test_flat_grid_has_low_neighbor_stdev(self):
        rows = [
            {"chunk_size": 1000, "overlap": 50, "overall_score": 50.0},
            {"chunk_size": 1000, "overlap": 100, "overall_score": 50.0},
            {"chunk_size": 2000, "overlap": 50, "overall_score": 50.0},
            {"chunk_size": 2000, "overlap": 100, "overall_score": 50.0},
        ]
        out = annotate_neighbor_stability(rows)
        for r in out:
            assert r["neighbor_score_stdev"] == pytest.approx(0.0, abs=1e-6)

    def test_spike_middle_has_lower_neighbor_stdev_than_edges(self):
        """
        On a 1×3 size grid with a spike at the middle, stdev is taken over the full
        3-cell window including the center — so the middle cell averages [50, 90, 50]
        and has lower spread than edge cells, which only see [50, 90].
        """
        rows = [
            {"chunk_size": 1500, "overlap": 100, "overall_score": 50.0},
            {"chunk_size": 2000, "overlap": 100, "overall_score": 90.0},
            {"chunk_size": 2500, "overlap": 100, "overall_score": 50.0},
        ]
        out = annotate_neighbor_stability(rows)
        by_key = {(r["chunk_size"], r["overlap"]): r["neighbor_score_stdev"] for r in out}
        assert by_key[(2000, 100)] < by_key[(1500, 100)]
        assert by_key[(2000, 100)] < by_key[(2500, 100)]
        assert by_key[(1500, 100)] == pytest.approx(by_key[(2500, 100)])


class TestCompositeRankScore:
    def test_prefers_low_stdev_when_weight_positive(self):
        assert composite_rank_score(80.0, 5.0, 1.0) < composite_rank_score(80.0, 1.0, 1.0)


class TestRunChunkSweep:
    def test_mocked_scorer_runs_each_pair(self):
        criteria = [{"title": "A", "description": "d", "weight": 1.0}]

        mock_scorer = MagicMock()
        mock_scorer.embed_texts.side_effect = lambda texts: [[0.0, 1.0]] * len(texts)
        mock_scorer.score.return_value = {
            "overall_score": 42.5,
            "criteria_results": [{"score": 42.5}],
        }

        with patch("app.stability.OrdinanceScorer", return_value=mock_scorer):
            rows = run_chunk_sweep(
                raw_text="word " * 500,
                criteria=criteria,
                model_name="mock",
                chunk_sizes=[400, 500],
                overlaps=[50],
                top_k=1,
            )

        assert len(rows) == 2
        assert mock_scorer.embed_texts.call_count >= 3
        assert all("neighbor_score_stdev" in r for r in rows)
        assert rows[0]["overall_score"] == 42.5
