"""Calibration grid defaults stay small enough for ``scripts/calibrate.py --max-pairs``."""

from __future__ import annotations

from app.chunking_presets import (
    SWEEP_CHUNK_SIZE_MAX_DEFAULT,
    SWEEP_CHUNK_SIZE_MIN_DEFAULT,
    SWEEP_CHUNK_SIZE_STEP_DEFAULT,
    SWEEP_OVERLAP_MAX_DEFAULT,
    SWEEP_OVERLAP_MIN_DEFAULT,
    SWEEP_OVERLAP_STEP_DEFAULT,
)
from app.stability import valid_chunk_pairs


def test_default_sweep_fits_calibrate_max_pairs_budget():
    sizes = list(
        range(
            SWEEP_CHUNK_SIZE_MIN_DEFAULT,
            SWEEP_CHUNK_SIZE_MAX_DEFAULT + 1,
            SWEEP_CHUNK_SIZE_STEP_DEFAULT,
        )
    )
    overlaps = list(
        range(
            SWEEP_OVERLAP_MIN_DEFAULT,
            SWEEP_OVERLAP_MAX_DEFAULT + 1,
            SWEEP_OVERLAP_STEP_DEFAULT,
        )
    )
    pairs = valid_chunk_pairs(sizes, overlaps)
    assert pairs, "expected non-empty sweep grid"
    assert len(pairs) <= 72, (
        f"default grid has {len(pairs)} pairs; raise calibrate --max-pairs "
        "or shrink SWEEP_* presets"
    )

def test_smallest_chunk_exceeds_largest_overlap():
    assert SWEEP_OVERLAP_MAX_DEFAULT < SWEEP_CHUNK_SIZE_MIN_DEFAULT
