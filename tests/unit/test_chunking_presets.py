"""Tests for ``app.chunking_presets``."""
from __future__ import annotations

import pytest

from app.chunking_presets import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    describe_preset,
    preset_label,
    resolve_chunking,
)


def test_balanced_matches_module_defaults():
    cs, ov = resolve_chunking("balanced", 9999, 9999)
    assert cs == DEFAULT_CHUNK_SIZE
    assert ov == DEFAULT_CHUNK_OVERLAP


def test_custom_uses_arguments():
    assert resolve_chunking("custom", 1800, 120) == (1800, 120)


def test_fine_and_coarse_are_ordered_sizes():
    fine_cs, _ = resolve_chunking("fine", 0, 0)
    bal_cs, _ = resolve_chunking("balanced", 0, 0)
    coarse_cs, _ = resolve_chunking("coarse", 0, 0)
    assert fine_cs < bal_cs < coarse_cs


def test_preset_label_custom():
    assert "Custom" in preset_label("custom")


def test_describe_nonempty():
    for key in ("balanced", "fine", "coarse", "custom"):
        assert len(describe_preset(key)) > 5


def test_unknown_preset_key_raises():
    with pytest.raises(KeyError):
        resolve_chunking("nonexistent", 1000, 100)
