"""Tests for :mod:`app.extraction_quality`."""

from __future__ import annotations

from app.extraction_quality import assess_extraction_quality


def test_empty_string_not_usable():
    q = assess_extraction_quality("")
    assert q.char_count == 0
    assert q.likely_image_only_or_corrupt
    assert not q.looks_usable


def test_normal_ordinance_prose_usable():
    text = (
        "Section 1. All outdoor lighting shall be fully shielded. " * 80
    )
    q = assess_extraction_quality(text)
    assert q.looks_usable
    assert not q.likely_image_only_or_corrupt
    assert q.char_count > 500


def test_garbage_low_alpha_not_usable():
    text = "#####" * 200
    q = assess_extraction_quality(text)
    assert not q.looks_usable
