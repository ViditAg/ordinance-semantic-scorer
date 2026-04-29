"""Sanity checks for ``app.defaults``."""
from __future__ import annotations

from app import defaults


def test_default_model_is_non_empty_string():
    assert isinstance(defaults.DEFAULT_SENTENCE_TRANSFORMER_MODEL, str)
    assert len(defaults.DEFAULT_SENTENCE_TRANSFORMER_MODEL) > 3


def test_chunk_constants_reexported():
    assert defaults.DEFAULT_CHUNK_SIZE > 0
    assert 0 < defaults.DEFAULT_CHUNK_OVERLAP < defaults.DEFAULT_CHUNK_SIZE
