"""
Central defaults for the Streamlit app and reproducible batch runs.

The web UI does not expose embedding model or chunk hyperparameters; change
constants here (and in ``chunking_presets`` for segmentation) then redeploy.
"""

from __future__ import annotations

from app.chunking_presets import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
]
