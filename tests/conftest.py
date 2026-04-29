"""
Shared pytest fixtures for the ordinance-semantic-scorer test suite.

**Design goals**

* **Speed** — Real ``SentenceTransformer`` weights are large and download from the
  internet on first use. Patching ``app.scorer._get_sentence_transformer`` keeps
  unit tests milliseconds-fast and fully offline.
* **Determinism** — Fake embeddings use NumPy RNGs with fixed seeds so cosine
  arithmetic and ordering assertions are reproducible across machines.
* **Parity with production paths** — Fixtures still exercise the real function
  signatures and data shapes the Streamlit app uses; only the heavyweight backends
  are swapped for fakes.

**How to use this module**

Import fixtures by name in test signatures (``criteria_data``, ``minimal_criteria``,
``mock_sentence_transformer``, etc.). Pytest discovers them automatically because
``conftest.py`` lives under ``tests/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CRITERIA_PATH = ROOT / "app" / "criteria.json"


# ---------------------------------------------------------------------------
# Criteria data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def criteria_data() -> list[dict]:
    """Load the real criteria.json once per test session."""
    data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    return data["criteria"]


@pytest.fixture()
def minimal_criteria() -> list[dict]:
    """Two lightweight criteria suitable for fast unit tests."""
    return [
        {
            "title": "Shielding",
            "short": "Lights must be shielded.",
            "description": "All fixtures shall be fully shielded and downcast.",
            "weight": 1.8,
        },
        {
            "title": "Color Temperature",
            "short": "Use amber lights.",
            "description": "Fixtures should emit amber light below 2200K.",
            "weight": 1.0,
        },
    ]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _make_unit_vector(dim: int, seed: int = 0) -> list[float]:
    """Return a deterministic L2-normalised vector of length *dim*."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    v /= np.linalg.norm(v)
    return v.tolist()


@pytest.fixture()
def fake_embeddings() -> list[list[float]]:
    """Three distinct 8-dimensional unit vectors for deterministic tests."""
    return [_make_unit_vector(8, seed=i) for i in range(3)]


@pytest.fixture()
def mock_sentence_transformer():
    """
    Patch *app.scorer._get_sentence_transformer* so that no model download
    occurs. The mock's `encode` method returns normalised float32 vectors of
    dimension 8.
    """
    mock_model = MagicMock()

    def _fake_encode(texts, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=True):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((len(texts), 8)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    mock_model.encode.side_effect = _fake_encode

    with patch(
        "app.scorer._get_sentence_transformer",
        return_value=mock_model,
    ):
        yield mock_model
