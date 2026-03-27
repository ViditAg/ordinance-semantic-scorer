"""
Tests for streamlit_app.py via Streamlit AppTest (script runs in-process).

Validates that the app script executes, surfaces expected chrome (title,
sidebar controls, rubric expander), and does not raise script exceptions.

Heavy ML and PDF parsing are not exercised here; see test_ordinance_workflow.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_APP = ROOT / "streamlit_app.py"

pytest.importorskip("streamlit.testing.v1", reason="Streamlit AppTest required")

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def at() -> AppTest:
    assert STREAMLIT_APP.is_file(), f"Missing {STREAMLIT_APP}"
    return AppTest.from_file(STREAMLIT_APP, default_timeout=15)


def test_streamlit_app_runs_without_exception(at: AppTest):
    at.run()
    if len(at.exception) > 0:
        exc = at.exception[0]
        msg = getattr(exc, "message", None) or getattr(exc, "value", str(exc))
        pytest.fail(f"App raised: {msg}")


def test_streamlit_app_main_title(at: AppTest):
    at.run()
    assert len(at.title) >= 1
    assert "Dark Sky Ordinance Semantic Scorer" in at.title[0].value


def test_streamlit_app_sidebar_top_k_number_input(at: AppTest):
    at.run()
    ni = at.sidebar.number_input
    assert len(ni) >= 1
    labels = [w.label.lower() for w in ni]
    assert any("excerpt" in lb or "top" in lb for lb in labels)


def test_streamlit_app_scoring_rubric_expander(at: AppTest):
    at.run()
    assert len(at.expander) >= 1
    labels = [e.label for e in at.expander]
    assert any("Scoring rubric" in lab and "criteria" in lab for lab in labels)


def test_streamlit_app_intro_mentions_pdf_upload(at: AppTest):
    """AppTest has no file_uploader widget yet; check copy instead."""
    at.run()
    bodies = [m.value for m in at.markdown]
    assert any("Upload" in b and "PDF" in b for b in bodies)
