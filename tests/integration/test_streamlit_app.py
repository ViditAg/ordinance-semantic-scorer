"""
Smoke tests for ``streamlit_app.py`` using Streamlit's ``AppTest`` API.

Uses ``streamlit.testing.v1.AppTest`` (Streamlit version is pinned in ``requirements.txt``). Tests run the
script once and assert on the initial layout **without** uploading a PDF, so no
embedding model is loaded.

``AppTest.file_uploader`` exists only in newer Streamlit releases; we assert on
title and body markdown instead so the smoke test passes across minor versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_APP = REPO_ROOT / "streamlit_app.py"


def test_streamlit_app_initial_render():
    at = AppTest.from_file(str(STREAMLIT_APP), default_timeout=30)
    at.run()
    assert not at.exception, repr(at.exception)
    title_values = [t.value for t in at.title]
    assert any("Dark Sky Ordinance Semantic Scorer" in (v or "") for v in title_values)
    intro = [m.value for m in at.markdown if m.value]
    assert any(
        "Upload a dark sky or outdoor lighting ordinance PDF" in (v or "")
        for v in intro
    ), intro
