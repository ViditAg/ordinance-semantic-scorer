"""
Unit tests for app.utils.pdf_parser.extract_text_from_pdf
==========================================================

pdfplumber is mocked throughout so no actual PDF file is needed.
The shared ``mock_pdfplumber_open`` fixture (defined in conftest.py) builds a
context-manager-compatible mock whose pages return the strings we supply.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from app.utils.pdf_parser import extract_text_from_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_file():
    """Return a minimal binary file-like object (content doesn't matter because
    pdfplumber.open is fully mocked)."""
    return io.BytesIO(b"%PDF-1.4 fake")


def _build_mock_pdf(page_texts: list):
    """Build a mock pdfplumber PDF whose pages return *page_texts* in order."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    mock_pdf = MagicMock()
    mock_pdf.pages = pages
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_single_page_returns_page_text(self):
        mock_pdf = _build_mock_pdf(["Hello ordinance."])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == "Hello ordinance."

    def test_multiple_pages_joined_with_double_newline(self):
        mock_pdf = _build_mock_pdf(["Page one.", "Page two.", "Page three."])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == "Page one.\n\nPage two.\n\nPage three."

    def test_return_type_is_str(self):
        mock_pdf = _build_mock_pdf(["Some text."])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Empty / None page handling
# ---------------------------------------------------------------------------


class TestEmptyAndNonePages:
    def test_empty_pdf_no_pages_returns_empty_string(self):
        mock_pdf = _build_mock_pdf([])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == ""

    def test_page_returning_none_is_skipped(self):
        """extract_text() can return None for image-based pages; they must be
        silently dropped."""
        mock_pdf = _build_mock_pdf([None])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == ""

    def test_page_returning_empty_string_is_skipped(self):
        mock_pdf = _build_mock_pdf([""])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == ""

    def test_mixed_none_and_text_pages_skips_none(self):
        """Only pages with real text contribute to the output."""
        mock_pdf = _build_mock_pdf([None, "Real text.", None, "More text."])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == "Real text.\n\nMore text."

    def test_all_none_pages_returns_empty_string(self):
        mock_pdf = _build_mock_pdf([None, None, None])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == ""


# ---------------------------------------------------------------------------
# Separator behaviour
# ---------------------------------------------------------------------------


class TestSeparator:
    def test_double_newline_separator_between_pages(self):
        mock_pdf = _build_mock_pdf(["A", "B"])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert "\n\n" in result
        assert result.index("A") < result.index("B")

    def test_page_text_content_preserved_exactly(self):
        text = "Article 1: All lights shall be shielded.\nSection 2: Uplight prohibited."
        mock_pdf = _build_mock_pdf([text])
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_text_from_pdf(_fake_file())
        assert result == text
