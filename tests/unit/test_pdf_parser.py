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
    # Initialize an empty list to store the pages
    pages = []
    # Iterate through the page texts
    for text in page_texts:
        # Create a mock page
        page = MagicMock()
        # Set the extract_text method of the mock page to return the text
        page.extract_text.return_value = text
        # Add the mock page to the list of pages
        pages.append(page)
    # Create a mock PDF object
    mock_pdf = MagicMock()
    # Set the pages of the mock PDF to the list of pages
    mock_pdf.pages = pages
    # Set the __enter__ method of the mock PDF to return the mock PDF
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    # Set the __exit__ method of the mock PDF to return False
    mock_pdf.__exit__ = MagicMock(return_value=False)
    # Return the mock PDF
    return mock_pdf


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    """Test the basic extraction of text from a PDF."""
    def test_single_page_returns_page_text(self):
        """
        Test that a single page returns the page text.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with a single page containing the text "Hello ordinance."
        mock_pdf = _build_mock_pdf(["Hello ordinance."])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text matches the expected text
        assert result == "Hello ordinance."

    def test_multiple_pages_joined_with_double_newline(self):
        """
        Test that multiple pages are joined with double newlines.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with three pages containing the text "Page one.", "Page two.", and "Page three."
        mock_pdf = _build_mock_pdf(["Page one.", "Page two.", "Page three."])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text matches the expected text
        assert result == "Page one.\n\nPage two.\n\nPage three."

    def test_return_type_is_str(self):
        """
        Test that the return type is a string.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with a single page containing the text "Some text."
        mock_pdf = _build_mock_pdf(["Some text."])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text is a string
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Empty / None page handling
# ---------------------------------------------------------------------------


class TestEmptyAndNonePages:
    """Test that empty and None pages are skipped."""
    def test_empty_pdf_no_pages_returns_empty_string(self):
        """
        Test that an empty PDF returns an empty string.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with no pages
        mock_pdf = _build_mock_pdf([])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text is an empty string
        assert result == ""

    @pytest.mark.parametrize("page_values, label", [
        ([None],             "single_none"),
        ([""],               "single_empty_string"),
        ([None, None, None], "all_none_multi_page"),
    ])
    def test_blank_pages_are_skipped(self, page_values, label):
        """
        Test that pages whose extract_text() returns None or an empty string
        are silently dropped, whether it is one page or many.

        Args:
            page_values: List of per-page return values to feed to the mock.
            label: Human-readable identifier shown in the test name.
        Raises:
            AssertionError: If the extracted text is not an empty string.
        """
        # Build a mock PDF whose pages return the supplied values
        mock_pdf = _build_mock_pdf(page_values)
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # All blank/None pages must be dropped → result must be empty
        assert result == ""

    def test_mixed_none_and_text_pages_skips_none(self):
        """
        Test that mixed None and text pages are skipped.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with four pages containing None, "Real text.", None, and "More text."
        mock_pdf = _build_mock_pdf([None, "Real text.", None, "More text."])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text matches the expected text
        assert result == "Real text.\n\nMore text."

# ---------------------------------------------------------------------------
# Separator behaviour
# ---------------------------------------------------------------------------


class TestSeparator:
    """Test the separator behaviour between pages."""
    def test_double_newline_separator_between_pages(self):
        """
        Test that the separator between pages is a double newline.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with two pages containing the text "A" and "B"
        mock_pdf = _build_mock_pdf(["A", "B"])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text contains a double newline
        assert "\n\n" in result
        # Assert that the text "A" appears before the text "B"
        assert result.index("A") < result.index("B")

    def test_page_text_content_preserved_exactly(self):
        """
        Test that the page text content is preserved exactly.
        
        Args:
            mock_pdf: The mock PDF object.
        Returns:
            The extracted text from the PDF.
        Raises:
            AssertionError: If the extracted text does not match the expected text.
        """
        # Build a mock PDF with a single page containing the text "Article 1: All lights shall be shielded.\nSection 2: Uplight prohibited."
        text = "Article 1: All lights shall be shielded.\nSection 2: Uplight prohibited."
        mock_pdf = _build_mock_pdf([text])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text matches the expected text
        assert result == text
