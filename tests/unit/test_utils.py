"""
Unit tests for ``app.utils`` — PDF text extraction, character chunking, and
embedding geometry helpers used by the scorer.

**Why mock pdfplumber?**

``extract_text_from_pdf`` only orchestrates ``pdfplumber.open`` and per-page
``extract_text`` calls. By returning synthetic ``MagicMock`` pages we can assert
joining behaviour, blank-page skipping, and separators **without** binary PDF
fixtures for every edge case.

**chunk_text coverage**

We validate invariants (empty input, overlap rules, CRLF normalization) and a few
behavioural properties (coverage of long strings, overlap between windows) that
protect the downstream embedding step from surprising segmentation.
"""
from __future__ import annotations

import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.utils import (
    chunk_text,
    cosine_similarities_array,
    extract_text_from_pdf,
    similarity_to_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_file():
    """
    Return a tiny in-memory ``BytesIO`` standing in for an uploaded PDF.

    The bytes are not parsed because every test patches ``pdfplumber.open``; the
    object only needs to satisfy the "file-like" contract ``pdfplumber`` expects.
    """
    return io.BytesIO(b"%PDF-1.4 fake")


def _build_mock_pdf(page_texts: list):
    """
    Construct a context-managed fake PDF whose ``pages`` mirror *page_texts*.

    **Args:**
        page_texts: Ordered ``extract_text()`` return values, one per mock page.

    **Returns:**
        A ``MagicMock`` configured so ``with pdfplumber.open(...) as pdf`` returns
        the same object exposing ``pdf.pages`` and per-page ``extract_text``.
    """
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
    """Happy-path ``extract_text_from_pdf`` behaviour with mocked pdfplumber."""

    def test_single_page_returns_page_text(self):
        """
        A PDF whose first (only) page returns a short string should round-trip that
        string unchanged through ``extract_text_from_pdf``.
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
        Multi-page PDFs must join non-empty page texts with ``\\n\\n`` so chunkers
        see a weak visual break between original pages.
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
        """Contract check: successful extraction always returns a ``str``, never ``None``."""
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
    """``extract_text_from_pdf`` must tolerate empty and falsy page payloads."""

    def test_empty_pdf_no_pages_returns_empty_string(self):
        """Zero pages → join of an empty list → ``""`` (not ``None``)."""
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
        Pages whose ``extract_text()`` returns ``None`` or ``""`` contribute nothing.

        **Parametrized cases:** single blank page, single empty string, or multiple
        ``None`` pages — all should collapse to an empty aggregate string.

        The ``label`` parameter exists only to give each parametrized case a stable
        pytest node name for CI logs.
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
        Interleaved ``None`` pages should disappear while real text pages stay in
        order, still separated by the double-newline convention.
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
    """Explicit assertions on the ``\\n\\n`` join strategy between pages."""

    def test_double_newline_separator_between_pages(self):
        """Two non-empty pages must contain a blank-line gap (``\\n\\n``) between them."""
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
        """Internal newlines inside a page must not be stripped or altered."""
        # Build a mock PDF with a single page containing the text "Article 1: All lights shall be shielded.\nSection 2: Uplight prohibited."
        text = "Article 1: All lights shall be shielded.\nSection 2: Uplight prohibited."
        mock_pdf = _build_mock_pdf([text])
        # Patch the pdfplumber.open function to return the mock PDF
        with patch("pdfplumber.open", return_value=mock_pdf):
            # Extract the text from the PDF using the extract_text_from_pdf function
            result = extract_text_from_pdf(_fake_file())
        # Assert that the extracted text matches the expected text
        assert result == text



# ---------------------------------------------------------------------------
# Guards / validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Guard rails on ``chunk_text`` hyperparameters."""

    @pytest.mark.parametrize(
        "chunk_size",
        [0, -1]
    )
    def test_zero_or_negative_chunk_size_raises(
        self,
        chunk_size: int
    ):
        """Non-positive ``chunk_size`` is invalid because windows cannot advance."""
        # Call the chunk_text function with the chunk size and expect a ValueError
        with pytest.raises(
            ValueError,
            match="chunk_size must be positive"
        ):
            chunk_text("hello", chunk_size=chunk_size)

    @pytest.mark.parametrize("overlap, label", [
        (5,  "equal"),
        (10, "greater"),
    ])
    def test_invalid_overlap_raises(
        self,
        overlap: int,
        label: str
    ):
        """
        ``overlap >= chunk_size`` would stall or walk backwards; reject early.

        ``label`` distinguishes parametrized variants in pytest output only.
        """
        with pytest.raises(
            ValueError,
            match="overlap must be smaller than chunk_size"
        ):
            # Call the chunk_text function with the text and chunk size and overlap
            chunk_text(
                "hello world",
                chunk_size=5,
                overlap=overlap
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions for ``chunk_text`` (empty body, whitespace-only)."""

    @pytest.mark.parametrize(
        "text",
        ["", "   \n\t  "]
    )
    def test_empty_string_returns_empty_list(
        self,
        text: str
    ):
        """After strip, nothing to embed → no chunks (empty list, not ``[""]``)."""
        assert chunk_text(text) == []

    def test_text_shorter_than_chunk_size_gives_single_chunk(self):
        """Short documents should produce exactly one window spanning the full text."""
        # Call the chunk_text function with the text and chunk size and overlap
        text = "Short text."
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(
            text,
            chunk_size=100,
            overlap=10
        )
        # Assert that the result is a single chunk
        assert len(chunks) == 1
        # Assert that the chunk is the text
        assert chunks[0] == text.strip()

    def test_text_exactly_chunk_size_gives_single_chunk(self):
        """
        When ``len(text) == chunk_size`` the first window already reaches EOF, so the
        ``while`` loop should terminate after one iteration (no zero-length tail).
        """
        # Initialize the text to "a" repeated 50 times
        text = "a" * 50
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        # Assert that the result is a single chunk
        assert len(chunks) == 1
        # Assert that the chunk is the text
        assert chunks[0] == text


# ---------------------------------------------------------------------------
# Basic chunking behaviour
# ---------------------------------------------------------------------------


class TestBasicChunking:
    """Smoke tests that long inputs actually fan out into multiple windows."""

    def test_produces_multiple_chunks_for_long_text(self):
        """A 1000-character run with a 200-char window must create >1 chunk."""
        # Initialize the text to "x" repeated 1000 times
        text = "x" * 1000
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        # Assert that the result is greater than 1 chunk
        assert len(chunks) > 1
        # Assert that the chunks are not empty
        assert all(chunk.strip() for chunk in chunks)

    def test_all_chunks_at_most_chunk_size_characters(self):
        """
        **Post-strip upper bound:** each slice is at most ``chunk_size`` characters
        before ``.strip()``; stripping only removes edges, never grows the string.
        """
        # Initialize the text to "word " repeated 500 times
        text = "word " * 500  # 2500 chars
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        # Assert that all chunks are at most the chunk size characters
        for chunk in chunks:
            # Assert that the chunk is at most the chunk size characters
            assert len(chunk) <= 300

    def test_no_empty_chunks_in_output(self):
        """Every emitted chunk should contain non-whitespace payload."""
        # Initialize the text to "hello world " repeated 200 times
        text = "hello world " * 200
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Assert that no empty chunks are in the output
        for chunk in chunks:
            # Assert that the chunk is not empty
            assert chunk.strip() != ""

    def test_chunks_cover_full_text(self):
        """
        Loose **coverage** heuristic: first chunk aligns with the document start and
        the last chunk aligns with the tail (embedders rely on this intuition).
        """
        # Initialize the text to "abcdefghij" repeated 100 times
        text = "abcdefghij" * 100  # 1000 chars
        # Initialize the chunk size to 100
        chunk_size = 100
        # Initialize the overlap to 10
        overlap = 10
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        # Assert that the first chunk starts with the beginning of the text
        # First chunk must start with the beginning of the text.
        assert text.startswith(chunks[0])
        # Last chunk must end with (or contain) the end of the text.
        assert text.endswith(chunks[-1]) or chunks[-1] in text


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------


class TestOverlap:
    """Properties that only make sense when ``overlap > 0``."""

    def test_consecutive_chunks_share_overlap_content(self):
        """
        With generous overlap, some substring from the end of chunk *i* should recur
        inside chunk *i+1* (exact equality can be perturbed by ``strip()``).
        """
        # Use a large overlap relative to chunk_size to make it obvious.
        text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=50, overlap=20)
        # Assert that there are at least 2 chunks
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # Loop through the chunks
        for i in range(len(chunks) - 1):
            # Get the tail of the current chunk
            tail = chunks[i][-20:].strip()    # last 20 chars of current chunk
            # Get the head of the next chunk
            head = chunks[i + 1][:20].strip()  # first 20 chars of next chunk
            # Assert that the tail of the current chunk is in the next chunk
            assert tail[:10] in chunks[i + 1] or head[:10] in chunks[i]

    def test_zero_overlap_no_repeated_content(self):
        """
        ``overlap=0`` turns the splitter into a partition: joining chunks rebuilds
        the original string with no duplicated characters across chunk boundaries.
        """
        # Initialize the text to "abcdefghijklmnopqrstuvwxyz" repeated 4 times
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=13, overlap=0)
        # Reconstruct and verify coverage.
        # Assert that the reconstructed text is the same as the original text
        reconstructed = "".join(chunks)
        assert reconstructed == text


# ---------------------------------------------------------------------------
# Line-ending normalisation
# ---------------------------------------------------------------------------


class TestLineEndingNormalisation:
    """``chunk_text`` pre-processes ``\\r\\n`` so excerpts look Unix-native."""

    def test_crlf_converted_to_lf(self):
        """Windows-style terminators must not survive inside returned chunks."""
        # Initialize the text to "line one\r\nline two\r\nline three"
        text = "line one\r\nline two\r\nline three"
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=200, overlap=10)
        # Assert that there is no carriage-return in the output
        for chunk in chunks:
            # Assert that there is no carriage-return in the chunk
            assert "\r" not in chunk

    def test_mixed_line_endings_normalised(self):
        """Mixed ``\\r\\n`` and bare ``\\n`` still yield a carriage-return-free join."""
        # Initialize the text to "a\r\nb\nc\r\nd"
        text = "a\r\nb\nc\r\nd"
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=200, overlap=5)
        # Assert that there is no carriage-return in the output
        joined = "".join(chunks)
        # Assert that there is no carriage-return in the output
        assert "\r" not in joined


# ---------------------------------------------------------------------------
# similarity_to_score & cosine_similarities_array (scoring geometry)
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else v


def _orthogonal_basis(dim: int, n: int) -> list[list[float]]:
    assert n <= dim
    eye = np.eye(dim)
    return [eye[i].tolist() for i in range(n)]


def _manual_row_cosines(vec: list[float], matrix: list[list[float]]) -> list[float]:
    v = np.array(vec, dtype=float)
    vn = np.linalg.norm(v)
    out = []
    for row in matrix:
        r = np.array(row, dtype=float)
        rn = np.linalg.norm(r)
        denom = vn * rn
        if denom == 0:
            out.append(float("nan"))
        else:
            out.append(float(np.dot(v, r) / denom))
    return out


class TestSimilarityToScore:
    def test_zero_similarity_gives_zero_score(self):
        assert similarity_to_score(0.0) == 0.0

    def test_one_similarity_gives_hundred(self):
        assert similarity_to_score(1.0) == pytest.approx(100.0)

    def test_half_similarity_gives_fifty(self):
        assert similarity_to_score(0.5) == pytest.approx(50.0)

    def test_negative_similarity_clamped_to_zero(self):
        assert similarity_to_score(-0.5) == 0.0
        assert similarity_to_score(-1.0) == 0.0

    def test_arbitrary_positive_value(self):
        sim = 0.73
        assert similarity_to_score(sim) == pytest.approx(73.0)

    def test_result_is_rounded_to_two_decimal_places(self):
        result = similarity_to_score(0.12345)
        assert result == round(result, 2)


class TestCosineSimilaritiesArray:
    def test_identical_vector_gives_one(self):
        v = [1.0, 0.0, 0.0]
        M = [[1.0, 0.0, 0.0]]
        sims = cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vector_gives_zero(self):
        v = [1.0, 0.0, 0.0]
        M = [[0.0, 1.0, 0.0]]
        sims = cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vector_gives_minus_one(self):
        v = [1.0, 0.0]
        M = [[-1.0, 0.0]]
        sims = cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_in_matrix_gets_safe_denominator(self):
        v = [1.0, 0.0]
        M = [[0.0, 0.0], [1.0, 0.0]]
        sims = cosine_similarities_array(v, M)
        assert not any(math.isnan(s) for s in sims.tolist())

    def test_multiple_rows_computed_correctly(self):
        v = [1.0, 0.0, 0.0]
        M = _orthogonal_basis(3, 3)
        sims = cosine_similarities_array(v, M)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)
        assert sims[2] == pytest.approx(0.0, abs=1e-6)

    def test_returns_numpy_array(self):
        v = [1.0, 0.0]
        M = [[1.0, 0.0]]
        result = cosine_similarities_array(v, M)
        assert isinstance(result, np.ndarray)

    def test_matches_per_row_cosine_formula_nonzero_rows(self):
        v = _unit([1.0, 2.0, -0.5])
        M = [
            _unit([0.1, -3.0, 2.0]),
            _unit([2.0, 2.0, 2.0]),
        ]
        sims = cosine_similarities_array(v, M)
        expected = _manual_row_cosines(v, M)
        for got, exp in zip(sims.tolist(), expected):
            assert got == pytest.approx(exp, abs=1e-6)
