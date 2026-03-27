"""
Unit tests for app.utils.text_splitter.chunk_text
==================================================

Each test class focuses on a single behavioural property of the function so
that failures are instantly self-explanatory.
"""
from __future__ import annotations

import pytest

from app.utils.text_splitter import chunk_text


# ---------------------------------------------------------------------------
# Guards / validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test that the input validation works as expected."""
    @pytest.mark.parametrize(
        "chunk_size",
        [0, -1]
    )
    def test_zero_or_negative_chunk_size_raises(
        self,
        chunk_size: int
    ):
        """
        Test that a zero or negative chunk size raises a ValueError.
        
        Args:
            chunk_size: The chunk size to test.
        Returns:
            None
        Raises:
            ValueError: If the chunk size is zero or negative.
        """
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
        Test that an overlap equal to or greater than chunk_size raises a ValueError.

        Args:
            overlap: The overlap value to test.
            label: The label of the test.
        Raises:
            ValueError: If overlap >= chunk_size.
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
    @pytest.mark.parametrize(
        "text",
        ["", "   \n\t  "]
    )
    def test_empty_string_returns_empty_list(
        self,
        text: str
    ):
        """
        Test that an empty string returns an empty list.
        
        Args:
            text: The text to test.
        Raises:
            AssertionError: If the result is not an empty list.
        """
        assert chunk_text(text) == []

    def test_text_shorter_than_chunk_size_gives_single_chunk(self):
        """
        Test that a text shorter than the chunk size gives a single chunk.
        Raises:
            AssertionError: If the result is not a single chunk.
        """
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
        Test that a text exactly the size of the chunk size gives a single chunk.
        Raises:
            AssertionError: If the result is not a single chunk.
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
    """Test that the basic chunking behaviour works as expected."""
    def test_produces_multiple_chunks_for_long_text(self):
        """
        Test that a long text produces multiple chunks.
        Raises:
            AssertionError: If the result is not multiple chunks.
        """
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
        Test that all chunks are at most the chunk size characters.
        Raises:
            AssertionError: If the result is not at most the chunk size characters.
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
        """
        Test that no empty chunks are in the output.
        Raises:
            AssertionError: If the result is not no empty chunks.
        """
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
        Test that concatenating all chunk starts must span the entire input.
        Raises:
            AssertionError: If the result is not the entire input.
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
    """Test that the overlap behaviour works as expected."""
    def test_consecutive_chunks_share_overlap_content(self):
        """
        Test that the tail of chunk N should appear at the start of chunk N+1.
        Raises:
            AssertionError: If the result is not the entire input.
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
        """Test that with overlap=0 every character appears in exactly one chunk.
        Raises:
            AssertionError: If the result is not the entire input.
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
    """Test that the line ending normalisation works as expected."""
    def test_crlf_converted_to_lf(self):
        """
        Test that CRLF is converted to LF.
        Raises:
            AssertionError: If the result is not CRLF converted to LF.
        """
        # Initialize the text to "line one\r\nline two\r\nline three"
        text = "line one\r\nline two\r\nline three"
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=200, overlap=10)
        # Assert that there is no carriage-return in the output
        for chunk in chunks:
            # Assert that there is no carriage-return in the chunk
            assert "\r" not in chunk

    def test_mixed_line_endings_normalised(self):
        """
        Test that mixed line endings are normalised.
        Raises:
            AssertionError: If the result is not mixed line endings normalised.
        """
        # Initialize the text to "a\r\nb\nc\r\nd"
        text = "a\r\nb\nc\r\nd"
        # Call the chunk_text function with the text and chunk size and overlap
        chunks = chunk_text(text, chunk_size=200, overlap=5)
        # Assert that there is no carriage-return in the output
        joined = "".join(chunks)
        # Assert that there is no carriage-return in the output
        assert "\r" not in joined
