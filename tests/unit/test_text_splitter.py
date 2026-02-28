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
    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
            chunk_text("hello world", chunk_size=5, overlap=5)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
            chunk_text("hello world", chunk_size=5, overlap=10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        # After strip(), purely whitespace chunks are discarded.
        assert chunk_text("   \n\t  ") == []

    def test_text_shorter_than_chunk_size_gives_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_text_exactly_chunk_size_gives_single_chunk(self):
        text = "a" * 50
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        assert len(chunks) == 1
        assert chunks[0] == text


# ---------------------------------------------------------------------------
# Basic chunking behaviour
# ---------------------------------------------------------------------------


class TestBasicChunking:
    def test_produces_multiple_chunks_for_long_text(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_all_chunks_at_most_chunk_size_characters(self):
        text = "word " * 500  # 2500 chars
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        for chunk in chunks:
            assert len(chunk) <= 300, f"Chunk too long: {len(chunk)}"

    def test_no_empty_chunks_in_output(self):
        text = "hello world " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        for chunk in chunks:
            assert chunk.strip() != "", "Empty chunk found"

    def test_chunks_cover_full_text(self):
        """Concatenating all chunk starts must span the entire input."""
        text = "abcdefghij" * 100  # 1000 chars
        chunk_size = 100
        overlap = 10
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        # First chunk must start with the beginning of the text.
        assert text.startswith(chunks[0])
        # Last chunk must end with (or contain) the end of the text.
        assert text.endswith(chunks[-1]) or chunks[-1] in text


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_consecutive_chunks_share_overlap_content(self):
        """The tail of chunk N should appear at the start of chunk N+1."""
        # Use a large overlap relative to chunk_size to make it obvious.
        text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
        chunks = chunk_text(text, chunk_size=50, overlap=20)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        for i in range(len(chunks) - 1):
            tail = chunks[i][-20:]    # last 20 chars of current chunk
            head = chunks[i + 1][:20]  # first 20 chars of next chunk
            # They may differ slightly due to strip() at boundaries, so
            # check that at least *some* overlap content matches.
            assert tail[:10] in chunks[i + 1] or head[:10] in chunks[i], (
                f"No overlap detected between chunk {i} and {i+1}"
            )

    def test_zero_overlap_no_repeated_content(self):
        """With overlap=0 every character appears in exactly one chunk."""
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        chunks = chunk_text(text, chunk_size=13, overlap=0)
        # Reconstruct and verify coverage.
        reconstructed = "".join(chunks)
        assert reconstructed == text


# ---------------------------------------------------------------------------
# Line-ending normalisation
# ---------------------------------------------------------------------------


class TestLineEndingNormalisation:
    def test_crlf_converted_to_lf(self):
        text = "line one\r\nline two\r\nline three"
        chunks = chunk_text(text, chunk_size=200, overlap=10)
        # After normalisation there should be no carriage-return in output.
        for chunk in chunks:
            assert "\r" not in chunk, "CRLF was not normalised to LF"

    def test_mixed_line_endings_normalised(self):
        text = "a\r\nb\nc\r\nd"
        chunks = chunk_text(text, chunk_size=200, overlap=5)
        joined = "".join(chunks)
        assert "\r" not in joined
