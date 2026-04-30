"""Smoke tests for hexagonal adapters and protocol wiring."""

from __future__ import annotations

import io

from app.adapters.chunking_char import FixedCharacterChunker
from app.adapters.text_pdf import PdfTextSource
from app.adapters.text_plain import PlainTextSource
from app.domain.ports import Chunker, TextSource
from app.utils import chunk_text


def test_plain_text_source_is_text_source():
    src: TextSource = PlainTextSource("hello")
    assert src.load_plain_text() == "hello"


def test_fixed_character_chunker_matches_utils():
    text = "abcdefghijklmnop"
    c = FixedCharacterChunker(chunk_size=5, overlap=2)
    assert c.chunk(text) == chunk_text(text, chunk_size=5, overlap=2)


def test_fixed_character_chunker_is_chunker():
    ch: Chunker = FixedCharacterChunker(10, 2)
    assert isinstance(ch, Chunker)


def test_pdf_text_source_is_text_source(monkeypatch):
    def fake_extract(_file_obj):
        return "extracted"

    import app.adapters.text_pdf as mod

    monkeypatch.setattr(mod, "extract_text_from_pdf", fake_extract)
    src: TextSource = PdfTextSource(io.BytesIO(b"%PDF"))
    assert src.load_plain_text() == "extracted"
