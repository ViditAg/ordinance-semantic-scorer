"""Character-window chunker delegating to :func:`app.utils.chunk_text`."""

from __future__ import annotations

from typing import List

from app.utils import chunk_text


class FixedCharacterChunker:
    """Fixed ``chunk_size`` and ``overlap`` (policy lives in constructor, not UI)."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> List[str]:
        return chunk_text(text, chunk_size=self._chunk_size, overlap=self._overlap)
