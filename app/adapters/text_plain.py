"""Text source backed by an in-memory string (already extracted or pasted)."""

from __future__ import annotations


class PlainTextSource:
    """Return the same string on each ``load_plain_text`` call."""

    def __init__(self, text: str) -> None:
        self._text = text

    def load_plain_text(self) -> str:
        return self._text
