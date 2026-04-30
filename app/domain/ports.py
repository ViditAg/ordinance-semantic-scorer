"""Port protocols — adapters (PDF, Streamlit, CLI) plug in here."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class TextSource(Protocol):
    """Provides full document plain text (e.g. from PDF bytes or a string)."""

    def load_plain_text(self) -> str:
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits plain text into segments for embedding."""

    def chunk(self, text: str) -> List[str]:
        ...
