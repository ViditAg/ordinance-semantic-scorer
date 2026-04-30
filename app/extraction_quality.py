"""
Heuristics for judging whether PDF text extraction looks usable.

These checks do **not** replace human review; they flag obvious failures such as
image-only PDFs (empty or near-empty text) or garbage with almost no letters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ExtractionQuality:
    """Lightweight stats derived from extracted plain text."""

    char_count: int
    alpha_ratio: float
    """Share of alphabetic characters in non-stripped text (0–1)."""
    word_like_count: int
    """Count of letter-only tokens of length ≥ 2 (split on whitespace)."""
    likely_image_only_or_corrupt: bool
    """True when extraction is empty or almost no alphabetic content."""
    looks_usable: bool
    """True when length and letter density suggest a normal text PDF."""


def assess_extraction_quality(
    text: str,
    *,
    min_chars: int = 500,
    min_alpha_ratio: float = 0.12,
    min_word_like: int = 40,
) -> ExtractionQuality:
    """
    Score extraction quality for downstream chunking / embedding.

    Args:
        text: Raw string from :func:`app.utils.extract_text_from_pdf` (may be empty).
        min_chars: Below this (after strip) → not usable.
        min_alpha_ratio: If letter density is below this, treat as corrupt/OCR-needed.
        min_word_like: Need at least this many simple word tokens for a real ordinance.
    """
    raw = text or ""
    stripped = raw.strip()
    n = len(stripped)
    if n == 0:
        return ExtractionQuality(
            char_count=0,
            alpha_ratio=0.0,
            word_like_count=0,
            likely_image_only_or_corrupt=True,
            looks_usable=False,
        )

    alpha = sum(1 for c in stripped if c.isalpha())
    ratio = alpha / len(stripped)

    tokens: List[str] = stripped.split()
    word_like = sum(1 for t in tokens if len(t) >= 2 and t.isalpha())

    likely_bad = n < min_chars or ratio < min_alpha_ratio or word_like < min_word_like
    usable = not likely_bad

    return ExtractionQuality(
        char_count=n,
        alpha_ratio=round(ratio, 4),
        word_like_count=word_like,
        likely_image_only_or_corrupt=likely_bad,
        looks_usable=usable,
    )
