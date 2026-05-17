"""
Low-level document utilities for the ordinance semantic scorer.

**Responsibilities**

1. **PDF ingestion** — ``extract_text_from_pdf`` walks every page with
   ``pdfplumber``, concatenates non-empty ``extract_text()`` results, and joins
   pages with a blank line so paragraph boundaries stay visible downstream.

2. **Chunking** — ``chunk_text`` splits long plain text into overlapping windows
   of *characters* (not tokens). Embedding models have context limits; chunking
   keeps each segment embeddable while overlap reduces the chance that a
   critical sentence sits on a chunk boundary and gets diluted.

3. **Scoring geometry** — ``cosine_similarities_array`` and ``similarity_to_score``
   map embeddings to cosine similarities and then to a 0–100 stakeholder scale.
   They are shared by :class:`~app.scorer.OrdinanceScorer` so vector math stays
   in one testable place.

These helpers are deliberately **framework-agnostic** (no Streamlit imports)
so unit tests can exercise them in isolation.
"""

from __future__ import annotations

import pdfplumber
from typing import BinaryIO, List, Sequence, Union

import numpy as np

from app.chunking_presets import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

Vector = Sequence[float]
Matrix = Union[Sequence[Sequence[float]], np.ndarray]


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extract all extractable text from a PDF provided as a binary file-like object.

    **Behavior**

    * Opens the PDF with ``pdfplumber.open`` (supports in-memory uploads from
      Streamlit, which wrap bytes in a ``BytesIO``).
    * For each page, calls ``page.extract_text()``. ``None`` and empty strings
      are skipped (scanned pages or image-only PDFs often yield no text per page).
    * Joins successful page texts with ``"\\n\\n"`` so multi-page documents read
      naturally and chunk boundaries tend to align with page breaks.

    **Args:**
        file_obj: Any readable binary stream positioned at the start of a PDF
            (e.g. ``open(path, "rb")`` or ``io.BytesIO(uploaded_bytes)``).

    **Returns:**
        A single string containing the full document text. May be empty if no
        page returned text.

    **Raises:**
        Exceptions from ``pdfplumber`` / ``pdfminer`` if the file is corrupt or
        not a valid PDF (e.g. wrong magic bytes).

    **Note:**
        Extraction quality depends on how the PDF stores text (vector text vs.
        OCR). This project does not run OCR; image-only PDFs may yield ``""``.
    """
    # Collect per-page strings; skip falsy values so we do not insert extra separators.
    text_parts: List[str] = []

    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)

    # Double newline between pages preserves a weak "paragraph" signal for chunkers.
    return "\n\n".join(text_parts)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping character windows suitable for embedding.

    **Why character-based chunking?**

    Tokenizers differ per model; counting characters is simple, predictable, and
    matches how operators think about "about N characters per chunk" in the UI
    (defaults follow :mod:`app.chunking_presets`).
    Overlap means the tail of chunk *N* reappears at the start of chunk *N+1*, so
    a clause split across a boundary still appears in full in at least one chunk.

    **Algorithm (sliding window)**

    * Normalize Windows line endings to ``\\n`` so lengths and boundaries are stable.
    * Start at index 0; take ``text[start : start + chunk_size]``, strip leading/trailing
      whitespace on that slice, and append non-empty results.
    * Advance ``start`` by ``chunk_size - overlap`` until the end of the string.

    **Args:**
        text: Full document body (ordinance plain text).
        chunk_size: Maximum characters per chunk (strict upper bound before strip).
        overlap: Characters shared between consecutive chunks; must be ``< chunk_size``.

    **Returns:**
        List of non-empty chunk strings, in document order.

    **Raises:**
        ValueError: If ``chunk_size <= 0`` or ``overlap >= chunk_size`` (invalid window).

    **Edge cases:**
        * Empty or whitespace-only input → ``[]``.
        * Text shorter than ``chunk_size`` → a single chunk (after strip).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # CRLF → LF avoids counting "\r" toward chunk_size and keeps excerpts readable.
    text = text.replace("\r\n", "\n")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Slide the window left by `overlap` chars so consecutive chunks share context.
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def cosine_similarities_array(vec: Vector, matrix: Matrix) -> np.ndarray:
    """
    Cosine similarity between one vector and each row of a 2-D matrix.

    Zero-norm rows (or a zero ``vec``) would divide by zero; denominators are
    clamped to a small epsilon so results stay finite.
    """
    v = np.asarray(vec, dtype=float)
    M = np.asarray(matrix, dtype=float)
    v_norm = float(np.linalg.norm(v))
    M_norm = np.linalg.norm(M, axis=1)
    denom = v_norm * M_norm
    denom = np.asarray(denom, dtype=float)
    denom[denom == 0] = 1e-10
    sims = (M @ v) / denom
    return sims


def similarity_to_score(sim: float) -> float:
    """
    Map cosine similarity in ``[-1, 1]`` to a 0–100 score.

    Negative similarity is clipped to 0 before scaling. Output is rounded to
    two decimal places for stable tables and JSON.
    """
    scaled = max(sim, 0.0) * 100.0
    return float(round(scaled, 2))
