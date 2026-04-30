"""Text source backed by a PDF binary stream (e.g. ``BytesIO`` or Streamlit upload)."""

from __future__ import annotations

from typing import BinaryIO

from app.utils import extract_text_from_pdf


class PdfTextSource:
    """Load plain text via ``pdfplumber`` (see :func:`app.utils.extract_text_from_pdf`)."""

    def __init__(self, file_obj: BinaryIO) -> None:
        self._file_obj = file_obj

    def load_plain_text(self) -> str:
        return extract_text_from_pdf(self._file_obj)
