import pdfplumber
from typing import BinaryIO

def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extracts text from an uploaded PDF file-like object using pdfplumber.
    Returns concatenated text from all pages.
    """
    text_parts = []
    # pdfplumber accepts file-like objects
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n\n".join(text_parts)