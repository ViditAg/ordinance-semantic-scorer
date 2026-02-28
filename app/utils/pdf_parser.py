import pdfplumber
from typing import BinaryIO

def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extracts text from an uploaded PDF file-like object using pdfplumber.
    Returns concatenated text from all pages.

    Args:
        file_obj: BinaryIO - The uploaded PDF file-like object.
        
    Returns:
        str - The concatenated text from all pages of the PDF.
        
    Raises:
        pdfplumber.PDFSyntaxError: If the PDF file is not valid.
        pdfplumber.PDFPageError: If the PDF page is not valid.
        pdfplumber.PDFSyntaxError: If the PDF file is not valid.
    """
    # Initialize an empty list to store the text parts
    text_parts = []

    # Open the PDF file using pdfplumber and iterate through each page
    with pdfplumber.open(file_obj) as pdf:
        # Extract text from each page
        for page in pdf.pages:
            # Extract text from the page
            txt = page.extract_text()
            # If text is found, add it to the list
            if txt:
                # Add the text to the list
                text_parts.append(txt)

    # Return the concatenated text from all pages
    return "\n\n".join(text_parts)