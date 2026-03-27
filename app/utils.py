"""Utility functions for the app."""

# imports
import pdfplumber
from typing import BinaryIO, List

# function to extract text from a PDF file
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


# function to split text into overlapping chunks by characters
def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks by characters.
    This is a naive splitter intended for semantic embeddings.

    Args:
        text: The text to split.
        chunk_size: The size of each chunk.
        overlap: The overlap between chunks.
    
    Returns:
        A list of chunks.
    
    """
    # Check if the chunk size is positive
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    # Check if the overlap is smaller than the chunk size
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    # Replace CRLF (Windows) with LF (Unix) format line endings
    text = text.replace("\r\n", "\n")
    
    # Initialize an empty list to store the chunks
    chunks = []
    
    # Initialize the start index to 0
    start = 0
    
    # Get the length of the text
    text_len = len(text)
    
    # Loop through the text
    while start < text_len:
        
        # Calculate the end index of the chunk
        end = min(start + chunk_size, text_len)
        
        # Get the chunk
        chunk = text[start:end].strip()
        
        # If the chunk is not empty, add it to the list of chunks
        if chunk:
            # Add the chunk to the list of chunks
            chunks.append(chunk)
        
        # If the end index is greater than or equal to the length of the text, break
        if end >= text_len:
            break
        
        # Calculate the start index of the next chunk
        start = end - overlap
        # If the start index is less than 0, set it to 0
        if start < 0:
            start = 0
   
    # Return the list of chunks
    return chunks
