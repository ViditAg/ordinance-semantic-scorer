# function to split text into overlapping chunks by characters

from typing import List

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
