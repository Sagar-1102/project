# import os
# import tempfile
# from typing import List, Dict, Optional
# import re
# import hashlib

# def get_file_hash(file_path: str) -> str:
#     """Generate a hash of a file to check if it's already processed"""
#     hasher = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         buf = f.read()
#         hasher.update(buf)
#     return hasher.hexdigest()

# def clean_text(text: str) -> str:
#     """Clean extracted text from PDF"""
#     # Remove excessive whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Remove page numbers
#     text = re.sub(r'\b\d+\b\s+of\s+\b\d+\b', '', text)
#     return text.strip()

# def create_temp_copy(file_path: str) -> str:
#     """Create a temporary copy of a file"""
#     suffix = os.path.splitext(file_path)[1]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
#         with open(file_path, 'rb') as f:
#             tmp_file.write(f.read())
#         return tmp_file.name

# def chunk_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#     """Chunk text with overlap"""
#     if not text:
#         return []
        
#     chunks = []
#     start = 0
#     text_length = len(text)
    
#     while start < text_length:
#         end = min(start + chunk_size, text_length)
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
    
#     return chunks
import re

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
        
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters and normalize whitespace
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into chunks with overlap"""
    if not text:
        return []
        
    text = text.strip()
    chunks = []
    
    if len(text) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        # Try to find a natural break point (period, newline, etc.)
        if end < len(text):
            # Look for a natural break within the last 20% of the chunk
            look_back = int(chunk_size * 0.2)
            natural_break = max(
                text.rfind('. ', end - look_back, end),
                text.rfind('\n', end - look_back, end),
                text.rfind('\t', end - look_back, end),
                text.rfind('? ', end - look_back, end),
                text.rfind('! ', end - look_back, end)
            )
            
            if natural_break != -1:
                end = natural_break + 1  # Include the break character
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks

