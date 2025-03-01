#document_processor.py
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict, Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from utils.helpers import clean_text, chunk_text_with_overlap

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize the document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    def process_document(self, document_path: str) -> List[str]:
        """Process a document and return chunks of text"""
        # Extract text based on file type
        if document_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(document_path)
        elif document_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = self.extract_text_from_image(document_path)
        elif document_path.lower().endswith('.txt'):
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {document_path}")
        
        # Clean the text
        text = clean_text(text)
        
        # Split text into chunks
        if not text:
            return []
            
        return self.text_splitter.split_text(text)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file, including scanned PDFs using OCR"""
        text = ""
        try:
            # First try to extract text directly
            pdf = PdfReader(pdf_path)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text.strip():  # If text was extracted successfully
                    text += page_text
            
            # If no text was extracted, try OCR
            if not text.strip():
                print("No text extracted directly from PDF, trying OCR...")
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    # Perform OCR on each image
                    page_text = pytesseract.image_to_string(image)
                    text += page_text
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR"""
        try:
            # Perform OCR on the image
            text = pytesseract.image_to_string(image_path)
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def get_document_metadata(self, document_path: str) -> Dict:
        """Get metadata for a document"""
        file_name = os.path.basename(document_path)
        return {
            "file_name": file_name,
            "file_type": os.path.splitext(file_name)[1][1:].lower(),
            "file_size": os.path.getsize(document_path)
        }
