#main.py
import os
from dotenv import load_dotenv
from backend.appwrite_client import AppwriteClient
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStore
from backend.gemini_handler import GeminiHandler
from typing import Dict, List

load_dotenv()

class DocumentQA:
    def __init__(self):
        """Initialize the Document QA system"""
        self.appwrite_client = AppwriteClient()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.gemini_handler = GeminiHandler(self.vector_store)
        
        # Initialize the system by loading documents from Appwrite
        self.initialize()
        
    def initialize(self):
        """Initialize the system by loading documents from Appwrite"""
        print("Initializing system...")
        
        # List documents from Appwrite
        result = self.appwrite_client.list_documents()
        files = result.get("files", [])
        
        # Process each document if not already in vector store
        for file in files:
            # Download and process the document
            file_path = self.appwrite_client.download_document(file["$id"])
            if file_path:
                try:
                    # Process the document
                    chunks = self.document_processor.process_document(file_path)
                    
                    # Add document chunks to vector store
                    metadatas = [{"source": file["name"], "file_id": file["$id"]} for _ in chunks]
                    self.vector_store.add_documents(chunks, metadatas)
                    
                    print(f"Processed document: {file['name']}")
                except Exception as e:
                    print(f"Error processing document {file['name']}: {e}")
                finally:
                    # Clean up the temporary file
                    os.remove(file_path)
    
    
    def process_uploaded_document(self, file_path: str, file_name: str) -> bool:
        """Process an uploaded document and store it in Appwrite"""
        try:
            # Process the document to extract text chunks
            chunks = self.document_processor.process_document(file_path)
            
            # Upload the document to Appwrite
            file_id = self.appwrite_client.upload_document(file_path, file_name)
            if not file_id:
                return False
                
            # Add document chunks to vector store
            metadatas = [{"source": file_name, "file_id": file_id} for _ in chunks]
            self.vector_store.add_documents(chunks, metadatas)
            
            return True
        except Exception as e:
            print(f"Error processing uploaded document: {e}")
            return False
    
    def ask(self, question: str) -> Dict:
        """Ask a question about the documents"""
        return self.gemini_handler.answer_question(question)
