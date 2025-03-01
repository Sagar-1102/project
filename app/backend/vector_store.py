# vector_store.py
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Union
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, persist_directory="faiss_index"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model="embedding-001"
        )
        
        # Create directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            self.db = None
        else:
            # Load existing index if it exists
            try:
                self.db = FAISS.load_local(persist_directory, self.embeddings)
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.db = None
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        if not texts:
            return
            
        if self.db is None:
            self.db = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.db.add_texts(texts, metadatas=metadatas)
        
        # Save the updated index
        self.db.save_local(self.persist_directory)
    
    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Search the vector store for relevant documents"""
        if self.db is None:
            return []
            
        results = self.db.similarity_search_with_score(query, k=k)
        return [{"content": doc.page_content, "score": score, "metadata": doc.metadata} 
                for doc, score in results]
    
    def search_with_threshold(self, query: str, k: int = 4, score_threshold: float = 0.7) -> List[Dict]:
        """Search with a relevance threshold"""
        results = self.search(query, k=k)
        return [r for r in results if r["score"] > score_threshold]
