# appwrite_client.py
import os
from appwrite.client import Client
from appwrite.services.storage import Storage
import tempfile
import mimetypes
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class AppwriteClient:
    def __init__(self):
        """Initialize the Appwrite client"""
        # Create a client
        self.client = Client()
        
        # Set endpoint and project ID
        self.client.set_endpoint(os.getenv('APPWRITE_ENDPOINT', 'https://cloud.appwrite.io/v1'))
        self.client.set_project(os.getenv('APPWRITE_PROJECT_ID'))
        self.client.set_key(os.getenv('APPWRITE_API_KEY'))
        
        # Create Storage service
        self.storage = Storage(self.client)
        
        # Store bucket ID
        self.bucket_id = os.getenv('APPWRITE_BUCKET_ID')
        
    def list_documents(self):
        """List all documents in the bucket"""
        try:
            return self.storage.list_files(self.bucket_id)
        except Exception as e:
            print(f"Error listing documents: {e}")
            return {"files": []}
    
    def download_document(self, file_id):
        """Download a document from Appwrite storage"""
        try:
            # Create a temporary file to store the downloaded document
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            
            # Get file download as bytes
            result = self.storage.get_file_download(self.bucket_id, file_id)
            
            # Write the bytes to the file
            with open(temp_path, 'wb') as f:
                f.write(result)
                
            return temp_path
        except Exception as e:
            print(f"Error downloading document: {e}")
            return None
    
    
    def upload_document(self, file_path: str, file_name: str) -> Optional[str]:
        """Upload a document to Appwrite storage"""
        try:
            # Get the file MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Upload the file
            result = self.storage.create_file(
                bucket_id=self.bucket_id,
                file_id='unique()',
                file=open(file_path, 'rb'),
                permissions=['role:all'],
                file_name=file_name
            )
            
            return result["$id"]
        except Exception as e:
            print(f"Error uploading document: {e}")
            return None
