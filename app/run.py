#run.py
import os
from dotenv import load_dotenv
import streamlit
import sys
from pathlib import Path

# Load environment variables
load_dotenv()

# Check if environment variables are set
required_vars = [
    'APPWRITE_API_KEY', 
    'APPWRITE_PROJECT_ID', 
    'APPWRITE_BUCKET_ID',
    'GOOGLE_API_KEY'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file.")
    sys.exit(1)

# Get the directory of the current script
current_dir = Path(__file__).parent

# Run the Streamlit app
streamlit_cmd = [
    "streamlit", 
    "run", 
    str(current_dir / "frontend" / "app.py"),
    "--server.port=8501",
    "--browser.serverAddress=localhost",
    "--server.headless=false"
]

os.system(" ".join(streamlit_cmd))
