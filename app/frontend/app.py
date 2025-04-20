#app.py
import streamlit as st
import os
import tempfile
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import DocumentQA

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide"
)


if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.session_id}")

# Initialize the DocumentQA class (only once per session)
if 'document_qa' not in st.session_state:
    logger.info("Initializing DocumentQA instance")
    st.session_state.document_qa = DocumentQA()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized new chat history")
else:
    logger.info(f"Using existing chat history with {len(st.session_state.messages)} messages")

# Add a button to clear conversation
with st.sidebar:
    st.title("📑 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF, image, or text file", 
                                     type=["pdf", "png", "jpg", "jpeg", "txt"])
    
    if uploaded_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        # Process the document
        with st.spinner("Processing document..."):
            if st.session_state.document_qa.process_uploaded_document(temp_path, uploaded_file.name):
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
            else:
                st.error(f"Failed to process document '{uploaded_file.name}'")
        
        # Clean up the temporary file
        os.remove(temp_path)
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        # Also clear the memory in the gemini_handler
        if hasattr(st.session_state.document_qa, 'gemini_handler'):
            st.session_state.document_qa.gemini_handler.memory.clear()
            logger.info("Conversation and memory cleared")
        st.rerun()

# Main content
st.title("📚 THE COLLEGE HUB")
st.markdown("""
Ask questions about your uploaded documents. The system will search for relevant information 
and provide answers based on the document content.
""")

# Display debug info in an expander
with st.expander("Debug Information", expanded=False):
    if hasattr(st.session_state.document_qa, 'gemini_handler'):
        chat_history = st.session_state.document_qa.gemini_handler.memory.load_memory_variables({}).get("chat_history", [])
        st.write(f"Current memory contains {len(chat_history)} messages")
        if chat_history:
            st.write("Last 3 messages in memory:")
            for i, msg in enumerate(chat_history[-3:]):
                st.write(f"{i+1}. {type(msg).__name__}: {msg.content[:100]}...")
    st.write(f"Session ID: {st.session_state.session_id}")
    st.write(f"UI message history: {len(st.session_state.messages)} messages")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            logger.info(f"Processing question: {prompt}")
            
            # Get the response from DocumentQA
            response = st.session_state.document_qa.ask(prompt)
            answer = response["answer"]
            sources = response.get("sources", [])
            from_kb = response.get("from_kb", False)
            
            # Display answer
            st.markdown(answer)
            
            # Display sources if available
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {source}")
            elif not from_kb:
                st.info("No specific information found in your documents. This answer is based on general knowledge.")
            
            logger.info(f"Response generated. Sources: {len(sources)}, From KB: {from_kb}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })

# Add a download button for the conversation history
if st.session_state.messages:
    import json
    from datetime import datetime
    
    def export_chat():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_export = {
            "timestamp": timestamp,
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages
        }
        return json.dumps(chat_export, indent=2)
    
    st.download_button(
        label="Download Conversation",
        data=export_chat(),
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"

    )
