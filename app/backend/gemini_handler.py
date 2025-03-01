import os
import pickle
import logging
from typing import List, Dict, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiHandler:
    def __init__(self, vector_store=None, memory_file="conversation_memory.pkl"):
        """Initialize the Gemini handler with LLM and memory"""
        logger.info("Initializing Gemini handler")
        
        self.memory_file = memory_file
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash",
            temperature=0.7,
            top_k=40,
            top_p=0.8,
            max_output_tokens=2048,
        )
        
        # Initialize or load existing memory
        self.memory = self.load_memory() or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up the QA chain if we have a vector store
        self.vector_store = vector_store
        self.qa_chain = None
        
        if vector_store and vector_store.db:
            logger.info("Vector store found, setting up QA chain")
            try:
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=vector_store.db.as_retriever(
                        search_kwargs={"k": 4}
                    ),
                    memory=self.memory,
                    return_source_documents=True
                )
                logger.info("QA chain created successfully")
            except Exception as e:
                logger.error(f"Error creating QA chain: {e}")
                self.qa_chain = None
        else:
            logger.info("No vector store provided or empty vector store")
    
    def load_memory(self):
        """Load memory from disk if available"""
        try:
            if os.path.exists(self.memory_file):
                logger.info(f"Loading memory from {self.memory_file}")
                with open(self.memory_file, 'rb') as f:
                    memory = pickle.load(f)
                    logger.info(f"Loaded memory with {len(memory.chat_memory.messages)} messages")
                    return memory
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
        return None
    
    def save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
                logger.info(f"Saved memory with {len(self.memory.chat_memory.messages)} messages")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def get_relevant_history(self, max_messages=10):
        """Get the most recent N messages to stay within token limits"""
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        # Filter out any messages with empty content
        valid_history = [msg for msg in chat_history if hasattr(msg, 'content') and 
                         msg.content and isinstance(msg.content, str) and msg.content.strip()]
        return valid_history[-max_messages:] if valid_history else []
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using the QA chain or direct LLM"""
        # Validate input
        if not question or not question.strip():
            return {
                "answer": "I received an empty question. Please provide some text.",
                "sources": [],
                "from_kb": False
            }
            
        # Log memory state
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        history_length = len(chat_history)
        logger.info(f"Memory contains {history_length} messages")
        
        # Check if vector store is properly initialized
        has_vector_store = (self.qa_chain is not None and 
                        self.vector_store is not None and 
                        self.vector_store.db is not None)
                        
        logger.info(f"Vector store available: {has_vector_store}")
        
        if has_vector_store:
            try:
                # Try retrieving relevant documents first
                logger.info(f"Searching for relevant documents for: {question}")
                result = self.qa_chain({"question": question})
                source_docs = [doc.page_content for doc in result.get("source_documents", [])]
                
                if source_docs:
                    logger.info(f"Found {len(source_docs)} relevant document chunks")
                    
                    # Explicitly update memory
                    self.memory.save_context({"input": question}, {"output": result["answer"]})
                    self.save_memory()
                    
                    return {
                        "answer": result["answer"],
                        "sources": source_docs,
                        "from_kb": True
                    }
                else:
                    logger.info("No relevant documents found, falling back to direct LLM")
            except Exception as e:
                logger.error(f"Error using QA chain: {e}")
        
        # Fallback to direct LLM
        logger.info("Using direct LLM for response")
        try:
            # Filter out any invalid messages from history
            valid_messages = []
            for msg in self.get_relevant_history():
                if hasattr(msg, 'content') and msg.content and isinstance(msg.content, str) and msg.content.strip():
                    valid_messages.append(msg)
            
            # Log the filtered conversation context
            logger.info(f"Using {len(valid_messages)} valid messages as context")
            
            # Add the current question
            valid_messages.append(HumanMessage(content=question))
            
            # Ensure we have at least the current question
            if len(valid_messages) > 0:
                # Get response from LLM
                response = self.llm.invoke(valid_messages)
                
                logger.info(f"Direct LLM response received")
                
                # Explicitly update memory
                self.memory.save_context({"input": question}, {"output": response.content})
                self.save_memory()
                
                return {
                    "answer": response.content,
                    "sources": [],
                    "from_kb": False
                }
            else:
                raise ValueError("No valid messages to send to the model")
        except Exception as e:
            logger.error(f"Error using direct LLM: {e}")
            return {
                "answer": f"I'm sorry, I encountered an error: {str(e)}",
                "sources": [],
                "from_kb": False
            }
