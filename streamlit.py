import streamlit as st
import os
from dotenv import load_dotenv
from app import get_rag_chain # Import the function to get the RAG chain
from config import KNOWLEDGE_BASE_DIR # Import KNOWLEDGE_BASE_DIR for path checks

load_dotenv() # Load environment variables from .env

st.set_page_config(page_title="Hydrogen Rules of Thumb Agent", layout="centered")

# --- Initialize RAG Chain (Cached for efficiency) ---
@st.cache_resource # Cache the RAG chain to avoid re-initialization on every rerun
def initialize_rag_chain():
    """Initializes and returns the RAG chain, with a check for knowledge base existence."""
    if not os.path.exists(KNOWLEDGE_BASE_DIR) or not os.listdir(KNOWLEDGE_BASE_DIR):
        st.error(
            f"Knowledge base not found at '{KNOWLEDGE_BASE_DIR}'. "
            "Please run `python ingest.py` first to build the knowledge base."
        )
        st.stop() # Stop the Streamlit app if knowledge base is missing
    return get_rag_chain()

# --- Streamlit UI ---
st.title("💡 Hydrogen Rules of Thumb Agent")
st.markdown(
    """
    Your go-to AI assistant for quick, fact-based guidance on hydrogen-related inquiries.
    Ask me anything about hydrogen rules of thumb, safety, applications, etc.!
    """
)

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial assistant message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your Hydrogen Rules of Thumb Agent. How can I assist you today?"})

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask me about hydrogen..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize RAG chain (this will use the cached version if already run)
    rag_chain = initialize_rag_chain()

    if rag_chain: # Proceed only if RAG chain was successfully initialized
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."): # Show a spinner while processing
                try:
                    # Directly invoke the rag_chain
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred while processing your request: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        # Error message already displayed by initialize_rag_chain if it returned None
        pass

# --- Clear Chat Button ---
if st.button("Clear Chat", type="secondary"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Hydrogen Rules of Thumb Agent. How can I assist you today?"}]
    st.experimental_rerun() # Rerun the app to clear chat display