import streamlit as st
import os
import sys
import time # For retry logic

# Add the parent directory of the current script to the Python path
# This allows importing pdf_processor and vector_store_manager as modules
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming pdf_processor.py and vector_store_manager.py are in the same directory as app.py
sys.path.insert(0, current_dir)

# Import necessary components from your existing scripts
from vector_store_manager import ChromaDBManager, CHROMA_DB_PATH, COLLECTION_NAME, \
                                 generate_embeddings, CHROMA_BATCH_SIZE, EMBEDDING_MODEL_NAME, embedding_model
from rag_agent import rag_agent_query, OLLAMA_MODEL_NAME, N_RETRIEVED_CHUNKS


# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="🔋 Hydrogen Rules of Thumb RAG Chatbot", # Changed title to reflect hydrogen and RAG
    page_icon="⚛️", # Changed icon to hydrogen atom
    layout="centered"
)

# --- Custom Styling (Copied from your previous code) ---
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: #FAFAFA;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    .stChatInputContainer > div {
        background-color: #1e1e1e !important;
        border-radius: 10px;
    }
    [data-testid="stChatInput"] textarea {
        color: #fafafa !important;
        background-color: #1e1e1e !important;
        border: none;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #1e1e1e;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize ChromaDB and Embedding Model using Streamlit's caching ---
@st.cache_resource
def initialize_rag_components():
    """Initializes the ChromaDBManager and verifies the collection."""
    print("Attempting to initialize RAG components...")
    
    chroma_manager = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME, create_new=False) 
    
    retries = 5
    count = 0
    for i in range(retries):
        try:
            count = chroma_manager.collection.count()
            if count > 0:
                print(f"ChromaDB has {count} documents loaded from {CHROMA_DB_PATH}.")
                break
        except Exception as e:
            st.error(f"Error checking collection count: {e}")
        print(f"Retrying collection count (attempt {i+1}/{retries})...")
        time.sleep(2)

    if count == 0:
        st.warning(f"""
        **Warning:** ChromaDB collection `{COLLECTION_NAME}` is empty or could not be loaded.
        Please ensure `vector_store_manager.py` ran successfully to ingest data.
        Expected path: `{CHROMA_DB_PATH}`
        """)
        return None 
    
    return chroma_manager

# Attempt to initialize components
chroma_manager_instance = initialize_rag_components()

# --- Main App Title and Description ---
st.markdown("<h1 style='text-align:center; color:#FAFAFA;'>⚡ Hydrogen 'Rules of Thumb' RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#AAAAAA;'>Ask about hydrogen rules and guidelines, and I'll use my knowledge base (derived from your PDF documents) to provide concise answers, powered by Llama3.</p>", unsafe_allow_html=True)


# --- Chat Interface ---
if chroma_manager_instance:
    st.success(f"RAG system ready! ChromaDB has {chroma_manager_instance.collection.count()} documents loaded.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    # Using your avatar logic
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("Ask your hydrogen-related question...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container with user avatar
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"): # Using assistant avatar
            with st.spinner("Thinking..."):
                response = rag_agent_query(user_input, chroma_manager_instance)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("RAG system could not be initialized. Please check the console for errors and ensure data ingestion.")

# --- System Information Sidebar ---
st.sidebar.header("System Information")
st.sidebar.markdown(f"**Ollama Model:** `{OLLAMA_MODEL_NAME}`")
st.sidebar.markdown(f"**ChromaDB Path:** `{CHROMA_DB_PATH}`")
st.sidebar.markdown(f"**ChromaDB Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**Retrieved Chunks per Query:** `{N_RETRIEVED_CHUNKS}`")
st.sidebar.markdown(f"**ChromaDB Batch Size (Ingestion):** `{CHROMA_BATCH_SIZE}`")
st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**Embedding Device:** `{embedding_model.device}`")


# --- Footer (Copied from your previous code) ---
st.markdown("""
<div class="footer">
    ⚛️ Powered by Llama3 & ChromaDB. Made by IntuiNext Inc.
</div>
""", unsafe_allow_html=True)
