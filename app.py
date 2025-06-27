import streamlit as st
import os
import sys
import time  # For retry logic

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# --- Module Imports ---
from vector_store_manager import ChromaDBManager, CHROMA_DB_PATH, COLLECTION_NAME, \
                                 generate_embeddings, CHROMA_BATCH_SIZE, EMBEDDING_MODEL_NAME, embedding_model
from rag_agent import rag_agent_query, OLLAMA_MODEL_NAME, N_RETRIEVED_CHUNKS

# --- Streamlit App Config ---
st.set_page_config(
    page_title="🔋 Hydrogen Rules of Thumb",
    page_icon="⚛️",
    layout="centered"
)

# --- Custom Dark Theme Styling ---
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], main, section {
    background-color: #FFFFFF !important;  /* White background */
    color: #000000 !important;             /* Black text */
}

/* Chat messages */
.stChatMessage {
    color: #000000 !important;
}

/* Chat input outer container */
[data-testid="stBottomContainer"] {
    background-color: #FFFFFF !important;
    border-top: 1px solid #CCC !important;
}

/* Chat input inner box */
.stChatInputContainer > div {
    background-color: #F5F5F5 !important;  /* Light gray input box */
    border-radius: 10px !important;
}

/* Chat input textarea */
[data-testid="stChatInput"] textarea {
    color: #000000 !important;
    background-color: #F5F5F5 !important;
    border: 1px solid #CCC !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #F5F5F5;
    color: #333;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)




# --- Initialize ChromaDB ---
@st.cache_resource
def initialize_rag_components():
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

# --- Try Initialization ---
chroma_manager_instance = initialize_rag_components()

# --- Title and Description ---
st.markdown("<h1 style='text-align:center; color:#FAFAFA;'>⚡ Hydrogen 'Rules of Thumb'</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#AAAAAA;'>Ask about hydrogen energy rules.</p>", unsafe_allow_html=True)

# --- Chatbot Interface ---
if chroma_manager_instance:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask your hydrogen-related question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_agent_query(user_input, chroma_manager_instance)
                except Exception as e:
                    response = f"⚠️ Error: {e}"
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("RAG system could not be initialized. Please check the console for errors and ensure data ingestion.")

# --- Optional Sidebar (uncomment if needed) ---
# st.sidebar.header("System Information")
# st.sidebar.markdown(f"**Ollama Model:** `{OLLAMA_MODEL_NAME}`")
# st.sidebar.markdown(f"**ChromaDB Path:** `{CHROMA_DB_PATH}`")
# st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
# st.sidebar.markdown(f"**Chunks per Query:** `{N_RETRIEVED_CHUNKS}`")
# st.sidebar.markdown(f"**Batch Size:** `{CHROMA_BATCH_SIZE}`")
# st.sidebar.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
# st.sidebar.markdown(f"**Embedding Device:** `{embedding_model.device}`")

# --- Footer ---
st.markdown("""
<div class="footer">
    ⚛️ Made by IntuiNext Inc.
</div>
""", unsafe_allow_html=True)
