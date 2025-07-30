import streamlit as st
import os
import sys
import time 


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


from vector_store_manager import FAISSManager, FAISS_INDEX_PATH, FAISS_METADATA_PATH, \
                                       EMBEDDING_MODEL_NAME, embedding_model, \
                                       CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS # Import chunking params for sidebar
from rag_agent import rag_agent_query, OLLAMA_MODEL_NAME, N_RETRIEVED_CHUNKS


# Streamlit
st.set_page_config(
    page_title="🔋 Hydrogen Rules of Thumb",
    page_icon="⚛️",
    layout="centered"
)


st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], main, section {
    background-color: #FFFFFF !important;  
    color: #000000 !important;             
}


.stChatMessage {
    color: #000000 !important; 
}


[data-testid="stBottomContainer"] {
    background-color: #FFFFFF !important;
    border-top: 1px solid #CCC !important;
}

/* Chat input inner box */
.stChatInputContainer > div {
    background-color: #F5F5F5 !important;  
    border-radius: 10px !important;
}


[data-testid="stChatInput"] textarea {
    color: #000000 !important;            
    background-color: #F5F5F5 !important;
    border: 1px solid #CCC !important;
}


[data-testid="stChatInput"] p {
    color: #555555 !important;
}


[data-testid="stSidebar"] {
    background-color: #FFFFFF !important; 
    color: #000000 !important;            
}


[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, [data-testid="stSidebar"] .stMarkdown {
    color: #000000 !important; 
}



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


@st.cache_resource
def initialize_rag_components():
    """Initializes the FAISSManager and verifies the index."""
    print("Attempting to initialize RAG components (FAISS)...")
    
    
    faiss_manager = FAISSManager(FAISS_INDEX_PATH, FAISS_METADATA_PATH) 
    
    # Checking if index is loaded and has documents
    count = faiss_manager.count()
    if count == 0:
        st.warning(f"""
        **Warning:** FAISS index is empty or could not be loaded.
        Please ensure `faiss_vector_store_manager.py` ran successfully to ingest data.
        Expected files: `{FAISS_INDEX_PATH}` and `{FAISS_METADATA_PATH}`
        """)
        return None 
    else:
        print(f"FAISS index has {count} documents loaded from {FAISS_INDEX_PATH}.")
        #st.success(f"RAG system ready! FAISS index has {count} documents loaded.")
    
    return faiss_manager


faiss_manager_instance = initialize_rag_components()


st.markdown("<h1 style='text-align:center;'>⚡ Hydrogen 'Rules of Thumb'</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask about hydrogen energy rules.</p>", unsafe_allow_html=True)

# Chatbot Interface 
if faiss_manager_instance: 
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
                    response = rag_agent_query(user_input, faiss_manager_instance)
                except Exception as e:
                    response = f"⚠️ Error: {e}"
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("RAG system could not be initialized. Please check the console for errors and ensure data ingestion.")





st.markdown("""
<div class="footer">
    Please do not share any sensitive or personal information
</div>
""", unsafe_allow_html=True)
