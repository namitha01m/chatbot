import requests
import json
import os
from typing import List, Dict, Any
import time # For retries

# Assuming vector_store_manager.py is in the same directory and already ingested data
# Import the ChromaDBManager from vector_store_manager.py directly
from vector_store_manager import ChromaDBManager, CHROMA_DB_PATH, COLLECTION_NAME

# --- Configuration for Ollama ---
OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3"
N_RETRIEVED_CHUNKS = 5

# --- Ollama Llama Model Interaction ---
def generate_llama_response(prompt: str) -> str:
    # ... (this function remains the same as before) ...
    url = f"{OLLAMA_API_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server at {OLLAMA_API_BASE_URL}.")
        print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL_NAME}' is downloaded (ollama pull {OLLAMA_MODEL_NAME}).")
        return "ERROR: Ollama server not reachable."
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out. The model might be taking too long to respond, or the prompt is too complex.")
        return "ERROR: Ollama request timed out."
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred during Ollama API call: {e}")
        return "ERROR: An unexpected error occurred with Ollama."
    except KeyError:
        print(f"Error: 'response' key not found in Ollama API response. Response: {result}")
        return "ERROR: Invalid response from Ollama."


# --- RAG Agent Core ---
def rag_agent_query(user_query: str, chroma_manager: ChromaDBManager) -> str:
    print(f"\nUser Query: '{user_query}'")

    print(f"Retrieving top {N_RETRIEVED_CHUNKS} relevant chunks from ChromaDB...")
    retrieved_results = chroma_manager.query_collection(user_query, n_results=N_RETRIEVED_CHUNKS)
    
    context_chunks = []
    if retrieved_results and retrieved_results['documents'] and retrieved_results['documents'][0]:
        for i in range(len(retrieved_results['documents'][0])):
            doc_content = retrieved_results['documents'][0][i]
            metadata = retrieved_results['metadatas'][0][i]
            context_chunks.append(
                f"Source: {metadata.get('filename', 'N/A')} (Page: {metadata.get('page_number', 'N/A')})\n"
                f"Content: {doc_content}"
            )
        print(f"Retrieved {len(context_chunks)} relevant chunks.")
    else:
        print("No relevant context found in ChromaDB.")
        return "I couldn't find enough relevant information in my knowledge base to answer that question. Please try rephrasing or ask about a different topic."

  
    system_instruction = (
    "You are a helpful and precise AI assistant specialized in hydrogen energy rules of thumb. "
    "Answer user questions strictly based on the provided CONTEXT. "
    "If the required information is not found in the CONTEXT, clearly state: 'There is not enough information in the provided context.' "
    "Do not provide answers beyond the CONTEXT or outside the scope of hydrogen energy. "
    "Where possible, provide **yes/no decisions** or a **high-level initial assessment**. "
    "For multi-part answers, use a **numbered list**, with each point on a new line. "
    "Use **bold** formatting for key terms, methods, or decisions to improve readability. "
    "End each response with a 'Sources:' section listing references in this format: (Source: filename.pdf, Page: X).")
     


    context_string = "\n\n".join(context_chunks)

    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instruction}\n"
        f"CONTEXT:\n{context_string}\n\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_query}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("Generating response with Llama model...")
    llama_response = generate_llama_response(full_prompt)
    
    return llama_response

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Initializing RAG Agent...")
    print(f"ChromaDB persistence path: {CHROMA_DB_PATH}")
    
    # Initialize ChromaDB Manager for RAG (do NOT create new/delete collection here)
    try:
        # ChromaDBManager(..., create_new=False) is implied default, but explicit for clarity
        chroma_manager = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME, create_new=False) 
        
        # Robust check for collection count
        retries = 5
        count = 0
        for i in range(retries):
            try:
                count = chroma_manager.collection.count()
                if count > 0:
                    break
            except Exception as e:
                print(f"Error checking collection count: {e}")
            print(f"Retrying collection count (attempt {i+1}/{retries})...")
            time.sleep(2) # Wait longer between retries

        if count == 0:
            print("WARNING: ChromaDB collection is empty. Please ensure vector_store_manager.py ran successfully and data is persisted.")
            print(f"Checked ChromaDB at: {CHROMA_DB_PATH}, Collection: {COLLECTION_NAME}")
            exit()
        print(f"ChromaDB has {count} documents loaded.")
    except Exception as e:
        print(f"Failed to initialize ChromaDB Manager: {e}")
        print(f"Attempted to access ChromaDB at: {CHROMA_DB_PATH}, Collection: {COLLECTION_NAME}")
        print("Please ensure vector_store_manager.py ran successfully and data is persisted.")
        import traceback
        traceback.print_exc()
        exit()

    print(f"\n--- Hydrogen 'Rules of Thumb' Agent ---")
    print(f"Powered by Llama3 ({OLLAMA_MODEL_NAME}) and ChromaDB.")
    print("Type your query and press Enter. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Query: ")
        if user_input.lower() == 'exit':
            print("Exiting agent. Goodbye!")
            break
        
        response = rag_agent_query(user_input, chroma_manager)
        print(f"\nAgent Response:\n{response}")