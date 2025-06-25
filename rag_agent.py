import requests
import json
import os
from typing import List, Dict, Any

# Assuming vector_store_manager.py is in the same directory and already ingested data
from vector_store_manager import ChromaDBManager, CHROMA_DB_PATH, COLLECTION_NAME

# --- Configuration for Ollama ---
OLLAMA_API_BASE_URL = "http://localhost:11434" # Default Ollama API address
OLLAMA_MODEL_NAME = "llama3" # The Llama model you pulled using 'ollama pull llama3'
N_RETRIEVED_CHUNKS = 5 # Number of top relevant chunks to retrieve from ChromaDB

# --- Ollama Llama Model Interaction ---
def generate_llama_response(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama Llama model and returns the generated response.
    """
    url = f"{OLLAMA_API_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False # We want the full response at once
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=300) # 5 min timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()
        return result["response"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server at {OLLAMA_API_BASE_URL}.")
        print("Please ensure Ollama is running and the model '{OLLAMA_MODEL_NAME}' is downloaded (ollama pull {OLLAMA_MODEL_NAME}).")
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
    """
    Performs a RAG query: retrieves relevant docs and generates a response using Llama.
    """
    print(f"\nUser Query: '{user_query}'")

    # 1. Retrieve relevant chunks from ChromaDB
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

    # 2. Construct the prompt for Llama
    # System instruction for Llama (crucial for RAG)
    system_instruction = (
        "You are an expert AI assistant providing concise, fact-based guidance on hydrogen rules of thumb. "
        "Answer the user's question ONLY based on the provided CONTEXT. "
        "If the answer is not in the CONTEXT, state that you don't have enough information from the provided context. "
        "Prioritize providing yes/no decision guidance or initial high-level assessment where applicable, as per your goal. "
        "Cite the filename and page number from the source for each piece of information you provide."
    )

    context_string = "\n\n".join(context_chunks)

    # Combine instruction, context, and user query
    # Llama 3 often responds well to explicit roles like <|begin_of_text|> and <|end_of_text|>
    # and <|start_header_id|>...<|end_header_id|> for instructions/roles
    # For simplicity with the /api/generate endpoint, a clear instruction prefix usually works.
    
    # Using a common prompt template structure that Llama models understand
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instruction}\n"
        f"CONTEXT:\n{context_string}\n\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_query}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    # 3. Generate response using Llama
    print("Generating response with Llama model...")
    llama_response = generate_llama_response(full_prompt)
    
    return llama_response

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Initializing RAG Agent...")
    
    # Initialize ChromaDB Manager
    # This assumes your ChromaDB already contains the hydrogen rules of thumb embeddings
    try:
        chroma_manager = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME)
        # Verify if there are documents in the collection
        if chroma_manager.collection.count() == 0:
            print("WARNING: ChromaDB collection is empty. Please run vector_store_manager.py first to ingest data.")
            exit()
        print(f"ChromaDB has {chroma_manager.collection.count()} documents loaded.")
    except Exception as e:
        print(f"Failed to initialize ChromaDB Manager: {e}")
        print("Please ensure vector_store_manager.py ran successfully and data is persisted.")
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