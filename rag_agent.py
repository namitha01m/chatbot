import requests
import json
import os
from typing import List, Dict, Any
import time


from vector_store_manager import FAISSManager, FAISS_INDEX_PATH, FAISS_METADATA_PATH, generate_embeddings
# checking for evaltion
#from evaluation import evaluate_groundedness, evaluate_relevance

OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3"
N_RETRIEVED_CHUNKS = 5

#  Llama 
def generate_llama_response(prompt: str) -> str:
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


# rag
def rag_agent_query(user_query: str, faiss_manager: FAISSManager) -> str: 
    print(f"\nUser Query: '{user_query}'")

    
    simple_greetings = ["hi", "hello", "hey", "hola", "good morning", "good afternoon", "good evening"]
    if user_query.lower().strip() in simple_greetings:
        return "Hello! I'm an expert AI assistant providing concise, fact-based guidance on hydrogen rules of thumb. What would you like to know?"

    print(f"Retrieving top {N_RETRIEVED_CHUNKS} relevant chunks from FAISS...")
    retrieved_results = faiss_manager.query_collection(user_query, n_results=N_RETRIEVED_CHUNKS)
    
    context_chunks = []
    unique_sources = set() 

    if retrieved_results and retrieved_results['documents'] and retrieved_results['documents'][0]:
        for i in range(len(retrieved_results['documents'][0])):
            doc_content = retrieved_results['documents'][0][i]
            metadata = retrieved_results['metadatas'][0][i]
            
            source_info = f"{metadata.get('filename', 'N/A')} "
            
            context_chunks.append(f"Source: {source_info}\nContent: {doc_content}")
            unique_sources.add(source_info) # Collect unique sources here

        print(f"Retrieved {len(context_chunks)} relevant chunks.")
    else:
        print("No relevant context found in FAISS index.")
        return "I couldn't find enough relevant information in my knowledge base to answer that question. Please try rephrasing or ask about a different topic."
    
    # for evaluation
    #context_string = "\n\n".join(context_chunks)

    #citation
    sources_for_llm_citation = ""
    if unique_sources:
        #use after response
        sources_for_llm_citation = "\n\nAvailable Sources for Citation:\n" + "\n".join([f"- {s}" for s in sorted(list(unique_sources))])

    system_instruction = (
        "You are an expert AI assistant providing concise, fact-based guidance on hydrogen rules of thumb. "
        "Answer the user's question ONLY based on the provided CONTEXT. "
        "If the answer is not in the CONTEXT or the question is outside the scope of hydrogen energy, "
        "respond with the exact phrase: 'I am not able to answer questions that are not related to hydrogen energy.' "
        "Prioritize providing yes/no decision guidance or initial high-level assessment where applicable, as per your goal. "
        "For lists or multiple points, present the information as a **numbered list** with each item starting on a new line. "
        "**Bold** key terms or method names for better readability. "
        "After your complete answer, provide all source citations in a separate 'Sources:' section at the very end, "
        "listing each unique filename and page number for PDFs. "
        "For example: (Source: filename)."
    )

    context_string = "\n\n".join(context_chunks)

    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instruction}\n"
        f"CONTEXT:\n{context_string}\n\n"
        
        f"{sources_for_llm_citation}\n\n" 
        f"<|start_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_query}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("Generating response with Llama model...")
    llama_response = generate_llama_response(full_prompt)
    '''
    print("\n--- Evaluating Response Quality ---")
    groundedness_score, groundedness_reason = evaluate_groundedness(llama_response, context_string)
    print(f"Groundedness Score: {groundedness_score:.2f}, Reason: '{groundedness_reason}'")

    relevance_score, relevance_reason = evaluate_relevance(user_query, llama_response)
    print(f"Relevance Score: {relevance_score:.2f}, Reason: '{relevance_reason}'")
    print("-----------------------------------\n")
    '''
    return llama_response


if __name__ == "__main__":
    print("Initializing RAG Agent...")
    from vector_store_manager import FAISS_INDEX_PATH, FAISS_METADATA_PATH
    print(f"FAISS Index path: {FAISS_INDEX_PATH}")
    print(f"FAISS Metadata path: {FAISS_METADATA_PATH}")
    
    try:
        faiss_manager = FAISSManager(FAISS_INDEX_PATH, FAISS_METADATA_PATH) 
        
        if faiss_manager.count() == 0:
            print("WARNING: FAISS index is empty. Please ensure vector_store_manager.py ran successfully to ingest data.")
            print(f"Checked FAISS files at: {FAISS_INDEX_PATH}, {FAISS_METADATA_PATH}")
            exit()
        print(f"FAISS index has {faiss_manager.count()} documents loaded.")
    except Exception as e:
        print(f"Failed to initialize FAISS Manager: {e}")
        print(f"Attempted to access FAISS files at: {FAISS_INDEX_PATH}, {FAISS_METADATA_PATH}")
        print("Please ensure vector_store_manager.py ran successfully to ingest data.")
        import traceback
        traceback.print_exc()
        exit()

    print(f"\n--- Hydrogen 'Rules of Thumb' Agent ---")
    print(f"Powered by Llama3 ({OLLAMA_MODEL_NAME}) and FAISS.")
    print("Type your query and press Enter. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Query: ")
        if user_input.lower() == 'exit':
            print("Exiting agent. Goodbye!")
            break
        
        response = rag_agent_query(user_input, faiss_manager)
        print(f"\nAgent Response:\n{response}")
