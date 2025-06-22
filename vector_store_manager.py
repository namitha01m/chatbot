import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer # We'll load the model explicitly
import os
from typing import List, Dict, Any

# Assuming your pdf_processor.py is in the same directory and you want to import it
# If you prefer to save chunks to a file and load them, that's also an option.
from pdf_processor import process_pdfs_to_chunks, PDF_DIRECTORY # Import the processing function and directory

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db" # Directory where ChromaDB will store its data
COLLECTION_NAME = "hydrogen_rules_of_thumb" # Name of your ChromaDB collection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Sentence-BERT model for embeddings

# --- Initialize Embedding Model ---
# This loads the model. It will download it the first time.
# Set device to 'cuda' if you have an NVIDIA GPU for faster embedding generation
# otherwise, it defaults to CPU.
try:
    print(f"Loading Sentence-Transformer model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda' if chromadb.get_recommended_device() == 'cuda' else 'cpu')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure your environment is set up correctly (e.g., PyTorch, CUDA if using GPU).")
    # Fallback to CPU if CUDA fails, or raise error if model cannot be loaded at all
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    print("Attempting to load model on CPU as fallback.")


# --- Embedding Function for ChromaDB ---
# ChromaDB can take an embedding function directly.
# However, for using our specific SentenceTransformer model explicitly,
# we'll generate embeddings first and then add them.
# Alternatively, you can let ChromaDB handle it by passing an embedding_function.
# For custom models, generating them manually and passing is more flexible.
'''
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the pre-loaded SentenceTransformer model.
    """
    # normalize_embeddings=True is often recommended for cosine similarity
    embeddings = embedding_model.encode(texts, convert_to_numpy=False, normalize_embeddings=True).tolist()
    return embeddings
'''
# --- Embedding Function for ChromaDB ---
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using the pre-loaded SentenceTransformer model.
    """
    # Setting convert_to_numpy=True makes it return a numpy array
    # which can then be converted to a list of lists using .tolist()
    embeddings_numpy = embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = embeddings_numpy.tolist() # Convert the numpy array to a list of lists
    return embeddings

# --- ChromaDB Manager ---
class ChromaDBManager:
    def __init__(self, db_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        # Use get_or_create_collection to ensure it exists or is created
        # We'll add embeddings manually, so no need to specify embedding_function here
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"ChromaDB client initialized. Collection '{collection_name}' ready at {db_path}")

    def add_chunks_to_collection(self, chunks: List[Dict[str, Any]]):
        """
        Adds processed text chunks and their embeddings to the ChromaDB collection.
        """
        if not chunks:
            print("No chunks to add to ChromaDB.")
            return

        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "filename": chunk["source_filename"],
                "page_number": chunk["source_page_number"]
            }
            for chunk in chunks
        ]
        # Ensure IDs are unique strings
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]

        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = generate_embeddings(documents)
        print("Embeddings generated.")

        print(f"Adding {len(documents)} documents to ChromaDB collection '{self.collection.name}'...")
        try:
            self.collection.upsert( # Use upsert for idempotency (add or update if ID exists)
                documents=documents,
                embeddings=embeddings, # Pass the pre-generated embeddings
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(documents)} chunks to ChromaDB.")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")

    def query_collection(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the ChromaDB collection for semantically similar documents.
        """
        print(f"\nQuerying ChromaDB for: '{query_text}' (top {n_results} results)")
        
        # Generate embedding for the query
        query_embedding = generate_embeddings([query_text])[0] # encode returns a list of embeddings

        results = self.collection.query(
            query_embeddings=[query_embedding], # Pass the embedding for the query
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # What to include in the results
        )
        return results

# --- Main Execution ---
if __name__ == "__main__":
    
    print("Starting embedding generation and ChromaDB ingestion...")

    # Step 1: Process PDFs to get chunks
    print("Running PDF processing to get chunks...")
    chunks = process_pdfs_to_chunks(PDF_DIRECTORY)

    if not chunks:
        print("No chunks available for embedding. Please ensure PDFs are in data/pdfs and are text-searchable.")
    else:
        # Step 2: Initialize ChromaDB Manager
        chroma_manager = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME)

        # Step 3: Add chunks to ChromaDB
        chroma_manager.add_chunks_to_collection(chunks)

        # --- Example Queries ---
        # Your project's example queries:
        # "I am a transit fleet operation at City of Hamilton. I hear a lot about hydrogen and how it can help me decarbonize my operations. I currently have CNG/RNG buses and I want to explore the next generation clean technology. Where do I start my exploration journey in this topic?"
        # "I am looking for storing energy with lower cost."

        example_queries = [
            "How can hydrogen help decarbonize transit fleet operations?",
            "What are low-cost options for energy storage using hydrogen?",
            "What are typical infrastructure requirements for hydrogen fueling stations?",
            "Hydrogen blending with natural gas, what are the benefits?",
            "What are safety considerations for hydrogen storage?"
        ]

        for query in example_queries:
            query_results = chroma_manager.query_collection(query, n_results=3)
            
            print(f"\n--- Results for Query: '{query}' ---")
            if query_results and query_results['documents']:
                for i in range(len(query_results['documents'][0])):
                    doc_content = query_results['documents'][0][i]
                    metadata = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]
                    
                    print(f"Result {i+1} (Distance: {distance:.4f}):")
                    print(f"  Source: {metadata.get('filename', 'N/A')} (Page: {metadata.get('page_number', 'N/A')})")
                    print(f"  Content (first 200 chars): \"{doc_content[:200]}...\"")
                    print("-" * 20)
            else:
                print("No relevant results found.")
            print("=" * 50)