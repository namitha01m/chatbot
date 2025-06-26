import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os
import torch
from typing import List, Dict, Any

from pdf_processor import process_pdfs_to_chunks, PDF_DIRECTORY

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(CURRENT_DIR, "chroma_db")
COLLECTION_NAME = "hydrogen_rules_of_thumb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_BATCH_SIZE = 5000

# --- Initialize Embedding Model ---
try:
    print(f"Loading Sentence-Transformer model: {EMBEDDING_MODEL_NAME}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Detected device: {device}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    print("Attempting to load model on CPU as fallback.")


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings_numpy = embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = embeddings_numpy.tolist()
    return embeddings

# --- ChromaDB Manager ---
class ChromaDBManager:
    def __init__(self, db_path: str, collection_name: str, create_new: bool = False):
        # The PersistentClient itself handles persistence implicitly.
        self.client = chromadb.PersistentClient(path=db_path)
        
        if create_new:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Existing collection '{collection_name}' deleted for a fresh start.")
            except Exception as e:
                print(f"No existing collection '{collection_name}' to delete, or error during deletion: {e}")
            self.collection = self.client.create_collection(name=collection_name)
        else:
            self.collection = self.client.get_or_create_collection(name=collection_name)

        print(f"ChromaDB client initialized. Collection '{collection_name}' ready at {db_path}")


    def add_chunks_to_collection(self, chunks: List[Dict[str, Any]]):
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
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]

        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = generate_embeddings(documents)
        print("Embeddings generated.")

        total_added = 0
        
        for i in range(0, len(documents), CHROMA_BATCH_SIZE):
            batch_documents = documents[i:i + CHROMA_BATCH_SIZE]
            batch_embeddings = embeddings[i:i + CHROMA_BATCH_SIZE]
            batch_metadatas = metadatas[i:i + CHROMA_BATCH_SIZE]
            batch_ids = ids[i:i + CHROMA_BATCH_SIZE]

            print(f"Adding batch {i // CHROMA_BATCH_SIZE + 1} of {len(batch_documents)} documents to ChromaDB collection '{self.collection.name}'...")
            try:
                self.collection.add(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                total_added += len(batch_documents)
                print(f"  Batch added. Total added so far: {total_added}")
            except Exception as e:
                print(f"Error adding batch starting with ID {batch_ids[0]}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Successfully attempted to add {total_added} chunks to ChromaDB.")
        
        current_count = self.collection.count()
        print(f"Current final count in collection: {current_count}") 

        # --- REMOVED: self.client.persist() ---
        # The PersistentClient now handles persistence implicitly.
        # print(f"ChromaDB client implicitly persisted to disk.") # Changed text
    def query_collection(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        print(f"\nQuerying ChromaDB for: '{query_text}' (top {n_results} results)")
        
        # Ensure embedding_model is accessible here, it's defined globally in vector_store_manager.py
        # You can either pass it to the ChromaDBManager __init__ or use 'global embedding_model' if needed
        # but since generate_embeddings is a global function, it should be fine.
        query_embedding = generate_embeddings([query_text])[0] 

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return results
# --- Main Execution for vector_store_manager.py ---
if __name__ == "__main__":
    print("Starting embedding generation and ChromaDB ingestion...")
    print(f"ChromaDB persistence path: {CHROMA_DB_PATH}")

    print("Running PDF processing to get chunks...")
    chunks = process_pdfs_to_chunks(PDF_DIRECTORY)

    if not chunks:
        print("No chunks available for embedding. Please ensure PDFs are in data/pdfs and are text-searchable.")
    else:
        chroma_manager = ChromaDBManager(CHROMA_DB_PATH, COLLECTION_NAME, create_new=True)
        chroma_manager.add_chunks_to_collection(chunks)

        print("ChromaDB ingestion process finished.")
        print("-" * 50)
        print("IMPORTANT: Verify the 'chroma_db' folder exists and contains files.")
        print(f"You should see files in: {CHROMA_DB_PATH}")
        print("Then, run 'rag_agent.py' from the same directory.")
        print("-" * 50)