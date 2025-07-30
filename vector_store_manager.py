import os
import faiss
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional


from pdf_processor import process_pdfs_to_chunks, PDF_DIRECTORY


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index.bin")
FAISS_METADATA_PATH = os.path.join(CURRENT_DIR, "faiss_metadata.json") 

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_BATCH_SIZE = 5000 

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


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


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts.
    Returns a numpy array of embeddings.
    """
    embeddings = embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings


class FAISSManager:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.IndexFlatL2] = None 
        self.documents_and_metadatas: List[Dict[str, Any]] = [] 

        self._load_index_and_metadata()
        print(f"FAISS Manager initialized. Index path: {self.index_path}, Metadata path: {self.metadata_path}")

    def _load_index_and_metadata(self):
        """
        Attempts to load the FAISS index and associated metadata from disk.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents_and_metadatas = json.load(f)
                print(f"Loaded FAISS index and metadata from disk. Contains {len(self.documents_and_metadatas)} documents.")
            except Exception as e:
                print(f"Error loading FAISS index or metadata: {e}. Starting with empty index.")
                self.index = None 
                self.documents_and_metadatas = []
        else:
            print("No existing FAISS index or metadata found. Will create new upon ingestion.")
            self.index = None
            self.documents_and_metadatas = []

    def save_index_and_metadata(self):
        """
        Saves the FAISS index and associated metadata to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents_and_metadatas, f, indent=4)
            print(f"FAISS index and metadata saved to {self.index_path} and {self.metadata_path}.")
        else:
            print("No FAISS index to save.")

    def add_chunks_to_collection(self, chunks: List[Dict[str, Any]]):
        """
        Adds text chunks to the FAISS index and stores their content/metadata.
        """
        if not chunks:
            print("No chunks to add to FAISS.")
            return

        documents = [chunk["content"] for chunk in chunks]
        
        
        prepared_metadatas = []
        for chunk in chunks:
            meta = {
                "content": chunk["content"], 
                "source_type": chunk.get("source_type", "pdf"), 
                "filename": chunk.get("source_filename", "N/A"),
                "page_number": chunk.get("source_page_number", "N/A"),
                "chunk_id": chunk["chunk_id"]
            }
            prepared_metadatas.append(meta)

        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = generate_embeddings(documents) 
        print("Embeddings generated.")

        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d) 
            print(f"Created new FAISS IndexFlatL2 with dimension {d}.")
        
        
        self.index.add(embeddings)
        print(f"Added {embeddings.shape[0]} embeddings to FAISS index.")

       
        self.documents_and_metadatas.extend(prepared_metadatas)
        print(f"Stored {len(prepared_metadatas)} document contents and metadatas.")

        self.save_index_and_metadata() 

    def count(self) -> int:
        """Returns the number of documents in the FAISS index."""
        if self.index:
            return self.index.ntotal
        return 0

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, List[List[Any]]]:
        """
        Queries the FAISS index for relevant documents.
        Returns a dictionary mimicking ChromaDB's query result structure for compatibility.
        """
        if self.index is None or self.index.ntotal == 0:
            print("FAISS index is empty or not initialized. Cannot query.")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        print(f"\nQuerying FAISS index for: '{query_text}' (top {n_results} results)")
        
        query_embedding = generate_embeddings([query_text]) 
        
        
        distances, indices = self.index.search(query_embedding, n_results)

        retrieved_documents = []
        retrieved_metadatas = []
        retrieved_distances = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            original_data = self.documents_and_metadatas[idx]
            retrieved_documents.append(original_data["content"])
            
            
            meta_copy = original_data.copy()
            meta_copy.pop("content", None) 
            retrieved_metadatas.append(meta_copy)
            
            retrieved_distances.append(distances[0][i])

        
        return {
            'documents': [retrieved_documents],
            'metadatas': [retrieved_metadatas],
            'distances': [retrieved_distances]
        }


if __name__ == "__main__":
    print("Starting embedding generation and FAISS ingestion...")
    print(f"FAISS Index path: {FAISS_INDEX_PATH}")
    print(f"FAISS Metadata path: {FAISS_METADATA_PATH}")

    delete_existing = True 

    if delete_existing:
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
            print(f"Existing FAISS index '{FAISS_INDEX_PATH}' deleted.")
        if os.path.exists(FAISS_METADATA_PATH):
            os.remove(FAISS_METADATA_PATH)
            print(f"Existing FAISS metadata '{FAISS_METADATA_PATH}' deleted.")
        print("Starting with a fresh FAISS index.")
    
    faiss_manager = FAISSManager(FAISS_INDEX_PATH, FAISS_METADATA_PATH)

    print("\nProcessing PDFs...")
    pdf_chunks = process_pdfs_to_chunks(PDF_DIRECTORY, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
    
    all_chunks = []
    chunk_id_counter = 0
    # Assigning unique chunk IDs
    for chunk in pdf_chunks:
        chunk['chunk_id'] = chunk_id_counter
        chunk_id_counter += 1
    all_chunks.extend(pdf_chunks)
    print(f"Total PDF chunks prepared: {len(pdf_chunks)}")

    if not all_chunks:
        print("No chunks available for embedding. Please ensure PDFs are in data/pdfs and are text-searchable.")
    else:
        print(f"\nTotal chunks for ingestion: {len(all_chunks)}")
        faiss_manager.add_chunks_to_collection(all_chunks) 

        print("\nFAISS ingestion process finished.")
        print("-" * 50)
        print("IMPORTANT: Verify 'faiss_index.bin' and 'faiss_metadata.json' files exist.")
        print(f"You should see files in: {CURRENT_DIR}")
        print("Then, run 'app.py' from the same directory using: python -m streamlit run app.py")
        print("-" * 50)
