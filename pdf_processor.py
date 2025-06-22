import os
import fitz  # PyMuPDF is imported as fitz
import tiktoken
from typing import List, Dict, Any

# --- Configuration ---
PDF_DIRECTORY = "Data" # Directory where your PDF files will be stored
CHUNK_SIZE_TOKENS = 500      # Target chunk size in tokens
CHUNK_OVERLAP_TOKENS = 50    # Overlap between chunks in tokens
ENCODING_NAME = "cl100k_base" # Encoding for token counting (e.g., for GPT models)

# --- Helper Function for Token Counting ---
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# --- PDF Text Extractor (using PyMuPDF) ---
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF using PyMuPDF, preserving page numbers and filename.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a page and contains 'text', 'filename',
                              and 'page_number'.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    extracted_pages_data = []
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)

        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text:
                extracted_pages_data.append({
                    "text": text,
                    "filename": filename,
                    "page_number": i + 1  # Page numbers are usually 1-indexed
                })
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path} using PyMuPDF: {e}")
    return extracted_pages_data

# --- Text Chunker ---
def chunk_text(
    pages_data: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
    start_chunk_id: int # New argument for continuous ID
) -> (List[Dict[str, Any]], int): # Returns chunks and the next available ID
    
    """
    Splits text from extracted pages into smaller chunks with overlap,
    while retaining source metadata.

    Args:
        pages_data (List[Dict[str, Any]]): List of dictionaries from extract_text_from_pdf.
        chunk_size (int): The maximum size of each chunk in tokens.
        chunk_overlap (int): The number of tokens to overlap between chunks.
        encoding_name (str): The encoding name for tokenization.
        start_chunk_id (int): The starting ID for chunks generated in this call.

    Returns:
        Tuple[List[Dict[str, Any]], int]: A list of dictionaries representing chunks
                                           and the next available chunk ID.
    """
    text_splitter = tiktoken.get_encoding(encoding_name)
    chunks = []
    chunk_id_current = start_chunk_id # Initialize with the passed starting ID

    for page_data in pages_data:
        full_text = page_data["text"]
        filename = page_data["filename"]
        page_number = page_data["page_number"]

        tokens = text_splitter.encode(full_text)
        
        start_index = 0
        while start_index < len(tokens):
            end_index = min(start_index + chunk_size, len(tokens))
            chunk_tokens = tokens[start_index:end_index]
            chunk_content = text_splitter.decode(chunk_tokens)

            chunks.append({
                "content": chunk_content,
                "source_filename": filename,
                "source_page_number": page_number,
                "chunk_id": chunk_id_current # Use the current unique ID
            })
            chunk_id_current += 1 # Increment for the next chunk

            if end_index == len(tokens):
                break
            
            start_index += chunk_size - chunk_overlap
            if start_index < 0:
                start_index = 0

    return chunks, chunk_id_current # Return chunks and the ID for the next call

# --- Main Processing Logic ---
def process_pdfs_to_chunks(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Processes all PDF files in a directory to extract and chunk text using PyMuPDF.
    Manages a continuous chunk ID across all files.

    Args:
        pdf_dir (str): The directory containing PDF files.

    Returns:
        List[Dict[str, Any]]: A list of all processed chunks.
    """
    all_chunks = []
    global_chunk_id_counter = 0 # This will be the continuous counter

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created directory: {pdf_dir}. Please place your PDF files here.")
        return []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing PDF: {filename} with PyMuPDF")
            pages_data = extract_text_from_pdf(pdf_path)
            if pages_data:
                # Pass the current global counter and receive the updated one
                chunks, global_chunk_id_counter = chunk_text(
                    pages_data,
                    CHUNK_SIZE_TOKENS,
                    CHUNK_OVERLAP_TOKENS,
                    ENCODING_NAME,
                    global_chunk_id_counter # Pass the current global ID
                )
                all_chunks.extend(chunks)
                print(f"  Extracted {len(pages_data)} pages and created {len(chunks)} chunks from {filename}.")
                print(f"  Next available global chunk ID: {global_chunk_id_counter}") # For tracking
            else:
                print(f"  No text extracted from {filename} using PyMuPDF. Ensure it's not a scanned PDF without OCR.")
    return all_chunks

if __name__ == "__main__":
    print("Starting PDF processing with PyMuPDF...")
    final_chunks = process_pdfs_to_chunks(PDF_DIRECTORY)

    if final_chunks:
        print(f"\nSuccessfully processed {len(final_chunks)} chunks in total.")
        print("\n--- Example Chunk ---")
        for i, chunk in enumerate(final_chunks[:3]): # Print first 3 chunks as example
            print(f"Chunk {i+1} (ID: {chunk['chunk_id']}):")
            print(f"  Source: {chunk['source_filename']} (Page: {chunk['source_page_number']})")
            print(f"  Content (first 200 chars): \"{chunk['content'][:200]}...\"")
            print(f"  Tokens in chunk: {num_tokens_from_string(chunk['content'], ENCODING_NAME)}")
            print("-" * 30)
    else:
        print("\nNo chunks were generated. Please check your PDF directory and files.")

    print("\nNext step: Generating embeddings and storing in a vector database.")