import fitz # PyMuPDF
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter 


PDF_DIRECTORY = "./Data"

os.makedirs(PDF_DIRECTORY, exist_ok=True)

def process_pdfs_to_chunks(pdf_dir: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> List[Dict[str, Any]]:
    """
    Processes PDF files from a directory into text chunks with metadata.
    Accepts chunk_size_tokens and chunk_overlap_tokens for text splitting.
    """
    all_pdf_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        length_function=len, 
        is_separator_regex=False,
    )

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Please place your PDF documents there.")
        return []

    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\nProcessing PDF: {pdf_file}")
        text_content = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            doc.close()
            print(f"Extracted {len(text_content)} characters from {pdf_file}.")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue

        if not text_content.strip():
            print(f"Skipping {pdf_file} as no readable text was extracted.")
            continue
        
        # Spliting the document text into chunks
        chunks = text_splitter.split_text(text_content)
        
        if not chunks:
            print(f"No chunks generated for {pdf_file}. Content might be too short or unusual.")
            continue
        
        print(f"Generated {len(chunks)} chunks for {pdf_file}.")

        # Adding chunks with metadata
        for i, chunk_content in enumerate(chunks):
            all_pdf_chunks.append({
                "content": chunk_content,
                "source_type": "pdf",
                "source_filename": pdf_file,
                "source_page_number": "N/A" 
            })
    print(f"\nFinished processing all PDFs. Total PDF chunks: {len(all_pdf_chunks)}")
    return all_pdf_chunks


if __name__ == "__main__":
   
    test_pdf_dir = "./Data"
    os.makedirs(test_pdf_dir, exist_ok=True) 
    

    processed_chunks = process_pdfs_to_chunks(test_pdf_dir, 500, 100)
    if processed_chunks:
        print(f"\nExample of processed PDF chunk (first 200 chars):")
        print(f"Content: \"{processed_chunks[0]['content'][:200]}...\"")
        print(f"Source: {processed_chunks[0]['source_filename']} (Page: {processed_chunks[0]['source_page_number']})")
    else:
        print("No PDF chunks were processed.")
