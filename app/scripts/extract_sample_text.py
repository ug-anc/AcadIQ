import os
import asyncio
import fitz  # PyMuPDF

from app.ingestion.extractor import extracted_from_text
from app.ingestion.pipeline import ingest_extracted

async def ingest(txt_path: str):
    print("Loading and embedding text into ChromaDB...")
    with open(txt_path, encoding="utf-8") as fh:
        text = fh.read()
    doc = extracted_from_text("UG_Manual_2024.pdf", text)
    report = await ingest_extracted(doc)
    print(f"Successfully ingested sample corpus: {report['chunks_ingested']} chunks "
          f"from {report['source_document']}.")

def main():
    pdf_path = "app/data/UG_Manual.pdf"
    txt_path = "app/data/sample_ug_manual.txt"
    
    print(f"Reading {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
    doc = fitz.open(pdf_path)
    
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
        
    full_text = "\f".join(pages_text)
    
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Successfully wrote extracted text to {txt_path} ({len(doc)} pages).")
    
    # Run the ingestion in the event loop
    asyncio.run(ingest(txt_path))

if __name__ == "__main__":
    main()
