"""Ingestion orchestration: extract -> chunk -> embed -> upsert -> rebuild BM25."""

from __future__ import annotations

import asyncio
import os

from app.config import get_settings
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.extractor import ExtractedDoc, extract_pdf, extracted_from_text
from app.services.embedder import get_embedder
from app.services.retrieval import get_retrieval_engine
from app.services.vector_store import get_vector_store

_EMBED_BATCH = 64


async def _embed_and_upsert(chunks: list[Chunk]) -> int:
    store = get_vector_store()
    embedder = get_embedder()
    total = 0
    for i in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[i : i + _EMBED_BATCH]
        vectors = await embedder.embed([c.text for c in batch])
        store.upsert_batch(
            [
                {
                    "chunk_id": c.chunk_id,
                    "embedding": v,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c, v in zip(batch, vectors)
            ]
        )
        total += len(batch)
    return total


async def ingest_extracted(doc: ExtractedDoc) -> dict:
    chunks = chunk_document(doc)
    count = await _embed_and_upsert(chunks)
    get_retrieval_engine().build_bm25()  # keep sparse index in sync
    return {
        "source_document": doc.source_document,
        "chunks_ingested": count,
        "scanned_pages": doc.scanned_pages,
        "tables_found": len(doc.tables),
    }


async def ingest_pdf(path: str) -> dict:
    doc = extract_pdf(path)
    return await ingest_extracted(doc)


async def ingest_directory(pdf_dir: str | None = None) -> list[dict]:
    s = get_settings()
    directory = pdf_dir or s.PDF_DIR
    reports: list[dict] = []
    if not os.path.isdir(directory):
        print(f"[warn] PDF directory not found: {directory}")
        return reports
    # ------------------------
    directory = pdf_dir or s.PDF_DIR
    print(f"DEBUG: Looking in directory: {directory}")
    print(f"DEBUG: Files found: {os.listdir(directory)}")

    reports: list[dict] = []
    # ------------------------

    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        if fname.lower().endswith(".pdf"):
            print(f"Ingesting {fname} ...")
            reports.append(await ingest_pdf(path))
        elif fname.lower().endswith(".txt"):
            print(f"Ingesting {fname} (text) ...")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            doc = extracted_from_text(fname, text)
            reports.append(await ingest_extracted(doc))
    return reports


# async def ingest_directory(pdf_dir: str | None = None) -> list[dict]:
#     s = get_settings()

#     # Resolve absolute path based on project root
#     base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     target_dir = pdf_dir or s.PDF_DIR

#     if not os.path.isabs(target_dir):
#         directory = os.path.join(base_dir, target_dir)
#     else:
#         directory = target_dir

#     print(f"DEBUG: Absolute directory resolved to: {directory}")

#     reports: list[dict] = []
#     if not os.path.isdir(directory):
#         print(f"[warn] PDF directory not found: {directory}")
#         return reports

#     print(f"DEBUG: Files found: {os.listdir(directory)}")

#     for fname in sorted(os.listdir(directory)):
#         path = os.path.join(directory, fname)
#         if fname.lower().endswith(".pdf"):
#             print(f"Ingesting {fname} ...")
#             reports.append(await ingest_pdf(path))
#         elif fname.lower().endswith(".txt"):
#             print(f"Ingesting {fname} (text) ...")
#             with open(path, encoding="utf-8") as fh:
#                 text = fh.read()
#             doc = extracted_from_text(fname, text)
#             reports.append(await ingest_extracted(doc))
#     return reports


def ingest_directory_sync(pdf_dir: str | None = None) -> list[dict]:
    return asyncio.run(ingest_directory(pdf_dir))



if __name__ == "__main__":
    import asyncio
    # This will trigger the ingestion for your folder
    results = asyncio.run(ingest_directory("app/data"))
    print(f"Ingestion complete. Results: {results}")