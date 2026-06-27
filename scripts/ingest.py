"""Ingest every PDF under data/pdfs into the vector store.

Usage:  python -m scripts.ingest [optional/pdf/dir]
"""

import sys

from app.ingestion.pipeline import ingest_directory_sync


def main() -> None:
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else None
    reports = ingest_directory_sync(pdf_dir)
    if not reports:
        print("No PDFs ingested. Place files in data/pdfs/ or run "
              "scripts.load_sample_corpus for a zero-PDF demo.")
        return
    total = sum(r["chunks_ingested"] for r in reports)
    for r in reports:
        scanned = f", scanned pages: {r['scanned_pages']}" if r["scanned_pages"] else ""
        print(f"  {r['source_document']}: {r['chunks_ingested']} chunks, "
              f"{r['tables_found']} tables{scanned}")
    print(f"Done. {total} chunks across {len(reports)} document(s).")


if __name__ == "__main__":
    main()
