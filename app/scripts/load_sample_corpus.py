"""Load the bundled sample manual so the demo works with zero PDFs and zero keys.

Usage:  python -m app.scripts.load_sample_corpus
"""

import asyncio
import os

from app.ingestion.extractor import extracted_from_text
from app.ingestion.pipeline import ingest_extracted

SAMPLE_PATH = "app/data/sample_ug_manual.txt"


async def _run() -> None:
    if not os.path.exists(SAMPLE_PATH):
        print(f"Error: {SAMPLE_PATH} not found. Please run extraction first.")
        return
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        text = fh.read()
    doc = extracted_from_text("UG_Manual_2024.pdf", text)
    report = await ingest_extracted(doc)
    print(f"Ingested sample corpus: {report['chunks_ingested']} chunks "
          f"from {report['source_document']}.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
