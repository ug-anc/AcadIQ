"""Load the bundled sample manual so the demo works with zero PDFs and zero keys.

Usage:  python -m scripts.load_sample_corpus
"""

import asyncio

from app.ingestion.extractor import extracted_from_text
from app.ingestion.pipeline import ingest_extracted

SAMPLE_PATH = "data/sample_ug_manual.txt"


async def _run() -> None:
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
