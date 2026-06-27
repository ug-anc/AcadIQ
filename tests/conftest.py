"""Pytest fixtures: run everything in demo mode against an isolated store."""

import os
import tempfile

import pytest

# Configure BEFORE app modules import settings.
_TMP = tempfile.mkdtemp(prefix="academiq_test_")
os.environ["DEMO_MODE"] = "true"
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(_TMP, "chroma")
os.environ["FEEDBACK_DB_PATH"] = os.path.join(_TMP, "feedback.db")
os.environ["CONFIDENCE_HARD_REJECT"] = "0.30"
os.environ["CONFIDENCE_SOFT_WARN"] = "0.50"


@pytest.fixture(scope="session", autouse=True)
def corpus():
    import asyncio

    from app.ingestion.extractor import extracted_from_text
    from app.ingestion.pipeline import ingest_extracted

    with open("data/sample_ug_manual.txt", encoding="utf-8") as fh:
        text = fh.read()
    doc = extracted_from_text("UG_Manual_2024.pdf", text)
    asyncio.run(ingest_extracted(doc))
    yield
