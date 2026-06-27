"""Admin endpoints — re-ingest documents without a code deploy."""

import os
import tempfile

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import get_settings
from app.ingestion.pipeline import ingest_directory, ingest_pdf
from app.security import require_admin_key
from app.services.feedback import feedback_summary

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin_key)])


@router.post("/admin/reingest")
async def reingest_all() -> dict:
    reports = await ingest_directory()
    return {"status": "ok", "documents": reports}


@router.post("/admin/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    s = get_settings()
    os.makedirs(s.PDF_DIR, exist_ok=True)
    dest = os.path.join(s.PDF_DIR, file.filename or "uploaded.pdf")
    with open(dest, "wb") as fh:
        fh.write(await file.read())
    report = await ingest_pdf(dest)
    return {"status": "ok", "report": report}


@router.get("/admin/feedback-summary")
async def get_feedback_summary() -> dict:
    return feedback_summary()
