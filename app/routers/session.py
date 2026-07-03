"""Session CRUD routes.

GET  /sessions              — list all sessions (optionally filter by status)
POST /sessions              — create a new session
GET  /sessions/{session_id} — get full session with turn history
PATCH /sessions/{session_id}/archive — mark session as archived
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path

from app.models.schemas import (
    SessionDetail,
    SessionListResponse,
    SessionSummary,
    is_valid_session_id,
)
from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _validate_session_id_param(session_id: str) -> str:
    """Fix #3: Validate UUID4 format on path parameters — same pattern as
    the Pydantic validator in QueryRequest."""
    if not is_valid_session_id(session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id must be a valid UUID4",
        )
    return session_id


@router.get("", response_model=SessionListResponse)
async def list_sessions(status: str | None = None) -> SessionListResponse:
    """List sessions, optionally filtered by status ('active' or 'archived')."""
    store = get_session_store()
    sessions = store.list_sessions(status=status)
    return SessionListResponse(
        sessions=[SessionSummary(**s) for s in sessions]
    )


@router.post("", response_model=SessionSummary, status_code=201)
async def create_session(session_id: str) -> SessionSummary:
    """Create a new session. Idempotent — returns existing if present."""
    _validate_session_id_param(session_id)
    store = get_session_store()
    session = store.create(session_id)
    return SessionSummary(
        session_id=session["session_id"],
        title=session["title"],
        status=session["status"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        turn_count=len(session["turns"]),
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str = Path(..., description="UUID4 session identifier"),
) -> SessionDetail:
    """Get full session with conversation history."""
    _validate_session_id_param(session_id)
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(**session)


@router.patch("/{session_id}/archive", response_model=SessionSummary)
async def archive_session(
    session_id: str = Path(..., description="UUID4 session identifier"),
) -> SessionSummary:
    """Archive a session (soft delete)."""
    _validate_session_id_param(session_id)
    store = get_session_store()
    if not store.archive(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    session = store.get(session_id)
    assert session is not None  # archive succeeded, so it must exist
    return SessionSummary(
        session_id=session["session_id"],
        title=session["title"],
        status=session["status"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        turn_count=len(session["turns"]),
    )
