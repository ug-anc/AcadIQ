"""Feedback persistence.

Uses GCP Firestore in production for stateless serverless persistence, 
and falls back to SQLite for local development (DEMO_MODE=True).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    query      TEXT NOT NULL,
    answer     TEXT NOT NULL,
    helpful    INTEGER NOT NULL,
    comment    TEXT,
    created_at TEXT NOT NULL
);
"""


def _connect_sqlite() -> sqlite3.Connection:
    path = get_settings().FEEDBACK_DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def record_feedback(
    *, session_id: str | None, query: str, answer: str, helpful: bool, comment: str | None
) -> str | int:
    s = get_settings()
    
    # Use Firestore in production (when DEMO_MODE is False)
    if not s.DEMO_MODE:
        try:
            from google.cloud import firestore
            db = firestore.Client()
            doc_ref = db.collection("feedback").document()
            doc_ref.set({
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "helpful": helpful,
                "comment": comment,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            return doc_ref.id
        except Exception as e:
            # Fall back to SQLite if Firestore client initialization or write fails
            # (or log it as a warning)
            import logging
            logging.getLogger(__name__).warning("Firestore write failed, falling back to SQLite: %s", e)

    # Local SQLite Fallback
    conn = _connect_sqlite()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (session_id, query, answer, helpful, comment, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                query,
                answer,
                1 if helpful else 0,
                comment,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def feedback_summary() -> dict:
    s = get_settings()
    
    # Use Firestore in production
    if not s.DEMO_MODE:
        try:
            from google.cloud import firestore
            db = firestore.Client()
            coll_ref = db.collection("feedback")
            total = coll_ref.count().get()[0][0].value
            helpful = coll_ref.where("helpful", "==", True).count().get()[0][0].value
            return {"total": total, "helpful": helpful, "unhelpful": total - helpful}
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Firestore read failed, falling back to SQLite: %s", e)

    # Local SQLite Fallback
    conn = _connect_sqlite()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        helpful = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE helpful = 1"
        ).fetchone()[0]
        return {"total": total, "helpful": helpful, "unhelpful": total - helpful}
    finally:
        conn.close()
