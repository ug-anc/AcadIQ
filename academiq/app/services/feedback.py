"""Feedback persistence.

Prototype uses SQLite (zero-infra, file-backed). Swap the connection string for
PostgreSQL in production — the schema and access functions stay identical.
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


def _connect() -> sqlite3.Connection:
    path = get_settings().FEEDBACK_DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def record_feedback(
    *, session_id: str | None, query: str, answer: str, helpful: bool, comment: str | None
) -> int:
    conn = _connect()
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
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        helpful = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE helpful = 1"
        ).fetchone()[0]
        return {"total": total, "helpful": helpful, "unhelpful": total - helpful}
    finally:
        conn.close()
