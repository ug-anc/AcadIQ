"""In-memory session store.

Stores conversation history per session. Prototype uses a dict; the interface
is shaped so a SQLite/Postgres backend can replace it without changing callers.

Each session holds:
  - session_id (UUID4, client-generated)
  - title (auto-set from first user query)
  - status: "active" | "archived"
  - created_at / updated_at (ISO 8601)
  - turns: ordered list of {role, content, turn_index}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Thread-safe-enough for the single-process prototype."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    # -- CRUD ----------------------------------------------------------------

    def create(self, session_id: str, *, title: str = "New chat") -> dict:
        """Create a new session. Idempotent — returns existing if present."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        now = _now_iso()
        session = {
            "session_id": session_id,
            "title": title,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        self._sessions[session_id] = session
        logger.info("Session created: %s", session_id)
        return session

    def get(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def list_sessions(
        self, *, status: str | None = None
    ) -> list[dict]:
        """Return sessions ordered by updated_at descending."""
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s["status"] == status]
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        # Return lightweight summaries (no full turn content)
        return [
            {
                "session_id": s["session_id"],
                "title": s["title"],
                "status": s["status"],
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
                "turn_count": len(s["turns"]),
            }
            for s in sessions
        ]

    def archive(self, session_id: str) -> bool:
        """Mark a session as archived. Returns False if not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session["status"] = "archived"
        session["updated_at"] = _now_iso()
        logger.info("Session archived: %s", session_id)
        return True

    # -- Turn management -----------------------------------------------------

    def add_turn(
        self, session_id: str, *, role: Role, content: str
    ) -> int:
        """Append a turn and return its 0-based index.

        Auto-creates the session if it doesn't exist yet.
        Auto-titles the session from the first user message.
        """
        if session_id not in self._sessions:
            self.create(session_id)
        session = self._sessions[session_id]
        turn_index = len(session["turns"])
        session["turns"].append(
            {"role": role, "content": content, "turn_index": turn_index}
        )
        session["updated_at"] = _now_iso()

        # Auto-title from first user message
        if role == "user" and turn_index == 0:
            session["title"] = content[:80].strip()

        return turn_index

    def get_history(
        self, session_id: str, *, max_turns: int | None = None
    ) -> list[dict]:
        """Return conversation turns for the contextualizer.

        Returns the most recent `max_turns` turns (or all if None).
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        turns = session["turns"]
        if max_turns is not None and len(turns) > max_turns:
            turns = turns[-max_turns:]
        return turns

    def get_turn_count(self, session_id: str) -> int:
        session = self._sessions.get(session_id)
        return len(session["turns"]) if session else 0


# -- Singleton ---------------------------------------------------------------

_STORE: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE
