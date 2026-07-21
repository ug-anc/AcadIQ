"""Pydantic request and response models."""

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# First-line injection screen. The master system prompt (RULE 5) is the real
# defense; this just rejects the most blatant attempts before we spend tokens.
INJECTION_PATTERNS = [
    # Highly specific prompt injection patterns to avoid false positives on legitimate queries
    r"ignore\b[\w\s]{0,40}\b(system\s+instructions|system\s+rules|system\s+prompt)",
    r"disregard\b[\w\s]{0,40}\b(system\s+instructions|system\s+rules|system\s+prompt|previous\s+instructions|prior\s+instructions)",
    r"reveal\b[\w\s]{0,40}\b(system\s+prompt|system\s+instructions|instruction\s+set)",
    r"jailbreak\s+the\s+llm",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


# Greetings / small talk / "what can you do" meta-questions. Matched against
# the WHOLE (normalized) query, not as a substring — so "hi, what is the
# minimum CGPA to avoid probation" still falls through to real retrieval;
# only messages that are *just* a greeting or meta-question short-circuit.
_CHITCHAT_PATTERNS = [
    r"(h+i+|h+e+l+l+o+|h+e+y+|yo+|hola|greetings)(\s+(there|guys|team|everyone|friend|all))?",
    r"good\s*(morning|afternoon|evening|night)(\s+(there|guys|team|everyone|all))?",
    r"what'?s\s+up",
    r"what\s+(all\s+)?can\s+you\s+do",
    r"what\s+do\s+you\s+do",
    r"who\s+are\s+you",
    r"what\s+are\s+you",
    r"what\s+is\s+this(\s+bot|\s+assistant)?",
    r"help",
    r"how\s+can\s+you\s+help(\s+me)?",
    r"what\s+can\s+(i|you)\s+ask(\s+you)?",
    r"thanks?|thank\s*you|thx|ty",
    r"bye|goodbye|see\s*ya",
]
_CHITCHAT_RE = re.compile(
    r"^(?:" + "|".join(_CHITCHAT_PATTERNS) + r")[\s!.,?]*$",
    re.IGNORECASE,
)


def looks_like_chitchat(text: str) -> bool:
    return bool(_CHITCHAT_RE.match(text.strip()))


# -- UUID4 session_id validation (Fix #1 & #3) --------------------------------
# Canonical lowercase UUID4 with hyphens: 8-4-4-4-12 hex digits.
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_valid_session_id(value: str) -> bool:
    """Check that a session_id is a valid UUID4 string."""
    return bool(_UUID4_RE.match(value))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    session_id: str = Field(
        ...,
        description="Client-generated UUID4 session identifier",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        if looks_like_injection(v):
            raise ValueError("Query contains disallowed content")
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Fix #3: Validate session_id is UUID4 format — same pattern as
        the sanitize_query validator."""
        if not is_valid_session_id(v):
            raise ValueError(
                "session_id must be a valid UUID4 "
                "(e.g. '550e8400-e29b-41d4-a716-446655440000')"
            )
        return v


class SourceCitation(BaseModel):
    document_name: str
    section: str
    page_number: int
    excerpt: str = Field(..., max_length=150)
    file_url: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation] = []
    confidence_score: float
    is_found: bool
    confidence_band: str  # "found" | "low_confidence" | "not_found" | "chitchat"
    cached: bool = False
    session_id: str = ""
    standalone_query: Optional[str] = None  # contextualized query for debugging
    retrieval_latency_ms: int = 0
    generation_latency_ms: int = 0


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    answer: str
    helpful: bool
    comment: Optional[str] = Field(default=None, max_length=1000)


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    llm_provider: str
    reranker: str
    chunk_count: int


# -- Session models ----------------------------------------------------------

class SessionSummary(BaseModel):
    session_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    turn_count: int


class SessionDetail(BaseModel):
    session_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    turns: list[dict]


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
