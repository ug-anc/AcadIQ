"""Pydantic request and response models."""

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# First-line injection screen. The master system prompt (RULE 5) is the real
# defense; this just rejects the most blatant attempts before we spend tokens.
INJECTION_PATTERNS = [
    # Allow filler words (e.g. "all previous") between the verb and the noun.
    r"ignore\b[\w\s]{0,40}\b(instructions|rules|prompt)",
    r"disregard\b[\w\s]{0,40}\b(instructions|rules|prompt|previous|above|prior)",
    r"you are now",
    r"system prompt",
    r"reveal your",
    r"pretend (to be|you are)",
    r"jailbreak",
    r"act as (a|an)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    session_id: Optional[str] = None

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        if looks_like_injection(v):
            raise ValueError("Query contains disallowed content")
        return v.strip()


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
    confidence_band: str  # "found" | "low_confidence" | "not_found"
    cached: bool = False
    session_id: Optional[str] = None
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
