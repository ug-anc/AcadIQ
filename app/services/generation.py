"""End-to-end answer generation.

Pipeline: retrieve -> confidence gate -> (generate | NOT-FOUND) -> citation
enforcement -> structured response. The gate runs *before* the LLM, so when the
top chunk is below the hard-reject threshold no generation call is made at all —
this is the cheapest and strongest hallucination defense.
"""

from __future__ import annotations

import re
import time

from app.config import get_settings
from app.models.schemas import QueryResponse, SourceCitation
from app.prompts import NOT_FOUND_RESPONSE
from app.services import confidence
from app.services.llm import Chunk, get_llm, render_prompt
from app.services.retrieval import RetrievedChunk, get_retrieval_engine

# CITATION_RE = re.compile(
#     r"\[Source:\s*(?P<doc>.+?),\s*Section\s*(?P<section>.+?),\s*Page\s*(?P<page>\d+)\]"
# )
CITATION_RE = re.compile(
    r"(?:\[Source:\s*(?P<doc>.+?),\s*Section\s*(?P<section>.+?),\s*Page\s*(?P<page>\d+)\])|(?:\[(?P<id>\d+)\])"
)
LOW_CONFIDENCE_BANNER = "[LOW CONFIDENCE — verify with the official source]\n\n"


def validate_citations_present(text: str, minimum: int = 1) -> bool:
    return len(CITATION_RE.findall(text)) >= minimum


# def parse_citations(
#     text: str, chunks: list[RetrievedChunk]
# ) -> list[SourceCitation]:
#     """Build citation objects, attaching a short excerpt from the matching chunk."""
#     excerpt_by_section = {
#         c.metadata.get("section_number", ""): c.text for c in chunks
#     }
#     citations: list[SourceCitation] = []
#     seen: set[tuple[str, str, int]] = set()
#     for m in CITATION_RE.finditer(text):
#         doc = m.group("doc").strip()
#         section = m.group("section").strip()
#         page = int(m.group("page"))
#         key = (doc, section, page)
#         if key in seen:
#             continue
#         seen.add(key)
#         excerpt = excerpt_by_section.get(section, "")[:150]
#         citations.append(
#             SourceCitation(
#                 document_name=doc, section=section, page_number=page, excerpt=excerpt
#             )
#         )
#     return citations

# def parse_citations(
#     text: str, chunks: list[RetrievedChunk]
# ) -> list[SourceCitation]:
#     """Build citation objects for [1], [2] style citations."""
#     citations: list[SourceCitation] = []
#     seen: set[int] = set()

#     # Find all matches of [1], [2], etc.
#     for m in re.finditer(r"\[(\d+)\]", text):
#         idx = int(m.group(1)) - 1  # Convert to 0-based index

#         # Ensure the index is valid for our chunks list
#         if idx < 0 or idx >= len(chunks) or idx in seen:
#             continue

#         seen.add(idx)
#         chunk = chunks[idx]

#         # Extract metadata directly from the chunk
#         doc = chunk.metadata.get("source_document", "Unknown")
#         section = chunk.metadata.get("section_number", "N/A")
#         page = int(chunk.metadata.get("page_number", 0))
#         excerpt = chunk.text[:150]
#         base_url = "https://iitk.ac.in/doaa/data/UG_Manual.pdf"
#         citations.append(
#             SourceCitation(
#                 document_name=doc,
#                 section=section,
#                 page_number=page,
#                 excerpt=excerpt,
#                 file_url=f"{base_url}#page={page}"
#             )
#         )
#     return citations


def parse_citations(
    text: str, chunks: list[RetrievedChunk]
) -> list[SourceCitation]:
    """Build citation objects, handling both [1] and [Source: ...] formats."""
    citations: list[SourceCitation] = []
    seen: set = set()
    base_url = "https://iitk.ac.in/doaa/data/UG_Manual.pdf"

    for m in CITATION_RE.finditer(text):
        # 1. Handle new [1] style
        if m.group("id"):
            idx = int(m.group("id")) - 1
            if idx < 0 or idx >= len(chunks) or idx in seen:
                continue
            seen.add(idx)
            chunk = chunks[idx]
            doc = chunk.metadata.get("source_document", "Unknown")
            section = chunk.metadata.get("section_number", "N/A")
            page = int(chunk.metadata.get("page_number", 0))

        # 2. Handle legacy [Source: ...] style
        elif m.group("doc"):
            doc = m.group("doc").strip()
            section = m.group("section").strip()
            page = int(m.group("page"))
            key = (doc, section, page)
            if key in seen:
                continue
            seen.add(key)
        else:
            continue

        excerpt = next((c.text for c in chunks if c.metadata.get("section_number") == section), "")[:150]

        citations.append(
            SourceCitation(
                document_name=doc,
                section=section,
                page_number=page,
                excerpt=excerpt,
                file_url=f"{base_url}#page={page}"
            )
        )
    return citations







def _not_found_response(
    score: float, band: str, retrieval_ms: int, session_id: str | None
) -> QueryResponse:
    return QueryResponse(
        answer=NOT_FOUND_RESPONSE,
        citations=[],
        confidence_score=round(score, 4),
        is_found=False,
        confidence_band=band,
        session_id=session_id,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=0,
    )


async def answer_query(query: str, session_id: str | None = None) -> QueryResponse:
    s = get_settings()
    engine = get_retrieval_engine()
    llm = get_llm()

    # Optional query rewrite for short, colloquial queries (DR-07).
    search_query = query
    if s.ENABLE_QUERY_REWRITE and len(query.split()) <= 6:
        try:
            search_query = await llm.rewrite_query(query)
        except Exception:
            search_query = query  # never let rewrite failure break the request

    t0 = time.perf_counter()
    chunks = await engine.retrieve(search_query)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if not chunks:
        return _not_found_response(0.0, confidence.NOT_FOUND, retrieval_ms, session_id)

    decision = confidence.evaluate_confidence(chunks[0].score)
    if not decision.should_generate:
        return _not_found_response(
            decision.score, decision.band, retrieval_ms, session_id
        )

    # Generate against the master prompt.
    prompt_chunks = [
        Chunk(
            text=c.text,
            document_name=c.metadata.get("source_document", "Unknown"),
            section_number=c.metadata.get("section_number", "N/A"),
            page_number=int(c.metadata.get("page_number", 0)),
        )
        for c in chunks
    ]
    rendered = render_prompt(s.COLLEGE_NAME, prompt_chunks, query)

    t1 = time.perf_counter()
    raw = await llm.generate(rendered, query)
    generation_ms = int((time.perf_counter() - t1) * 1000)

    # If the model declined (RULE 3 NOT-FOUND text), surface it as not-found.
    if raw.strip().startswith("I could not find a verified answer"):
        return QueryResponse(
            answer=raw.strip(),
            citations=[],
            confidence_score=round(decision.score, 4),
            is_found=False,
            confidence_band=confidence.NOT_FOUND,
            session_id=session_id,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

    # Post-generation citation enforcement: no citation => reject as NOT-FOUND.
    if not validate_citations_present(raw):
        return _not_found_response(
            decision.score, confidence.NOT_FOUND, retrieval_ms, session_id
        )

    citations = parse_citations(raw, chunks)
    answer = raw.strip()
    if decision.band == confidence.LOW_CONFIDENCE:
        answer = LOW_CONFIDENCE_BANNER + answer

    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence_score=round(decision.score, 4),
        is_found=True,
        confidence_band=decision.band,
        session_id=session_id,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
    )
