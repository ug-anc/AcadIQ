"""End-to-end answer generation.

Pipeline: retrieve -> confidence gate -> (generate | NOT-FOUND) -> citation
enforcement -> structured response. The gate runs *before* the LLM, so when the
top chunk is below the hard-reject threshold no generation call is made at all —
this is the cheapest and strongest hallucination defense.

Multi-turn extension: answer_query_with_history() contextualizes follow-up
queries via the LLM contextualizer before running the standard pipeline.
RULE H2 ensures every turn triggers fresh retrieval.
"""

from __future__ import annotations

import logging
import re
import time

from app.config import get_settings
from app.models.schemas import QueryResponse, SourceCitation
from app.prompts import NOT_FOUND_RESPONSE
from app.services import confidence
from app.services.llm import Chunk, get_llm, render_prompt
from app.services.retrieval import RetrievedChunk, get_retrieval_engine
from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)

# CITATION_RE = re.compile(
#     r"\[Source:\s*(?P<doc>.+?),\s*Section\s*(?P<section>.+?),\s*Page\s*(?P<page>\d+)\]"
# )
CITATION_RE = re.compile(
    r"(?:\[Source:\s*(?P<doc>.+?),\s*Section\s*(?P<section>.+?),\s*Page\s*(?P<page>\d+)\])|(?:\[(?P<id>\d+)\])"
)
LOW_CONFIDENCE_BANNER = "[LOW CONFIDENCE — verify with the official source]\n\n"


def validate_citations_present(text: str, minimum: int = 1) -> bool:
    return len(CITATION_RE.findall(text)) >= minimum


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
    score: float, band: str, retrieval_ms: int, session_id: str,
    standalone_query: str | None = None,
) -> QueryResponse:
    return QueryResponse(
        answer=NOT_FOUND_RESPONSE,
        citations=[],
        confidence_score=round(score, 4),
        is_found=False,
        confidence_band=band,
        session_id=session_id,
        standalone_query=standalone_query,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=0,
    )


async def answer_query(
    query: str, session_id: str = "", history: list[dict] | None = None
) -> QueryResponse:
    """Original single-turn pipeline (backward-compatible)."""
    s = get_settings()
    engine = get_retrieval_engine()
    llm = get_llm()

    # Query Rewriting Step (Prior to Vector Search)
    search_queries = [query]
    if s.ENABLE_QUERY_REWRITE:
        try:
            search_queries = await llm.rewrite_query_to_search_terms(query)
            logger.info("Rewrote query '%s' to search terms: %s", query, search_queries)
        except Exception as e:
            logger.error(f"Error in query rewrite step: {e}", exc_info=True)
            search_queries = [query]

    t0 = time.perf_counter()
    chunks = await engine.retrieve(search_queries, original_query=query)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("Retrieved %d candidate chunks in %d ms", len(chunks), retrieval_ms)

    if not chunks:
        logger.warning("No chunks retrieved. Aborting LLM generation.")
        return _not_found_response(0.0, confidence.NOT_FOUND, retrieval_ms, session_id)

    decision = confidence.evaluate_confidence(chunks[0].score)
    logger.info("Top chunk score: %.4f | Decision band: %s | should_generate: %s", chunks[0].score, decision.band, decision.should_generate)
    if not decision.should_generate:
        logger.warning("Confidence score %.4f is below hard reject threshold. Aborting LLM generation.", decision.score)
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
    rendered = render_prompt(s.COLLEGE_NAME, prompt_chunks, query, history)

    logger.info("Sending rendered prompt to LLM...")
    t1 = time.perf_counter()
    raw = await llm.generate(rendered, query)
    generation_ms = int((time.perf_counter() - t1) * 1000)
    logger.info("LLM responded in %d ms", generation_ms)

    # If the model declined (RULE 3 NOT-FOUND text), surface it as not-found.
    cleaned_raw = raw.strip()
    if cleaned_raw.startswith("I could not find a verified answer") or "I could not find a verified answer" in cleaned_raw:
        logger.warning("LLM output indicates failure to find verified answer.")
        return QueryResponse(
            answer=cleaned_raw,
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
        logger.warning("LLM output is missing valid citations. Rejecting as NOT-FOUND to avoid hallucination.")
        return _not_found_response(
            decision.score, confidence.NOT_FOUND, retrieval_ms, session_id
        )

    citations = parse_citations(raw, chunks)
    answer = raw.strip()
    # if decision.band == confidence.LOW_CONFIDENCE:
        # answer = LOW_CONFIDENCE_BANNER + answer

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


async def answer_query_with_history(
    raw_query: str, session_id: str
) -> QueryResponse:
    """Multi-turn pipeline: contextualize → retrieve → generate.

    1. Fetch conversation history from the session store.
    2. Run the contextualizer to produce a standalone query.
    3. Record the user turn in the session store.
    4. Run the standard answer pipeline with the standalone query.
    5. Record the assistant turn in the session store.
    6. Return the response with standalone_query for observability.

    RULE H2: Fresh retrieval happens on every turn — the standalone_query
    always goes through the full retrieve → gate → generate pipeline.
    """
    s = get_settings()
    store = get_session_store()
    llm = get_llm()

    # 1. Fetch history (capped to contextualizer window)
    history = store.get_history(
        session_id, max_turns=s.SESSION_MAX_HISTORY_TURNS
    )

    # 2. Contextualize (RULE H1: topics only, no claimed facts)
    standalone_query = await llm.contextualize_query(raw_query, history)
    logger.info(
        "Session %s | raw='%s' | standalone='%s'",
        session_id[:8], raw_query[:60], standalone_query[:60],
    )

    # 3. Record user turn
    store.add_turn(session_id, role="user", content=raw_query)

    # 4. Run the standard pipeline with the standalone query
    #    RULE H2: every turn does fresh retrieval
    response = await answer_query(standalone_query, session_id, history=history)
    response.standalone_query = standalone_query

    # 5. Record assistant turn
    store.add_turn(session_id, role="assistant", content=response.answer)

    return response

