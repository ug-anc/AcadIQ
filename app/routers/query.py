"""Public query endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import (
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    looks_like_chitchat,
    looks_like_injection,
)
from app.prompts import CHITCHAT_RESPONSE, INJECTION_RESPONSE
from app.security import limiter
from app.services.cache import get_cache
from app.services.feedback import record_feedback
import logging
from app.services.generation import answer_query_with_history
from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    # Defense in depth: the schema validator already screens injections and
    # raises 422, but re-check here in case validation is bypassed upstream.

    logger.info("Received query: '%s' | session_id: '%s'", payload.query, payload.session_id)
    if looks_like_injection(payload.query):
        logger.warning("Query blocked by injection screen: '%s'", payload.query)
        return QueryResponse(
            answer=INJECTION_RESPONSE,
            citations=[],
            confidence_score=0.0,
            is_found=False,
            confidence_band="not_found",
            session_id=payload.session_id,
        )

    # Greetings / "what can you do" meta-questions never have a grounded
    # answer in the UG Manual — answer them directly instead of running them
    # through retrieval, where they'd hit the confidence gate and come back
    # as a cold "not found".
    if looks_like_chitchat(payload.query):
        logger.info("Query recognized as chit-chat: '%s'", payload.query)
        return QueryResponse(
            answer=CHITCHAT_RESPONSE.format(COLLEGE_NAME=get_settings().COLLEGE_NAME),
            citations=[],
            confidence_score=1.0,
            is_found=True,
            confidence_band="chitchat",
            session_id=payload.session_id,
        )

    # Cache is keyed by (session_id, turn_count, raw_query) so the same literal query in
    # different turns gets independent, context-specific results.
    cache = get_cache()
    store = get_session_store()
    turn_count = store.get_turn_count(payload.session_id)
    cache_key = f"{payload.session_id}:{turn_count}:{payload.query}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for key: '%s'", cache_key)
        resp = QueryResponse(**cached)
        resp.cached = True
        resp.session_id = payload.session_id
        return resp

    logger.info("Cache miss for key: '%s'. Executing pipeline...", cache_key)
    response = await answer_query_with_history(payload.query, payload.session_id)
    cache.set(cache_key, response.model_dump())
    logger.info("Response generated. is_found: %s | band: %s", response.is_found, response.confidence_band)
    return response


@router.post("/feedback")
async def feedback(payload: FeedbackRequest) -> JSONResponse:
    fid = record_feedback(
        session_id=payload.session_id,
        query=payload.query,
        answer=payload.answer,
        helpful=payload.helpful,
        comment=payload.comment,
    )
    return JSONResponse({"status": "recorded", "id": fid})
