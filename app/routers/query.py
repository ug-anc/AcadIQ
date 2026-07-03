"""Public query endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    looks_like_injection,
)
from app.prompts import INJECTION_RESPONSE
from app.security import limiter
from app.services.cache import get_cache
from app.services.feedback import record_feedback
from app.services.generation import answer_query_with_history

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    # Defense in depth: the schema validator already screens injections and
    # raises 422, but re-check here in case validation is bypassed upstream.
    if looks_like_injection(payload.query):
        return QueryResponse(
            answer=INJECTION_RESPONSE,
            citations=[],
            confidence_score=0.0,
            is_found=False,
            confidence_band="not_found",
            session_id=payload.session_id,
        )

    # Cache is keyed by (session_id, raw_query) so the same literal query in
    # different session contexts gets independent results.
    cache = get_cache()
    cache_key = f"{payload.session_id}:{payload.query}"
    cached = cache.get(cache_key)
    if cached is not None:
        resp = QueryResponse(**cached)
        resp.cached = True
        resp.session_id = payload.session_id
        return resp

    response = await answer_query_with_history(payload.query, payload.session_id)
    cache.set(cache_key, response.model_dump())
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
