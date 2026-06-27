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
from app.services.generation import answer_query

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

    cache = get_cache()
    cached = cache.get(payload.query)
    if cached is not None:
        resp = QueryResponse(**cached)
        resp.cached = True
        resp.session_id = payload.session_id
        return resp

    response = await answer_query(payload.query, payload.session_id)
    cache.set(payload.query, response.model_dump())
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
