"""Health and readiness endpoint."""

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    try:
        count = get_vector_store().count()
    except Exception:
        count = -1
    return HealthResponse(
        status="ok",
        version=__version__,
        embedding_provider=s.EMBEDDING_PROVIDER,
        llm_provider=s.LLM_PROVIDER,
        reranker=s.RERANKER,
        chunk_count=count,
    )
