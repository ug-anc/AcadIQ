"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from app.routers import health
from app.routers import admin
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.config import get_settings
from app.routers import query
from app.routers import session
from app.security import limiter
from app.services.retrieval import initialize_retrieval_engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the vector store connection and build the BM25 index.
    await initialize_retrieval_engine()
    yield


app = FastAPI(
    title="AskIITK API",
    version=__version__,
    docs_url="/docs",  # enable in prototype; disable in production
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

app.include_router(query.router, prefix="/api/v1")
app.include_router(session.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(health.router)

# Serve the demo chat UI at "/".
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")
