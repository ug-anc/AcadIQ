"""Application configuration.

All tunables live here. The provider fields (EMBEDDING_PROVIDER / LLM_PROVIDER /
RERANKER) let the same codebase run in three modes:

  - "openai" / "cohere": production path described in the PRD.
  - "local":             zero-key demo path (deterministic hash embeddings,
                         extractive generation, identity reranker) so the full
                         pipeline runs end-to-end with no network or API keys.

When DEMO_MODE=true every provider is forced to its local implementation.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    from dotenv import load_dotenv
    load_dotenv()  
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # ---- Mode -------------------------------------------------------------
    # DEMO_MODE forces local providers so the app runs with zero API keys.
    # DEMO_MODE: bool = True
    DEMO_MODE: bool = False

    # ---- Providers (overridden to "local" when DEMO_MODE is true) ---------
    EMBEDDING_PROVIDER: Literal["openai", "local"] = "openai"
    LLM_PROVIDER: Literal["openai", "local"] = "openai"
    RERANKER: Literal["cohere", "identity"] = "cohere"

    LLM_GEN_MODEL: str = "llama-3.1-8b-instant"
    LLM_REWRITE_MODEL: str = "llama-3.1-8b-instant"


    # ---- LLM / Embeddings -------------------------------------------------
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_GENERATION_MODEL: str = "gpt-4o"
    OPENAI_QUERY_REWRITE_MODEL: str = "gpt-4o-mini"

    # ---- Reranker ---------------------------------------------------------
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"

    # ---- Vector store -----------------------------------------------------
    # Prototype uses an embedded persistent Chroma client (no server needed).
    CHROMA_PERSIST_DIR: str = "storage/chroma"
    CHROMA_COLLECTION: str = "academiq_chunks"

    # ---- Retrieval tuning -------------------------------------------------
    RETRIEVAL_TOP_K_DENSE: int = 20
    RETRIEVAL_TOP_K_FINAL: int = 5
    ENABLE_QUERY_REWRITE: bool = True

    # ---- Confidence gate (DR-04: dual threshold) --------------------------
    # Signal is the reranker relevance score of the top chunk (0..1), or raw
    # cosine when reranking is unavailable. Tune with eval/calibrate_threshold.py.
    CONFIDENCE_HARD_REJECT: float = 0.30
    CONFIDENCE_SOFT_WARN: float = 0.50

    # ---- Ingestion --------------------------------------------------------
    MAX_SECTION_TOKENS: int = 800
    SCAN_TEXT_DENSITY_CHARS: int = 50  # < this with images present => scanned
    ENABLE_OCR: bool = True

    # ---- Cache ------------------------------------------------------------
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_ENTRIES: int = 512

    # ---- Security ---------------------------------------------------------
    API_SECRET_KEY: str = "change-me-in-prod"  # admin endpoints
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    RATE_LIMIT_PER_MINUTE: int = 30

    # ---- Feedback store ---------------------------------------------------
    FEEDBACK_DB_PATH: str = "storage/feedback.db"

    # ---- Document corpus --------------------------------------------------
    PDF_DIR: str = "data/pdfs"
    COLLEGE_NAME: str = "IIT KANPUR"

    # ---- Session / multi-turn ---------------------------------------------
    SESSION_MAX_HISTORY_TURNS: int = 10  # max turns fed to the contextualizer
    SESSION_CONTEXTUALIZER_MODEL: str = "llama-3.1-8b-instant"

    # ---- CRAG (Corrective RAG) ---------------------------------------------
    CRAG_EVAL_MODEL: str = "llama-3.1-8b-instant"

    # ----------------------------------------------------------------------
    def resolve_providers(self) -> None:
        """In demo mode, pin every provider to its local implementation."""
        if self.DEMO_MODE:
            self.EMBEDDING_PROVIDER = "local"
            self.LLM_PROVIDER = "local"
            self.RERANKER = "identity"
            return
        # Outside demo mode, downgrade gracefully if keys are missing.
        if not self.OPENAI_API_KEY:
            self.EMBEDDING_PROVIDER = "local"
            self.LLM_PROVIDER = "local"
        if not self.COHERE_API_KEY:
            self.RERANKER = "identity"


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    s.resolve_providers()
    return s
