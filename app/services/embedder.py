"""Embedding providers.

`get_embedder()` returns either the OpenAI embedder (production) or a
deterministic local embedder (demo/offline). Both expose the same async
`embed(texts) -> list[list[float]]` interface so the rest of the pipeline never
needs to know which one is active.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from app.config import Settings, get_settings
from sentence_transformers import SentenceTransformer

class Embedder(ABC):
    dim: int
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

class LocalTransformerEmbedder(Embedder):
    """High-quality local embeddings using sentence-transformers."""
    def __init__(self):
        # This downloads a small, powerful model once
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # This runs on your CPU, 100% free
        return self.model.encode(texts).tolist()

_EMBEDDER: Embedder | None = None

def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _EMBEDDER = LocalTransformerEmbedder()
    return _EMBEDDER
