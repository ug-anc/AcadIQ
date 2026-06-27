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

# from __future__ import annotations

# import hashlib
# import math
# from abc import ABC, abstractmethod
# from sentence_transformers import SentenceTransformer

# from app.config import Settings, get_settings

# # Local embedding dimensionality. Small + fixed so Chroma collections stay
# # consistent across runs in demo mode.
# LOCAL_DIM = 384


# class Embedder(ABC):
#     dim: int

#     @abstractmethod
#     async def embed(self, texts: list[str]) -> list[list[float]]:
#         ...


# def _l2_normalize(vec: list[float]) -> list[float]:
#     norm = math.sqrt(sum(x * x for x in vec)) or 1.0
#     return [x / norm for x in vec]


# class LocalHashEmbedder(Embedder):
#     """Deterministic bag-of-tokens hashing embedder.

#     Not semantically powerful, but stable and dependency-free: identical text
#     always maps to the same unit vector, so retrieval, ranking, and the demo UI
#     all behave deterministically without any network call.
#     """

#     dim = LOCAL_DIM

#     def _embed_one(self, text: str) -> list[float]:
#         vec = [0.0] * self.dim
#         tokens = [t for t in text.lower().split() if t]
#         for tok in tokens:
#             h = hashlib.sha256(tok.encode("utf-8")).digest()
#             # Two independent buckets per token reduce collisions.
#             idx1 = int.from_bytes(h[0:4], "big") % self.dim
#             idx2 = int.from_bytes(h[4:8], "big") % self.dim
#             sign = 1.0 if h[8] & 1 else -1.0
#             vec[idx1] += sign
#             vec[idx2] += sign * 0.5
#         return _l2_normalize(vec)

#     async def embed(self, texts: list[str]) -> list[list[float]]:
#         return [self._embed_one(t) for t in texts]


# class OpenAIEmbedder(Embedder):
#     def __init__(self, settings: Settings):
#         from openai import AsyncOpenAI  # imported lazily so demo mode needs no dep

#         self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
#         self._model = settings.OPENAI_EMBEDDING_MODEL
#         self.dim = 3072  # text-embedding-3-large

#     async def embed(self, texts: list[str]) -> list[list[float]]:
#         resp = await self._client.embeddings.create(model=self._model, input=texts)
#         return [d.embedding for d in resp.data]


# _EMBEDDER: Embedder | None = None


# def get_embedder() -> Embedder:
#     global _EMBEDDER
#     if _EMBEDDER is not None:
#         return _EMBEDDER
#     settings = get_settings()
#     if settings.EMBEDDING_PROVIDER == "openai":
#         _EMBEDDER = OpenAIEmbedder(settings)
#     else:
#         _EMBEDDER = LocalHashEmbedder()
#     return _EMBEDDER
