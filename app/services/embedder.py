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
import httpx

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
    
class RemoteInferenceEmbedder(Embedder):
    """Memory-efficient embeddings using an external API gateway."""
    def __init__(self):
        s = get_settings()
        # Example using Hugging Face Free Inference API for the same 384-dim model
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        # Ensure you add HUGGINGFACE_API_KEY or OPENAI_API_KEY to your Render Environment Variables
        self.headers = {"Authorization": f"Bearer {s.HF_API_KEY}"}
        self.dim = 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            return response.json()

_EMBEDDER: Embedder | None = None

def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _EMBEDDER = RemoteInferenceEmbedder()
    return _EMBEDDER
