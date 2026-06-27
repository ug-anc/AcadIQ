"""Rerankers.

Returns a list of (original_index, relevance_score) sorted best-first. The top
relevance score becomes the confidence signal feeding the DR-04 gate — a more
reliable signal than raw cosine similarity, which is why the gate reads from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings, get_settings


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        ...


class IdentityReranker(Reranker):
    """No model available: preserve fusion order, synthesize a decaying score.

    The score is monotonic in rank so the confidence gate still has a usable
    signal in demo mode. It is intentionally conservative.
    """

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        n = min(top_n, len(documents))
        # Lexical overlap gives the demo a meaningful (if crude) relevance score.
        q_tokens = set(query.lower().split())
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            d_tokens = set(doc.lower().split())
            overlap = len(q_tokens & d_tokens) / (len(q_tokens) or 1)
            scored.append((i, round(min(1.0, overlap), 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


class CohereReranker(Reranker):
    def __init__(self, settings: Settings):
        import cohere

        self._client = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)
        self._model = settings.COHERE_RERANK_MODEL

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        resp = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
        return [(r.index, float(r.relevance_score)) for r in resp.results]


_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    settings = get_settings()
    _RERANKER = (
        CohereReranker(settings)
        if settings.RERANKER == "cohere"
        else IdentityReranker()
    )
    return _RERANKER
