"""Hybrid retrieval: dense (Chroma) + sparse (BM25) -> RRF fusion -> rerank.

The BM25 index is built from the vector store's documents at startup, so the
sparse and dense halves always share the same corpus. Short, colloquial student
queries (DR-07) are optionally rewritten before retrieval; BM25 also naturally
rescues keyword-heavy short queries that dense search alone would miss.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.services.embedder import get_embedder
from app.services.reranker import get_reranker
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0  # reranker relevance (the confidence signal)
    cosine_similarity: float = 0.0


@dataclass
class RetrievalEngine:
    _bm25: BM25Okapi | None = None
    _corpus: list[dict[str, Any]] = field(default_factory=list)
    _ready: bool = False

    def build_bm25(self) -> None:
        store = get_vector_store()
        self._corpus = store.all_documents()
        if self._corpus:
            tokenized = [_tokenize(c["text"]) for c in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    async def retrieve(
        self, query_or_queries: str | list[str], original_query: str | None = None
    ) -> list[RetrievedChunk]:
        s = get_settings()
        if not self._ready:
            self.build_bm25()
        if not self._corpus:
            return []

        # Ensure queries is a list of strings
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        orig_query = original_query or (query_or_queries if isinstance(query_or_queries, str) else queries[0])

        dense_results_list = []
        sparse_results_list = []

        # Run retrieval for each query term/statement
        for q in queries:
            # 1. dense retrieval for this term
            try:
                query_vec = (await get_embedder().embed([q]))[0]
                dense = get_vector_store().query(query_vec, s.RETRIEVAL_TOP_K_DENSE)
                logger.info("Dense retrieval for query '%s' returned %d chunks", q, len(dense))
                dense_results_list.append(dense)
            except Exception as e:
                logger.error(f"Dense retrieval error for '{q}': {e}", exc_info=True)
                dense_results_list.append([])

            # 2. sparse retrieval for this term
            sparse: list[dict[str, Any]] = []
            if self._bm25 is not None:
                try:
                    scores = self._bm25.get_scores(_tokenize(q))
                    ranked = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[: s.RETRIEVAL_TOP_K_DENSE]
                    sparse = [
                        {
                            "chunk_id": self._corpus[i]["chunk_id"],
                            "text": self._corpus[i]["text"],
                            "metadata": self._corpus[i]["metadata"],
                            "cosine_similarity": 0.0,
                        }
                        for i in ranked
                        if scores[i] > 0
                    ]
                    logger.info("Sparse retrieval for query '%s' returned %d chunks", q, len(sparse))
                except Exception as e:
                    logger.error(f"Sparse retrieval error for '{q}': {e}", exc_info=True)
                sparse_results_list.append(sparse)

        # 3. reciprocal rank fusion of all sources
        all_sources = dense_results_list + sparse_results_list
        fused = reciprocal_rank_fusion(*all_sources)[: s.RETRIEVAL_TOP_K_DENSE]
        logger.info("Reciprocal Rank Fusion returned %d combined chunks", len(fused))
        if not fused:
            return []

        # 4. rerank -> final top-k with relevance scores against the original query
        logger.info("Running reranker for original query '%s' on %d chunks", orig_query, len(fused))
        reranked = await get_reranker().rerank(
            query=orig_query,
            documents=[c["text"] for c in fused],
            top_n=s.RETRIEVAL_TOP_K_FINAL,
        )
        logger.info("Reranking completed. Top chunk rerank score: %s", reranked[0][1] if reranked else "N/A")
        results: list[RetrievedChunk] = []
        for idx, relevance in reranked:
            c = fused[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    metadata=c["metadata"],
                    score=relevance,
                    cosine_similarity=c.get("cosine_similarity", 0.0),
                )
            )
        return results


def reciprocal_rank_fusion(
    *sources: list[dict[str, Any]], k: int = 60
) -> list[dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for source in sources:
        for rank, result in enumerate(source):
            cid = result["chunk_id"]
            entry = scores.setdefault(cid, {"score": 0.0, "data": result})
            entry["score"] += 1.0 / (k + rank + 1)
            # keep the richer (dense) record if we have a cosine value
            if result.get("cosine_similarity"):
                entry["data"] = result
    ordered = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [e["data"] for e in ordered]


_ENGINE: RetrievalEngine | None = None


def get_retrieval_engine() -> RetrievalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RetrievalEngine()
    return _ENGINE


async def initialize_retrieval_engine() -> None:
    get_retrieval_engine().build_bm25()
