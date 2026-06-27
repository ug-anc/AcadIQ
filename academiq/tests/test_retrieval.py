"""Retrieval-layer tests (demo mode, isolated store via conftest)."""

import pytest

from app.services.retrieval import get_retrieval_engine, reciprocal_rank_fusion


@pytest.mark.asyncio
async def test_retrieve_returns_chunks_for_known_topic():
    engine = get_retrieval_engine()
    chunks = await engine.retrieve("minimum attendance requirement")
    assert chunks, "expected at least one chunk for an in-corpus query"
    # Every chunk must carry the full metadata contract.
    for c in chunks:
        for field in ("source_document", "section_number", "page_number", "chunk_id"):
            assert field in c.metadata


@pytest.mark.asyncio
async def test_retrieve_top_chunk_has_relevance_score():
    engine = get_retrieval_engine()
    chunks = await engine.retrieve("branch change CGPA")
    assert chunks
    assert 0.0 <= chunks[0].score <= 1.0


def test_rrf_merges_and_orders_by_fused_score():
    dense = [
        {"chunk_id": "a", "text": "x", "metadata": {}, "cosine_similarity": 0.9},
        {"chunk_id": "b", "text": "y", "metadata": {}, "cosine_similarity": 0.8},
    ]
    sparse = [
        {"chunk_id": "b", "text": "y", "metadata": {}, "cosine_similarity": 0.0},
        {"chunk_id": "c", "text": "z", "metadata": {}, "cosine_similarity": 0.0},
    ]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [r["chunk_id"] for r in fused]
    # b appears in both lists, so it should fuse to the top.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
