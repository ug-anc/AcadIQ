"""Vector store wrapper around an embedded persistent ChromaDB client."""

from __future__ import annotations

from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

REQUIRED_METADATA_FIELDS = {
    "source_document",
    "section_path",
    "page_number",
    "chunk_type",
    "section_number",
    "extraction_date",
    "chunk_id",
}


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self._client = chromadb.PersistentClient(
            path=s.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=s.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # embeddings are computed externally
        )

    def upsert(
        self,
        chunk_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing:
            raise ValueError(f"Missing metadata fields: {sorted(missing)}")
        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def upsert_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        for r in records:
            missing = REQUIRED_METADATA_FIELDS - set(r["metadata"].keys())
            if missing:
                raise ValueError(f"Missing metadata fields: {sorted(missing)}")
        self._collection.upsert(
            ids=[r["chunk_id"] for r in records],
            embeddings=[r["embedding"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )

    def query(self, query_embedding: list[float], n_results: int) -> list[dict[str, Any]]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for i, chunk_id in enumerate(res["ids"][0]):
            distance = res["distances"][0][i]
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    # cosine distance -> similarity in [-1, 1], clamped to [0, 1]
                    "cosine_similarity": max(0.0, 1.0 - distance),
                }
            )
        return out

    def all_documents(self) -> list[dict[str, Any]]:
        """Return every chunk (text + metadata) — used to build the BM25 index."""
        res = self._collection.get(include=["documents", "metadatas"])
        out: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(res["ids"]):
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": res["documents"][i],
                    "metadata": res["metadatas"][i],
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        s = get_settings()
        self._client.delete_collection(s.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=s.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )


_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE