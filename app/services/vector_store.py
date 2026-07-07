"""Vector store wrapper around a deployed Neon Tech PostgreSQL database using pgvector."""

from __future__ import annotations

import json
from typing import Any
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

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
        # Connect to your remote Neon instance
        self.conn_string = s.DATABASE_URL
        
        # Self-initialize vector extension configuration for the client connection
        with psycopg.connect(self.conn_string) as conn:
            register_vector(conn)

    def _get_connection(self):
        """Returns a standard connection synced with pgvector registry."""
        conn = psycopg.connect(self.conn_string, row_factory=dict_row)
        register_vector(conn)
        return conn

    # ---- writes -----------------------------------------------------------
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
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_chunks (chunk_id, embedding, text, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, metadata = EXCLUDED.metadata;
                    """,
                    (chunk_id, embedding, text, json.dumps(metadata)),
                )
            conn.commit()

    def upsert_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
            
        for r in records:
            missing = REQUIRED_METADATA_FIELDS - set(r["metadata"].keys())
            if missing:
                raise ValueError(f"Missing metadata fields: {sorted(missing)}")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Process structural updates cleanly using a batch statement loop
                for r in records:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (chunk_id, embedding, text, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE 
                        SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, metadata = EXCLUDED.metadata;
                        """,
                        (r["chunk_id"], r["embedding"], r["text"], json.dumps(r["metadata"])),
                    )
            conn.commit()

    # ---- reads ------------------------------------------------------------
    def query(self, query_embedding: list[float], n_results: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Postgres pgvector uses <=> operator for Cosine Distance. 
                # Cosine Similarity = 1.0 - Cosine Distance
                cur.execute(
                    """
                    SELECT chunk_id, text, metadata, (1.0 - (embedding <=> %s::vector)) AS cosine_similarity
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, n_results),
                )
                rows = cur.fetchall()
                
                for row in rows:
                    out.append({
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                        "cosine_similarity": max(0.0, float(row["cosine_similarity"])),
                    })
        return out

    def all_documents(self) -> list[dict[str, Any]]:
        """Return every chunk (text + metadata) — used to build the BM25 index."""
        out: list[dict[str, Any]] = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, text, metadata FROM document_chunks;")
                rows = cur.fetchall()
                for row in rows:
                    out.append({
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                    })
        return out

    def count(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                res = cur.fetchone()
                return res["count"] if res else 0

    def reset(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE document_chunks;")
            conn.commit()


_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE