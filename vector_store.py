"""
vector_store.py — STAGE 5 (PostgreSQL Native Edition)
  Store embeddings in PostgreSQL and perform cosine-similarity search natively.
  Replaces the old PostgreSQL + NumPy cache implementation.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from db import get_pool

logger = logging.getLogger(__name__)

class VectorStore:
    """Read/write 768-dim vectors in a PostgreSQL table and search by cosine similarity."""

    def __init__(self):
        # PostgreSQL connection is managed via db.py
        self._pool = get_pool()

    # -- connection ----------------------------------------------------------

    def connect(self):
        """No-op for compatibility. db.py handles pooling natively."""
        pass

    def close(self):
        """No-op for compatibility. db.py handles connection pool closure."""
        pass

    # -- write ---------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Bulk upsert a list of chunk dicts.

        Each dict must contain: ``text``, ``embedding``, ``metadata``
        (with ``document_id`` and ``chunk_index``).

        Returns the number of upserted/modified documents.
        """
        if not chunks:
            return 0

        written = 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Use executemany for efficient batch upsert
                sql = """
                    INSERT INTO vector_embeddings (
                        document_id,
                        chunk_index,
                        text,
                        embedding_vector,
                        metadata,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index)
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding_vector = EXCLUDED.embedding_vector,
                        metadata = EXCLUDED.metadata,
                        created_at = EXCLUDED.created_at;
                """
                params = []
                for chunk in chunks:
                    meta = chunk["metadata"]
                    emb = chunk["embedding"]
                    if len(emb) != 768:
                        raise ValueError(f"Embedding must be 768 dimensions, got {len(emb)}")

                    doc_id = meta["document_id"]
                    chunk_idx = meta["chunk_index"]
                    text = chunk["text"]

                    now = datetime.now(timezone.utc)
                    created_at = meta.get("created_at", now)

                    # Ensure source_created_at is ISO string format for JSONB compatibility
                    if "source_created_at" in meta and isinstance(meta["source_created_at"], datetime):
                        meta["source_created_at"] = meta["source_created_at"].isoformat()

                    meta_json = json.dumps(meta, default=str)

                    params.append((doc_id, chunk_idx, text, emb, meta_json, created_at))

                cur.executemany(sql, params)
                written = cur.rowcount if cur.rowcount >= 0 else len(chunks)
            conn.commit()

        logger.debug("Upserted %d chunks into PostgreSQL.", len(chunks))
        return len(chunks)

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a given document_id.

        Used during re-embedding when a document's content has changed, to
        remove stale chunks before inserting the new ones.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM vector_embeddings WHERE document_id = %s",
                    (document_id,)
                )
                deleted = cur.rowcount
            conn.commit()
        logger.debug("Deleted %d chunks for document_id=%s", deleted, document_id)
        return deleted

    # -- read / search -------------------------------------------------------

    def total_chunks(self) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM vector_embeddings;")
                return cur.fetchone()[0]

    def get_embedded_doc_ids(self) -> set:
        """Return the set of document_id strings already stored."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT document_id FROM vector_embeddings;")
                return {row[0] for row in cur.fetchall()}

    def get_last_ingested_time(self) -> Optional[datetime]:
        """Return the latest created_at timestamp from stored chunks.
        Used for incremental ingestion — only process docs newer than this.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT metadata->>'source_created_at'
                    FROM vector_embeddings
                    WHERE metadata->>'source_created_at' IS NOT NULL
                    ORDER BY metadata->>'source_created_at' DESC
                    LIMIT 1;
                """)
                res = cur.fetchone()
                if res and res[0]:
                    try:
                        return datetime.fromisoformat(res[0].replace('Z', '+00:00'))
                    except ValueError:
                        return None
        return None

    def cosine_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        batch_size: int = 5000,
        query_text: str = "",
        source_collection: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        cutoff_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search.

        If ``source_collection`` is provided, results are restricted to chunks
        whose ``metadata.source_collection`` matches it.
        If ``doc_ids`` is provided, restricts to specific document_ids.
        """
        if not query_vector:
            return []

        if len(query_vector) != 768:
            raise ValueError(f"Query vector must be 768 dimensions, got {len(query_vector)}")

        query_norm = math.sqrt(sum(x * x for x in query_vector))
        if query_norm == 0:
            return []

        # Note: pgvector distances: 1 - cosine_distance = cosine_similarity
        sql = """
            SELECT
                text,
                metadata,
                1 - (embedding_vector <=> %s::vector) AS score
            FROM vector_embeddings
        """
        params = [query_vector]

        where_clauses = []
        if source_collection:
            where_clauses.append("metadata->>'source_collection' = %s")
            params.append(source_collection)

        if doc_ids:
            where_clauses.append("document_id = ANY(%s)")
            params.append(doc_ids)

        if cutoff_date:
            where_clauses.append("(metadata->>'source_created_at') IS NOT NULL AND (metadata->>'source_created_at')::timestamp >= %s::timestamp")
            params.append(cutoff_date)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY embedding_vector <=> %s::vector LIMIT %s;"
        params.append(query_vector)
        params.append(top_k)

        results = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    text, meta, score = row
                    results.append({
                        "text": text,
                        "metadata": meta if isinstance(meta, dict) else json.loads(meta),
                        "score": float(score)
                    })
        return results

    # -- cache management ----------------------------------------------------

    def refresh_cache(self):
        """No-op for compatibility. PostgreSQL is the source of truth natively."""
        logger.info("refresh_cache called - NO-OP (PostgreSQL handles search directly)")

    def invalidate_cache(self):
        """No-op for compatibility."""
        logger.info("invalidate_cache called - NO-OP")
