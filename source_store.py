import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from db import get_pool

logger = logging.getLogger(__name__)

class SourceStore:
    """PostgreSQL repository for source documents, replacing MongoStreamProcessor."""

    def __init__(self):
        self._pool = get_pool()

    def upsert_documents(self, documents: List[Dict[str, Any]], collection_name: str) -> int:
        """Batch upsert a list of source documents into PostgreSQL."""
        if not documents:
            return 0

        written = 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO source_documents (
                        id,
                        collection_name,
                        created_at,
                        document_data
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        collection_name = EXCLUDED.collection_name,
                        created_at = EXCLUDED.created_at,
                        document_data = EXCLUDED.document_data;
                """
                params = []
                for doc in documents:
                    doc_id = str(doc.get("_id"))
                    created_at = doc.get("created_at")
                    
                    # Convert to standard format
                    if isinstance(created_at, str):
                        try:
                            # Handle ISO strings if they're passed instead of datetime objects
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        except ValueError:
                            pass
                            
                    doc_data = json.dumps(doc, default=str)
                    params.append((doc_id, collection_name, created_at, doc_data))

                cur.executemany(sql, params)
                written = cur.rowcount if cur.rowcount >= 0 else len(documents)
            conn.commit()

        return written

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_data FROM source_documents WHERE id = %s",
                    (doc_id,)
                )
                res = cur.fetchone()
                if res:
                    return res[0] if isinstance(res[0], dict) else json.loads(res[0])
        return None

    def count_documents(self, collection_name: Optional[str] = None) -> int:
        """Count documents, optionally filtered by collection."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if collection_name:
                    cur.execute(
                        "SELECT count(*) FROM source_documents WHERE collection_name = %s",
                        (collection_name,)
                    )
                else:
                    cur.execute("SELECT count(*) FROM source_documents")
                return cur.fetchone()[0]

    def fetch_batch(
        self,
        collection_name: str,
        limit: int = 500,
        after_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch documents for incremental ingestion with ordering by ID.
        
        This perfectly mirrors MongoStreamProcessor's sorting and pagination capabilities.
        """
        sql = "SELECT document_data FROM source_documents WHERE collection_name = %s"
        params = [collection_name]
        
        if since:
            sql += " AND created_at > %s"
            params.append(since)
            
        if after_id:
            sql += " AND id > %s"
            params.append(after_id)
            
        sql += " ORDER BY id ASC LIMIT %s"
        params.append(limit)

        results = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    results.append(doc)
        return results
        
    def delete_test_documents(self, prefix: str = "pg_test_"):
        """Cleanup test documents."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM source_documents WHERE id LIKE %s",
                    (f"{prefix}%",)
                )
            conn.commit()
