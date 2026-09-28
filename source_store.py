import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from db import get_pool

logger = logging.getLogger(__name__)

class SourceStore:
    """PostgreSQL repository for source documents, replacing PostgreSQLStreamProcessor."""

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
        """Count documents, natively querying the table named collection_name."""
        if not collection_name or not collection_name.isidentifier():
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists first to avoid crashing
                cur.execute("SELECT to_regclass(%s)", (collection_name,))
                if not cur.fetchone()[0]:
                    return 0
                cur.execute(f"SELECT count(*) FROM {collection_name}")
                return cur.fetchone()[0]

    def fetch_batch(
        self,
        collection_name: str,
        limit: int = 500,
        after_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch documents natively from the table, ordering by id."""
        if not collection_name.isidentifier():
            return []
            
        sql = f"SELECT * FROM {collection_name} WHERE 1=1"
        params = []
        
        if since:
            # We assume a standard 'created_at' column exists for 'since' filtering
            sql += " AND created_at > %s"
            params.append(since)
            
        if after_id:
            sql += " AND id > %s"
            params.append(after_id)
            
        sql += " ORDER BY id ASC LIMIT %s"
        params.append(limit)

        results = []
        import psycopg.rows
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(f"SELECT to_regclass(%s)", (collection_name,))
                if not cur.fetchone()['to_regclass']:
                    return []
                cur.execute(sql, params)
                for row in cur.fetchall():
                    # Map the native 'id' to '_id' for the pipeline to use
                    if 'id' in row:
                        row['_id'] = str(row['id'])
                    results.append(row)
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
