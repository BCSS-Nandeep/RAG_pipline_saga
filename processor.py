"""
processor.py — STAGES 1 & 2
  Stage 1: Memory-safe streaming reader for MongoDB documents.
           Supports batch-based pagination with _id cursor for safe resumability.
  Stage 2: JSON → natural-language text converter.
"""

import logging
from datetime import datetime
from typing import Generator, Dict, Any, Optional, List

from bson import ObjectId

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Streaming Processor
# ---------------------------------------------------------------------------

class PostgresStreamProcessor:
    """Cursor-based PostgreSQL reader with batch pagination."""

    def __init__(
        self,
        collection_name: str,
        batch_size: int = 500,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Lazy import to avoid circular dependencies
        from source_store import SourceStore
        self._store = SourceStore()

    # -- public API ----------------------------------------------------------

    def count_documents(self, query: Optional[dict] = None) -> int:
        """Count documents in the PostgreSQL collection."""
        return self._store.count_documents(self.collection_name)

    def fetch_batch(
        self,
        batch_number: int,
        after_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a single batch of documents using ID cursor pagination.

        Args:
            batch_number: for logging only.
            after_id: string ID — fetch docs with id > this value.
            since: only fetch docs with created_at > this timestamp.

        Returns:
            List of documents (up to self.batch_size).
        """
        docs = self._store.fetch_batch(
            collection_name=self.collection_name,
            limit=self.batch_size,
            after_id=after_id,
            since=since
        )
        
        logger.debug("Batch %d: fetched %d docs (after_id=%s)", batch_number, len(docs), after_id)
        return docs

    def stream_documents(
        self,
        skip_ids: Optional[set] = None,
        since: Optional[datetime] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Legacy streaming interface — yields all documents one by one."""
        skip_ids = skip_ids or set()
        
        after_id = None
        processed = 0
        batch_number = 1
        
        while True:
            docs = self.fetch_batch(batch_number, after_id=after_id, since=since)
            if not docs:
                break
                
            for doc in docs:
                doc_id = str(doc.get("_id", ""))
                if skip_ids and doc_id in skip_ids:
                    continue
                    
                processed += 1
                if processed % 1000 == 0:
                    logger.info("Progress: %d documents streamed", processed)
                yield doc
                
            after_id = str(docs[-1].get("_id", ""))
            batch_number += 1

        logger.info("Stream finished — %d yielded", processed)


# ---------------------------------------------------------------------------
# Stage 2 — JSON → Text Converter
# ---------------------------------------------------------------------------

class DocumentConverter:
    """Flatten a MongoDB document into a readable text block."""

    # Fields to always skip (internal / binary / large)
    SKIP_FIELDS = {"__v", "password", "passwordHash", "salt", "refreshToken"}

    @classmethod
    def convert(cls, doc: Dict[str, Any]) -> str:
        """Return a ``Field: Value`` text representation of *doc*."""
        lines = cls._flatten(doc, prefix="")
        return "\n".join(lines)

    # -- internal ------------------------------------------------------------

    @classmethod
    def _flatten(cls, obj: Any, prefix: str) -> list[str]:
        lines: list[str] = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in cls.SKIP_FIELDS:
                    continue
                full_key = f"{prefix}.{key}" if prefix else key
                lines.extend(cls._flatten(value, full_key))

        elif isinstance(obj, (list, tuple)):
            if not obj:
                return lines
            # Short primitive lists → inline
            if all(isinstance(v, (str, int, float, bool)) for v in obj):
                label = cls._humanize(prefix)
                joined = ", ".join(str(v) for v in obj)
                lines.append(f"{label}: {joined}")
            else:
                for idx, item in enumerate(obj):
                    lines.extend(cls._flatten(item, f"{prefix}[{idx}]"))

        else:
            value = cls._serialize_value(obj)
            if value not in (None, "", "None", "null"):
                label = cls._humanize(prefix)
                lines.append(f"{label}: {value}")

        return lines

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Convert non-JSON-native types to strings."""
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            return "<binary data>"
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _humanize(dotted_key: str) -> str:
        """Turn ``user.address.city`` → ``User Address City``."""
        parts = dotted_key.replace("[", ".").replace("]", "").split(".")
        return " ".join(p.replace("_", " ").title() for p in parts if p)
