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
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Streaming Processor
# ---------------------------------------------------------------------------

class MongoStreamProcessor:
    """Cursor-based MongoDB reader with batch pagination."""

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str,
        batch_size: int = 500,
    ):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_size = batch_size
        self._client: Optional[MongoClient] = None

    # -- connection helpers --------------------------------------------------

    def _connect(self) -> Collection:
        self._client = MongoClient(self.uri)
        db = self._client[self.db_name]
        return db[self.collection_name]

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    # -- public API ----------------------------------------------------------

    def count_documents(self, query: Optional[dict] = None) -> int:
        collection = self._connect()
        if query:
            count = collection.count_documents(query)
        else:
            count = collection.estimated_document_count()
        self.close()
        return count

    def fetch_batch(
        self,
        batch_number: int,
        after_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a single batch of documents using _id cursor pagination.

        Args:
            batch_number: for logging only.
            after_id: ObjectId string — fetch docs with _id > this value.
            since: only fetch docs with created_at > this timestamp.

        Returns:
            List of documents (up to self.batch_size).
        """
        collection = self._connect()

        query: dict = {}
        if after_id:
            try:
                query["_id"] = {"$gt": ObjectId(after_id)}
            except Exception:
                pass
        if since:
            query["created_at"] = {"$gt": since}

        docs = list(
            collection.find(query)
            .sort("_id", ASCENDING)
            .limit(self.batch_size)
        )

        self.close()
        logger.debug("Batch %d: fetched %d docs (after_id=%s)", batch_number, len(docs), after_id)
        return docs

    def stream_documents(
        self,
        skip_ids: Optional[set] = None,
        since: Optional[datetime] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Legacy streaming interface — yields all documents one by one."""
        skip_ids = skip_ids or set()
        collection = self._connect()

        query = {}
        if since:
            query["created_at"] = {"$gt": since}
        elif skip_ids:
            oid_list = []
            for sid in skip_ids:
                try:
                    oid_list.append(ObjectId(sid))
                except Exception:
                    pass
            if oid_list:
                query = {"_id": {"$nin": oid_list}}

        cursor = collection.find(
            query,
            no_cursor_timeout=True,
            batch_size=self.batch_size,
        )

        processed = 0
        try:
            for doc in cursor:
                processed += 1
                if processed % 1000 == 0:
                    logger.info("Progress: %d documents streamed", processed)
                yield doc
        finally:
            cursor.close()
            self.close()
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
