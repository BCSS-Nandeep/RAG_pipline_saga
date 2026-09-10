"""
vector_store.py — STAGE 5
  Store embeddings in MongoDB and perform cosine-similarity search.
  Uses a local numpy cache for fast search — avoids pulling 73K+ vectors
  over the network every query.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".vector_cache")


class VectorStore:
    """Read/write 768-dim vectors in a MongoDB collection and search by cosine similarity."""

    def __init__(self, uri: str, db_name: str, collection_name: str = "vector_embeddings"):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client: Optional[MongoClient] = None
        self._collection: Optional[Collection] = None
        # In-memory cache
        self._cache_loaded = False
        self._embeddings: Optional[np.ndarray] = None  # (N, 768) normalized
        self._texts: List[str] = []
        self._metas: List[Dict] = []

    # -- connection ----------------------------------------------------------

    def connect(self) -> Collection:
        if self._collection is not None:
            return self._collection
        self._client = MongoClient(self.uri)
        db = self._client[self.db_name]
        self._collection = db[self.collection_name]
        self._ensure_indexes()
        return self._collection

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None

    def _ensure_indexes(self):
        """Create indexes for upsert deduplication and text search."""
        col = self._collection
        col.create_index(
            [("metadata.document_id", 1), ("metadata.chunk_index", 1)],
            unique=True,
            name="upsert_dedup_idx",
            background=True,
        )
        # Text index on the chunk text field — enables the fast text pre-filter
        # search path in cosine_search (avoids the unreliable $function JS fallback).
        existing = {idx["name"] for idx in col.list_indexes()}
        if "text_search_idx" not in existing:
            try:
                col.create_index(
                    [("text", "text")],
                    name="text_search_idx",
                    background=True,
                )
                logger.info("Created text index on '%s.text'.", self.collection_name)
            except Exception as exc:
                logger.warning("Could not create text index (may already exist): %s", exc)
        logger.debug("Ensured indexes on (document_id, chunk_index) and text.")

    # -- write ---------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Bulk upsert a list of chunk dicts.

        Each dict must contain: ``text``, ``embedding``, ``metadata``
        (with ``document_id`` and ``chunk_index``).

        Returns the number of upserted/modified documents.
        """
        col = self.connect()
        ops = []
        now = datetime.now(timezone.utc)

        for chunk in chunks:
            meta = chunk["metadata"]
            filt = {
                "metadata.document_id": meta["document_id"],
                "metadata.chunk_index": meta["chunk_index"],
            }
            update = {
                "$set": {
                    "text": chunk["text"],
                    "embedding": chunk["embedding"],
                    "metadata": {
                        **meta,
                        "created_at": now,
                        "source_created_at": meta.get("source_created_at"),
                    },
                }
            }
            ops.append(UpdateOne(filt, update, upsert=True))

        if not ops:
            return 0

        result = col.bulk_write(ops, ordered=False)
        written = result.upserted_count + result.modified_count
        logger.debug("Upserted %d chunks.", written)
        return written

    # -- read / search -------------------------------------------------------

    def total_chunks(self) -> int:
        col = self.connect()
        return col.estimated_document_count()

    def get_embedded_doc_ids(self) -> set:
        """Return the set of document_id strings already stored."""
        col = self.connect()
        ids = col.distinct("metadata.document_id")
        return set(ids)

    def get_last_ingested_time(self) -> Optional[datetime]:
        """Return the latest created_at timestamp from stored chunks.
        Used for incremental ingestion — only process docs newer than this.
        """
        col = self.connect()
        result = col.find_one(
            {},
            {"metadata.source_created_at": 1},
            sort=[("metadata.source_created_at", -1)],
        )
        if result and result.get("metadata", {}).get("source_created_at"):
            return result["metadata"]["source_created_at"]
        return None

    def cosine_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        batch_size: int = 5000,
        query_text: str = "",
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search.

        If ``source_collection`` is provided, results are restricted to chunks
        whose ``metadata.source_collection`` matches it (so a query against
        "users" never returns tweet chunks, etc.).

        Uses local numpy cache if available, otherwise falls back to
        server-side MongoDB text pre-filter + local cosine re-rank.
        """
        # Fast path: use numpy cache if already loaded or on disk
        if self._cache_loaded or self._cache_exists_on_disk():
            return self._search_with_cache(query_vector, top_k, source_collection=source_collection)

        # Slow path: text pre-filter + local cosine (fast) or sampled search (fallback)
        logger.info("No local cache — running server-side search...")
        return self._search_server_side(
            query_vector, top_k, query_text=query_text, source_collection=source_collection
        )

    def _cache_exists_on_disk(self) -> bool:
        emb_path = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.npy")
        meta_path = os.path.join(CACHE_DIR, f"{self.collection_name}_meta.json")
        return os.path.exists(emb_path) and os.path.exists(meta_path)

    def _search_with_cache(
        self,
        query_vector: List[float],
        top_k: int,
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search using the local numpy cache."""
        self._ensure_cache()

        if self._embeddings is None or len(self._texts) == 0:
            logger.warning("Vector cache is empty — no chunks to search.")
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self._embeddings @ q

        if source_collection:
            mask = np.array(
                [m.get("source_collection") == source_collection for m in self._metas],
                dtype=bool,
            )
            if not mask.any():
                logger.warning("Cache has no chunks for source_collection=%s", source_collection)
                return []
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if not np.isfinite(scores[idx]):
                break
            results.append({
                "text": self._texts[idx],
                "metadata": self._metas[idx],
                "score": float(scores[idx]),
            })
        return results

    def _search_server_side(
        self,
        query_vector: List[float],
        top_k: int,
        query_text: str = "",
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search using server-side computation.

        Strategy 1 (fast ~2-5s): MongoDB text index pre-filter → local cosine re-rank.
        Strategy 2 (fallback ~5-10s): $sample random subset → $function JS dot product.
        Strategy 3 (reliable fallback): Fetch recent chunks from MongoDB → local cosine.
        """
        col = self.connect()

        # Normalize query vector
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_normalized = q / q_norm

        # Strategy 1: text pre-filter (if query text available and text index exists)
        if query_text.strip():
            try:
                results = self._text_prefilter_search(
                    col, q_normalized, top_k, query_text, source_collection=source_collection
                )
                if results:
                    return results
            except Exception as exc:
                logger.warning("Text pre-filter search failed: %s — falling back to sampled", exc)

        # Strategy 2: $sample + $function
        logger.info("Using sampled search (2K random chunks)...")
        results = self._sampled_search(col, q_normalized.tolist(), top_k, source_collection=source_collection)
        if results:
            return results

        # Strategy 3: brute-force fetch + local cosine (reliable fallback)
        logger.info("Sampled search returned nothing — fetching chunks for local cosine...")
        return self._brute_force_search(col, q_normalized, top_k, source_collection=source_collection)

    def _text_prefilter_search(
        self,
        col: Collection,
        q_normalized: np.ndarray,
        top_k: int,
        query_text: str,
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Use MongoDB text index to find candidate chunks, then re-rank by cosine.

        ~2-5s: text search returns 500 candidates, then local numpy cosine on 500 vectors.
        """
        CANDIDATE_LIMIT = 500

        mongo_filter: Dict[str, Any] = {"$text": {"$search": query_text}}
        if source_collection:
            mongo_filter["metadata.source_collection"] = source_collection

        candidates = list(
            col.find(
                mongo_filter,
                {"text": 1, "embedding": 1, "metadata": 1, "score": {"$meta": "textScore"}},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(CANDIDATE_LIMIT)
        )

        if not candidates:
            logger.info("Text search returned 0 candidates for: %s", query_text[:100])
            return []

        logger.info("Text pre-filter found %d candidates — computing cosine locally", len(candidates))

        # Local cosine re-rank on just the candidates
        embeds = np.array([c["embedding"] for c in candidates], dtype=np.float32)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeds = embeds / norms

        scores = embeds @ q_normalized
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": candidates[idx].get("text", ""),
                "metadata": candidates[idx].get("metadata", {}),
                "score": float(scores[idx]),
            })
        return results

    def _sampled_search(
        self,
        col: Collection,
        q_list: List[float],
        top_k: int,
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """$sample random docs from MongoDB, compute cosine similarity via $function."""
        SAMPLE_SIZE = 2000

        js_body = (
            "function(emb, qv) {"
            "  let dot = 0, norm = 0;"
            "  for (let i = 0; i < emb.length; i++) {"
            "    dot += emb[i] * qv[i];"
            "    norm += emb[i] * emb[i];"
            "  }"
            "  return norm > 0 ? dot / Math.sqrt(norm) : 0;"
            "}"
        )

        pipeline: List[Dict[str, Any]] = []
        if source_collection:
            pipeline.append({"$match": {"metadata.source_collection": source_collection}})
        pipeline.extend([
            {"$sample": {"size": SAMPLE_SIZE}},
            {
                "$addFields": {
                    "score": {
                        "$function": {
                            "body": js_body,
                            "args": ["$embedding", q_list],
                            "lang": "js",
                        }
                    }
                }
            },
            {"$sort": {"score": -1}},
            {"$limit": top_k},
            {"$project": {"text": 1, "metadata": 1, "score": 1, "_id": 0}},
        ])

        results = []
        try:
            for doc in col.aggregate(pipeline, allowDiskUse=True, maxTimeMS=120000):
                results.append({
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0),
                })
        except Exception as exc:
            logger.error("Sampled search failed: %s", exc)
        return results

    def _brute_force_search(
        self,
        col: Collection,
        q_normalized: np.ndarray,
        top_k: int,
        source_collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks from MongoDB and compute cosine similarity locally.

        This is the most reliable fallback — no text index or $function JS needed.
        Fetches up to 2000 recent chunks, computes cosine in numpy.
        """
        FETCH_LIMIT = 2000
        mongo_filter: Dict[str, Any] = {}
        if source_collection:
            mongo_filter["metadata.source_collection"] = source_collection

        try:
            candidates = list(
                col.find(mongo_filter, {"text": 1, "embedding": 1, "metadata": 1})
                .sort("_id", -1)
                .limit(FETCH_LIMIT)
            )
        except Exception as exc:
            logger.error("Brute-force fetch failed: %s", exc)
            return []

        if not candidates:
            logger.info("No chunks found in collection for brute-force search.")
            return []

        logger.info("Brute-force: fetched %d chunks — computing cosine locally", len(candidates))

        embeds = np.array([c["embedding"] for c in candidates], dtype=np.float32)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeds = embeds / norms

        scores = embeds @ q_normalized
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": candidates[idx].get("text", ""),
                "metadata": candidates[idx].get("metadata", {}),
                "score": float(scores[idx]),
            })
        return results

    # -- cache management ----------------------------------------------------

    def _ensure_cache(self):
        """Load the local cache into memory, building it from MongoDB if needed."""
        if self._cache_loaded:
            return

        os.makedirs(CACHE_DIR, exist_ok=True)
        emb_path = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.npy")
        meta_path = os.path.join(CACHE_DIR, f"{self.collection_name}_meta.json")

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            logger.info("Loading vector cache from disk...")
            self._embeddings = np.load(emb_path)
            with open(meta_path, "r") as f:
                data = json.load(f)
            self._texts = data["texts"]
            self._metas = data["metas"]
            self._cache_loaded = True
            logger.info("Cache loaded — %d vectors ready.", len(self._texts))
            return

        # Build cache from MongoDB
        logger.info("Building vector cache from MongoDB (one-time download)...")
        self._build_cache(emb_path, meta_path)

    def _build_cache(self, emb_path: str, meta_path: str):
        """Download all vectors from MongoDB in small paginated batches and save to disk."""
        col = self.connect()
        total = col.estimated_document_count()
        logger.info("Downloading %d chunks from MongoDB (paginated)...", total)

        texts = []
        embeds = []
        metas = []

        # Use _id cursor pagination to avoid long-lived cursors timing out
        PAGE_SIZE = 500
        last_id = None
        count = 0

        while True:
            query = {"_id": {"$gt": last_id}} if last_id else {}
            batch = list(
                col.find(query, {"text": 1, "embedding": 1, "metadata": 1})
                .sort("_id", 1)
                .limit(PAGE_SIZE)
            )
            if not batch:
                break

            for doc in batch:
                texts.append(doc.get("text", ""))
                embeds.append(doc["embedding"])
                metas.append(doc.get("metadata", {}))

            last_id = batch[-1]["_id"]
            count += len(batch)
            if count % 5000 == 0 or count == len(batch):
                logger.info("  Downloaded %d / %d chunks...", count, total)

        logger.info("Download complete — %d chunks.", count)

        if not embeds:
            logger.warning("No embeddings found in MongoDB.")
            self._cache_loaded = True
            return

        # Normalize and save
        mat = np.array(embeds, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        np.save(emb_path, mat)
        with open(meta_path, "w") as f:
            json.dump({"texts": texts, "metas": metas}, f)

        self._embeddings = mat
        self._texts = texts
        self._metas = metas
        self._cache_loaded = True
        logger.info("Cache built and saved — %d vectors (%s).", len(texts), emb_path)

    def invalidate_cache(self):
        """Delete the local cache so it gets rebuilt on next search."""
        self._cache_loaded = False
        self._embeddings = None
        self._texts = []
        self._metas = []
        for f in os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else []:
            if f.startswith(self.collection_name):
                os.remove(os.path.join(CACHE_DIR, f))
                logger.info("Deleted cache file: %s", f)

    def refresh_cache(self):
        """Force rebuild the cache from MongoDB."""
        self.invalidate_cache()
        self._ensure_cache()


# ---------------------------------------------------------------------------
# OPTIONAL — Atlas Vector Search index creation (run once in Atlas UI or shell)
# ---------------------------------------------------------------------------
#
# If you are using MongoDB Atlas and want hardware-accelerated ANN search
# instead of the brute-force numpy approach above, create a Search index on
# the ``vector_embeddings`` collection:
#
#   {
#     "mappings": {
#       "dynamic": true,
#       "fields": {
#         "embedding": {
#           "type": "knnVector",
#           "dimensions": 768,
#           "similarity": "cosine"
#         }
#       }
#     }
#   }
#
# Then replace ``cosine_search`` with an aggregation pipeline:
#
#   pipeline = [
#       {
#           "$vectorSearch": {
#               "index": "vector_index",
#               "path": "embedding",
#               "queryVector": query_vector,
#               "numCandidates": 100,
#               "limit": top_k,
#           }
#       },
#       {
#           "$project": {
#               "text": 1,
#               "metadata": 1,
#               "score": {"$meta": "vectorSearchScore"},
#           }
#       },
#   ]
#   results = list(collection.aggregate(pipeline))
# ---------------------------------------------------------------------------
