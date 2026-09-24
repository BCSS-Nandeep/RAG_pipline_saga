"""
vector_store.py — STAGE 5
  Store embeddings in MongoDB and perform cosine-similarity search.
  Uses a local numpy cache for fast search — avoids pulling 73K+ vectors
  over the network every query.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".vector_cache")

# ---------------------------------------------------------------------------
# Cache safety limits
# ---------------------------------------------------------------------------

# Collections with more chunks than this are served exclusively via server-side
# search (text pre-filter → $sample → brute-force).  Their vectors are never
# accumulated in process RAM as a local numpy cache.  Override via the env var.
CACHE_SIZE_LIMIT = int(os.getenv("CACHE_SIZE_LIMIT", "500000"))

# Dimension of each stored embedding vector; must match the embedding model.
_EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# ---------------------------------------------------------------------------
# Per-collection build locks — prevents two threads from building the same
# cache file simultaneously (e.g. two queries arriving after an invalidation).
# ---------------------------------------------------------------------------
_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_BUILD_LOCKS_MUTEX = threading.Lock()


def _get_build_lock(collection_name: str) -> threading.Lock:
    """Return (creating if absent) the per-collection cache-build lock."""
    with _BUILD_LOCKS_MUTEX:
        if collection_name not in _BUILD_LOCKS:
            _BUILD_LOCKS[collection_name] = threading.Lock()
        return _BUILD_LOCKS[collection_name]


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
        # None = not yet checked; True = exceeds CACHE_SIZE_LIMIT; False = within limit.
        self._oversized: Optional[bool] = None


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

        Collections that exceed CACHE_SIZE_LIMIT always use server-side search;
        their vectors are never loaded into process RAM as a local numpy cache.
        """
        # Size guard: oversized collections always use server-side search.
        # This is evaluated once per VectorStore instance and cached; subsequent
        # calls are a single bool comparison with no network round-trip.
        if self._is_oversized():
            logger.debug(
                "Collection '%s' exceeds CACHE_SIZE_LIMIT — using server-side search.",
                self.collection_name,
            )
            return self._search_server_side(
                query_vector, top_k, query_text=query_text,
                source_collection=source_collection,
            )

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

    def _is_oversized(self) -> bool:
        """Return True if this collection exceeds CACHE_SIZE_LIMIT.

        The result is cached on the instance after the first MongoDB call so
        every subsequent query is a pure in-process bool check.
        ``invalidate_cache()`` resets this flag so a collection that shrinks
        (or a new CACHE_SIZE_LIMIT setting) takes effect on the next access.
        """
        if self._oversized is None:
            try:
                col = self.connect()
                n = col.estimated_document_count()
                self._oversized = n > CACHE_SIZE_LIMIT
                if self._oversized:
                    logger.warning(
                        "Collection '%s' has %d chunks which exceeds "
                        "CACHE_SIZE_LIMIT=%d.  Local numpy cache is disabled for "
                        "this collection; server-side search will be used instead.",
                        self.collection_name, n, CACHE_SIZE_LIMIT,
                    )
            except Exception as exc:
                logger.warning(
                    "Size check failed for '%s': %s — treating as within limit.",
                    self.collection_name, exc,
                )
                self._oversized = False
        return bool(self._oversized)

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
        """Load the local cache into memory, building it from MongoDB if needed.

        If the collection exceeds CACHE_SIZE_LIMIT this method returns without
        building anything — cosine_search already routed to server-side search
        before calling here, so this guard is defence-in-depth only.
        """
        if self._cache_loaded:
            return

        # Defence-in-depth size guard (primary check is in cosine_search).
        if self._is_oversized():
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
        """Download all vectors from MongoDB page-by-page and save to disk.

        **Memory contract**: at most one PAGE_SIZE-document batch of raw floats
        lives in Python memory during the download loop.  The full embedding
        matrix is materialised only once — when reading the temporary raw-binary
        file back into numpy — and is bounded by CACHE_SIZE_LIMIT because
        collections larger than that never reach this method.

        **Atomic guarantee**: the caller's ``emb_path`` and ``meta_path`` files
        are replaced only after both temporary files have been fully written and
        verified.  A crash or exception at any point leaves the previously-valid
        cache files untouched.

        **Concurrency**: only one build per collection name runs at a time.
        A second caller finds the lock taken and returns immediately; when the
        first caller finishes the cache is on disk and will be loaded normally.
        """
        lock = _get_build_lock(self.collection_name)
        if not lock.acquire(blocking=False):
            logger.info(
                "Cache build for '%s' already in progress in another thread — skipping.",
                self.collection_name,
            )
            return

        # Temp file paths — all start with the collection name so
        # invalidate_cache() cleans them up automatically.
        tmp_emb  = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.tmp.npy")
        tmp_raw  = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.raw.tmp")
        tmp_meta = os.path.join(CACHE_DIR, f"{self.collection_name}_meta.tmp.json")

        try:
            col = self.connect()
            total = col.estimated_document_count()
            logger.info("Downloading %d chunks from MongoDB (paginated)...", total)

            # Remove any leftover temp files from a previous failed build.
            for tmp in (tmp_emb, tmp_raw, tmp_meta):
                if os.path.exists(tmp):
                    os.remove(tmp)

            texts: List[str] = []
            metas: List[Dict] = []
            count   = 0
            last_id = None
            PAGE_SIZE = 500

            # Stream embedding bytes page-by-page into a raw binary temp file.
            # Each vector is normalised here so the final np.fromfile step
            # produces a ready-to-use unit-norm matrix without a second pass.
            with open(tmp_raw, "wb") as raw_f:
                while True:
                    query = {"_id": {"$gt": last_id}} if last_id is not None else {}
                    batch = list(
                        col.find(query, {"text": 1, "embedding": 1, "metadata": 1})
                        .sort("_id", ASCENDING)
                        .limit(PAGE_SIZE)
                    )
                    if not batch:
                        break

                    for doc in batch:
                        emb = doc.get("embedding")
                        if not emb:
                            continue
                        vec = np.array(emb, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0.0:
                            vec /= norm
                        raw_f.write(vec.tobytes())  # _EMBED_DIM × 4 bytes per vector
                        texts.append(doc.get("text", ""))
                        metas.append(doc.get("metadata", {}))
                        count += 1

                    last_id = batch[-1]["_id"]
                    if count % 5000 == 0 or count == len(batch):
                        logger.info("  Downloaded %d / %d chunks...", count, total)

            logger.info("Download complete — %d chunks.", count)

            if count == 0:
                logger.warning("No embeddings found in MongoDB — cache not created.")
                for tmp in (tmp_emb, tmp_raw, tmp_meta):
                    if os.path.exists(tmp):
                        os.remove(tmp)
                self._cache_loaded = True
                return

            # Materialise the full embedding matrix from the raw file.
            # This is the only moment the entire matrix occupies RAM, and it is
            # bounded: _ensure_cache() refuses to call this method for collections
            # that exceed CACHE_SIZE_LIMIT.
            mat = np.fromfile(tmp_raw, dtype=np.float32).reshape(count, _EMBED_DIM)
            os.remove(tmp_raw)

            np.save(tmp_emb, mat)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump({"texts": texts, "metas": metas}, f)

            # Atomic swap: replace the final files only after both temp files
            # are fully written.  os.replace() is atomic on POSIX.
            os.replace(tmp_emb, emb_path)
            os.replace(tmp_meta, meta_path)

            self._embeddings = mat
            self._texts      = texts
            self._metas      = metas
            self._cache_loaded = True
            logger.info("Cache built and saved — %d vectors (%s).", count, emb_path)

        except Exception:
            logger.exception(
                "Cache build failed for '%s'; cleaning up temp files.",
                self.collection_name,
            )
            for tmp in (tmp_emb, tmp_raw, tmp_meta):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            raise
        finally:
            lock.release()

    def invalidate_cache(self):
        """Delete the local cache so it gets rebuilt on next search."""
        self._cache_loaded = False
        self._oversized    = None   # re-evaluate on next access
        self._embeddings   = None
        self._texts        = []
        self._metas        = []
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
