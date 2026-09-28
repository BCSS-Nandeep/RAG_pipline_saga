import os
import sqlite3
import threading
import json
import time
import gc
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
from pymongo import MongoClient, UpdateOne, ASCENDING, DeleteOne
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".vector_cache")
HOT_CACHE_MAX_ITEMS = int(os.getenv("HOT_CACHE_MAX_ITEMS", "500000"))
# Set a conservative limit of 800 MB for the hot cache in memory
HOT_CACHE_MAX_MEMORY_MB = float(os.getenv("HOT_CACHE_MAX_MEMORY_MB", "800.0"))
_EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
_INITIAL_CAPACITY = 10000

def _get_today() -> str:
    """Returns the current local calendar date string (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")

def _dt_to_date_str(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    return ""

class HotCacheManager:
    """Process-wide HotCacheManager controlling global item limits and memory."""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.lock = threading.RLock()
        self.global_item_count = 0
        self.global_memory_bytes = 0
        self.caches = {} # col_name -> dict

    def _calculate_bytes(self, capacity: int) -> int:
        return capacity * _EMBED_DIM * 4

    def release(self, collection_name: str):
        with self.lock:
            if collection_name in self.caches:
                c = self.caches.pop(collection_name)
                self.global_item_count -= c["count"]
                self.global_memory_bytes -= self._calculate_bytes(c["capacity"])
                del c["mat"]
                del c["texts"]
                del c["metas"]
                del c["chunk_ids"]
                logger.info("Evicted/Released cache for %s. Global items: %d, Global bytes: %d", 
                            collection_name, self.global_item_count, self.global_memory_bytes)
                gc.collect()

    def _evict_lru(self, required_bytes: int, required_items: int) -> bool:
        max_bytes = HOT_CACHE_MAX_MEMORY_MB * 1024 * 1024
        
        # Evict until budget allows
        while (self.global_memory_bytes + required_bytes > max_bytes) or \
              (self.global_item_count + required_items > HOT_CACHE_MAX_ITEMS):
            if not self.caches:
                break
            lru_col = min(self.caches.keys(), key=lambda k: self.caches[k]["last_access"])
            self.release(lru_col)

        # Check process RSS emergency guard
        rss = 0  # psutil removed as per instructions
        while rss + required_bytes > max_bytes:
            if not self.caches:
                break
            lru_col = min(self.caches.keys(), key=lambda k: self.caches[k]["last_access"])
            self.release(lru_col)
        rss = 0  # psutil removed as per instructions
            
        rss = 0  # psutil removed as per instructions
        if (self.global_memory_bytes + required_bytes > max_bytes) or \
           (self.global_item_count + required_items > HOT_CACHE_MAX_ITEMS) or \
           (rss + required_bytes > max_bytes):
            return False
        return True

    def admit_new(self, collection_name: str, count: int, capacity: int, today: str) -> bool:
        with self.lock:
            if collection_name in self.caches:
                self.release(collection_name)
                
            req_bytes = self._calculate_bytes(capacity)
            if not self._evict_lru(req_bytes, capacity):
                logger.warning("Cache admission denied for %s (req_items=%d, req_bytes=%d)", collection_name, capacity, req_bytes)
                return False
                
            self.global_memory_bytes += req_bytes
            self.caches[collection_name] = {
                "mat": np.empty((capacity, _EMBED_DIM), dtype=np.float32),
                "texts": [],
                "metas": [],
                "chunk_ids": {},
                "count": 0,
                "capacity": capacity,
                "overflow": False,
                "date": today,
                "last_access": time.time()
            }
            logger.info("Admitted cache for %s. Global items: %d, Global bytes: %d", 
                        collection_name, self.global_item_count, self.global_memory_bytes)
            return True

    def get_cache(self, collection_name: str):
        with self.lock:
            if collection_name in self.caches:
                self.caches[collection_name]["last_access"] = time.time()
                return self.caches[collection_name]
            return None

    def grow_capacity(self, collection_name: str, new_capacity: int) -> bool:
        with self.lock:
            if collection_name not in self.caches:
                return False
            c = self.caches[collection_name]
            if new_capacity <= c["capacity"]:
                return True
                
            delta_bytes = self._calculate_bytes(new_capacity) - self._calculate_bytes(c["capacity"])
            
            if not self._evict_lru(delta_bytes, 0):
                c["overflow"] = True
                return False
                
            self.global_memory_bytes += delta_bytes
            new_mat = np.empty((new_capacity, _EMBED_DIM), dtype=np.float32)
            if c["count"] > 0:
                new_mat[:c["count"]] = c["mat"][:c["count"]]
            c["mat"] = new_mat
            c["capacity"] = new_capacity
            c["last_access"] = time.time()
            return True

# Global cache manager instance
_cache_manager = HotCacheManager()

class VectorStore:
    def __init__(self, uri: str, db_name: str, collection_name: str = "vector_embeddings", cache_dir=CACHE_DIR):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.cache_dir = cache_dir
        
        self._client: Optional[MongoClient] = None
        self._collection: Optional[Collection] = None
        self._lock = threading.RLock()

    # -- connection ----------------------------------------------------------

    def connect(self) -> Collection:
        if self._collection is not None:
            return self._collection
        self._client = MongoClient(self.uri)
        db = self._client[self.db_name]
        self._collection = db[self.collection_name]
        return self._collection

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None

    def _check_rollover(self):
        """Lazy midnight rollover check."""
        with self._lock:
            today = _get_today()
            cache = _cache_manager.get_cache(self.collection_name)
            
            if cache:
                if cache["date"] == today:
                    return
                
                if cache["date"] and not cache["overflow"] and cache["count"] > 0:
                    # Finalize to historical disk cache
                    logger.info("Midnight Rollover: Finalizing day %s for %s", cache["date"], self.collection_name)
                    try:
                        mat_active = cache["mat"][:cache["count"]]
                        self._write_historical_partition(cache["date"], mat_active, cache["texts"], cache["metas"])
                    except Exception as e:
                        logger.error("Rollover failed for %s: %s. MongoDB fallback will be used.", cache["date"], e)
                
                _cache_manager.release(self.collection_name)
            
            # Rebuild today's cache from MongoDB
            self._rebuild_hot_cache()

    def _rebuild_hot_cache(self):
        """Rebuild today's cache from MongoDB safely."""
        with self._lock:
            col = self.connect()
            today = _get_today()
            
            start_of_day = datetime.strptime(today, "%Y-%m-%d")
            end_of_day = datetime(start_of_day.year, start_of_day.month, start_of_day.day, 23, 59, 59, 999999)
            
            query = {
                "metadata.source_created_at": {
                    "$gte": start_of_day,
                    "$lte": end_of_day
                }
            }
            
            count = col.count_documents(query)
            capacity = max(_INITIAL_CAPACITY, count)
            
            # Allocation admission BEFORE allocation
            if not _cache_manager.admit_new(self.collection_name, count, capacity, today):
                return
                
            cache = _cache_manager.get_cache(self.collection_name)
            if not cache:
                return
                
            cursor = col.find(query)
            idx = 0
            
            for doc in cursor:
                # Need to update global item count tracking
                if cache["overflow"]:
                    break
                    
                emb = doc.get("embedding")
                if not emb: continue
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0.0: vec /= norm
                
                cache["mat"][idx] = vec
                cache["texts"].append(doc.get("text", ""))
                meta = doc.get("metadata", {})
                cache["metas"].append(meta)
                cache["chunk_ids"][meta.get("chunk_id", "")] = idx
                idx += 1
                
            cache["count"] = idx
            with _cache_manager.lock:
                _cache_manager.global_item_count += idx # update actual
            logger.info("Hot cache rebuilt for %s: %d chunks.", today, idx)

    # -- api compatibility ---------------------------------------------------
    
    def total_chunks(self) -> int:
        col = self.connect()
        return col.estimated_document_count()

    def refresh_cache(self):
        pass

    def invalidate_cache(self):
        with self._lock:
            _cache_manager.release(self.collection_name)
            logger.info("Cache invalidated and released for %s", self.collection_name)

    # -- ingestion / dynamic updates -----------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
            
        col = self.connect()
        ops = []
        for ch in chunks:
            ops.append(
                UpdateOne(
                    {
                        "metadata.document_id": ch["metadata"]["document_id"],
                        "metadata.chunk_index": ch["metadata"]["chunk_index"],
                    },
                    {"$set": ch},
                    upsert=True,
                )
            )
            
        res = col.bulk_write(ops, ordered=False)
        stored = res.upserted_count + res.modified_count
        
        # Dynamically update Hot Cache
        with self._lock:
            self._check_rollover()
            
            cache = _cache_manager.get_cache(self.collection_name)
            if not cache or cache["overflow"]:
                return stored
                
            today = cache["date"]
            
            for ch in chunks:
                dt = ch.get("metadata", {}).get("source_created_at")
                if _dt_to_date_str(dt) == today:
                    chunk_id = ch["metadata"].get("chunk_id", "")
                    
                    emb = ch.get("embedding")
                    if not emb: continue
                    vec = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0.0: vec /= norm
                    
                    if chunk_id in cache["chunk_ids"]:
                        # Update existing in-place
                        idx = cache["chunk_ids"][chunk_id]
                        cache["mat"][idx] = vec
                        cache["texts"][idx] = ch.get("text", "")
                        cache["metas"][idx] = ch.get("metadata", {})
                    else:
                        # Append new
                        if cache["count"] >= cache["capacity"]:
                            if not _cache_manager.grow_capacity(self.collection_name, cache["capacity"] * 2):
                                break # overflow
                                
                        idx = cache["count"]
                        cache["mat"][idx] = vec
                        cache["texts"].append(ch.get("text", ""))
                        cache["metas"].append(ch.get("metadata", {}))
                        cache["chunk_ids"][chunk_id] = idx
                        cache["count"] += 1
                        with _cache_manager.lock:
                            _cache_manager.global_item_count += 1
            
        return stored

    def delete_by_document_id(self, document_id: str) -> int:
        col = self.connect()
        res = col.delete_many({"metadata.document_id": document_id})
        
        with self._lock:
            # We invalidate cache completely to let it rebuild cleanly, rather than compact array
            _cache_manager.release(self.collection_name)
            
        return res.deleted_count

    # -- historical disk persistence -----------------------------------------

    def _get_hist_paths(self, date_str: str) -> Tuple[str, str]:
        base = os.path.join(self.cache_dir, self.collection_name)
        os.makedirs(base, exist_ok=True)
        return (os.path.join(base, f"{date_str}.npy"),
                os.path.join(base, f"{date_str}.sqlite"))

    def _write_historical_partition(self, date_str: str, mat: np.ndarray, texts: List[str], metas: List[Dict]):
        if mat.shape[0] == 0:
            return
            
        emb_path, db_path = self._get_hist_paths(date_str)
        tmp_emb = emb_path + ".tmp"
        np.save(tmp_emb, mat)
        
        tmp_db = db_path + ".tmp"
        if os.path.exists(tmp_db): os.remove(tmp_db)
        
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute("CREATE TABLE metadata (idx INTEGER PRIMARY KEY, text TEXT, meta TEXT)")
        
        rows = []
        for i, (txt, m) in enumerate(zip(texts, metas)):
            rows.append((i, txt, json.dumps(m)))
            
        cur.executemany("INSERT INTO metadata VALUES (?, ?, ?)", rows)
        conn.commit()
        conn.close()
        
        os.replace(tmp_emb, emb_path)
        os.replace(tmp_db, db_path)
        logger.info("Wrote historical partition %s for %s (%d vectors)", date_str, self.collection_name, mat.shape[0])

    def _read_historical_metadata(self, date_str: str, candidate_ids: List[int]) -> Tuple[List[str], List[Dict]]:
        _, db_path = self._get_hist_paths(date_str)
        texts, metas = [], []
        if not os.path.exists(db_path):
            return texts, metas
            
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        placeholders = ",".join("?" * len(candidate_ids))
        cur.execute(f"SELECT text, meta FROM metadata WHERE idx IN ({placeholders})", candidate_ids)
        
        for txt, m_json in cur.fetchall():
            texts.append(txt)
            metas.append(json.loads(m_json))
            
        conn.close()
        return texts, metas

    def _get_all_historical_dates(self) -> List[str]:
        base = os.path.join(self.cache_dir, self.collection_name)
        if not os.path.exists(base): return []
        
        import re
        dates = set()
        for f in os.listdir(base):
            m = re.match(r"^(\d{4}-\d{2}-\d{2})\.(npy|sqlite)$", f)
            if m: dates.add(m.group(1))
        return sorted(list(dates), reverse=True)

    # -- query routing -------------------------------------------------------

    def cosine_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_text: Optional[str] = None,
        source_collection: Optional[str] = None,
        explicit_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0: q /= norm
        
        with self._lock:
            self._check_rollover()
            
        results = []
        date_from, date_to = None, None
        
        try:
            import intent
            if query_text:
                date_from, date_to = intent.extract_dates(query_text)
        except Exception:
            pass

        if explicit_date:
            date_from = date_to = explicit_date
            
        if date_from and date_to:
            from datetime import datetime, timedelta
            try:
                start_dt = datetime.strptime(date_from, "%Y-%m-%d")
                end_dt = datetime.strptime(date_to, "%Y-%m-%d")
                
                target_dates = []
                curr = start_dt
                while curr <= end_dt:
                    target_dates.append(curr.strftime("%Y-%m-%d"))
                    curr += timedelta(days=1)
                
                for d in target_dates:
                    cache = _cache_manager.get_cache(self.collection_name)
                    if cache and d == cache["date"] and not cache["overflow"]:
                        res = self._search_hot_cache(q, top_k)
                        results.extend(res)
                    else:
                        res = self._search_historical(q, top_k, d)
                        results.extend(res)
                
                if not results:
                    results = self._search_server_side(query_vector, top_k, query_text, source_collection, explicit_date=date_from, date_to=date_to)
                else:
                    seen = set()
                    deduped = []
                    for r in sorted(results, key=lambda x: x["score"], reverse=True):
                        cid = r["metadata"]["chunk_id"]
                        if cid not in seen:
                            seen.add(cid)
                            deduped.append(r)
                    results = deduped[:top_k]
                return results
                
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Date routing failed: %s", e)
                
        # NO-DATE behavior
        cache = _cache_manager.get_cache(self.collection_name)
        if cache and not cache["overflow"]:
            results = self._search_hot_cache(q, top_k)
            
        if len(results) < top_k:
            hist_dates = self._get_all_historical_dates()
            for d in hist_dates:
                if len(results) >= top_k: break
                hist_res = self._search_historical(q, top_k, d)
                results.extend(hist_res)
                
            seen = set()
            deduped = []
            for r in sorted(results, key=lambda x: x["score"], reverse=True):
                cid = r["metadata"]["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(r)
            results = deduped[:top_k]
            
        if len(results) < top_k:
            mongo_results = self._search_server_side(query_vector, top_k, query_text, source_collection)
            seen = {r["metadata"]["chunk_id"] for r in results}
            for mr in mongo_results:
                if mr["metadata"]["chunk_id"] not in seen:
                    results.append(mr)
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            
        return results

    def _search_hot_cache(self, q: np.ndarray, top_k: int) -> List[Dict]:
        with self._lock:
            cache = _cache_manager.get_cache(self.collection_name)
            if not cache or cache["count"] == 0:
                return []
            # Slice active items only
            scores = cache["mat"][:cache["count"]] @ q
            
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = []
            for i in top_indices:
                results.append({
                    "text": cache["texts"][i],
                    "metadata": cache["metas"][i],
                    "score": float(scores[i])
                })
            return results

    def _search_historical(self, q: np.ndarray, top_k: int, date_str: str) -> List[Dict]:
        emb_path, db_path = self._get_hist_paths(date_str)
        if not os.path.exists(emb_path) or not os.path.exists(db_path):
            return []
            
        try:
            mat = np.load(emb_path, mmap_mode='r')
            if mat.shape[0] == 0: return []
            
            scores = mat @ q
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            cids = [int(i) for i in top_indices]
            texts, metas = self._read_historical_metadata(date_str, cids)
            
            results = []
            for i, txt, meta in zip(top_indices, texts, metas):
                results.append({
                    "text": txt,
                    "metadata": meta,
                    "score": float(scores[i])
                })
            return results
        except Exception as e:
            logger.error("Historical cache corrupted for %s: %s", date_str, e)
            return []

    # -- mongodb fallback ----------------------------------------------------

    def _search_server_side(
        self,
        query_vector: List[float],
        top_k: int,
        query_text: Optional[str] = None,
        source_collection: Optional[str] = None,
        explicit_date: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        col = self.connect()
        pipeline = []
        
        match_stage = {}
        if source_collection:
            match_stage["metadata.source_collection"] = source_collection
        if query_text:
            match_stage["$text"] = {"$search": query_text}
            
        if explicit_date:
            try:
                start = datetime.strptime(explicit_date, "%Y-%m-%d")
                if date_to:
                    end_dt = datetime.strptime(date_to, "%Y-%m-%d")
                    end = datetime(end_dt.year, end_dt.month, end_dt.day, 23, 59, 59, 999999)
                else:
                    end = datetime(start.year, start.month, start.day, 23, 59, 59, 999999)
                match_stage["metadata.source_created_at"] = {"$gte": start, "$lte": end}
            except ValueError:
                pass
                
        if match_stage:
            pipeline.append({"$match": match_stage})
            
        pipeline.append({"$sample": {"size": 2000}})
        
        try:
            candidates = list(col.aggregate(pipeline))
        except Exception as exc:
            logger.error("Server-side search failed: %s", exc)
            return []
            
        if not candidates:
            return []
            
        mat = np.array([c["embedding"] for c in candidates if "embedding" in c], dtype=np.float32)
        if mat.shape[0] == 0:
            return []
            
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
        
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q /= q_norm
        
        scores = mat @ q
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            idx = int(i)
            cand = candidates[idx]
            results.append({
                "text": cand.get("text", ""),
                "metadata": cand.get("metadata", {}),
                "score": float(scores[idx]),
            })
            
        return results
