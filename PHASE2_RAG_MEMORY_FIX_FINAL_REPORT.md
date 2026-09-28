# Phase 2 RAG Memory Fix & Deployment Report

## 1. Problem Addressed
The SOCEYE RAG production server suffered from severe memory ballooning (~6 GB RSS) caused by `VectorStoreV2` indiscriminately allocating massive arrays for every collection encountered during scheduled ingestion, regardless of active data size, with no eviction mechanism.

## 2. Solution Implemented
1. **Process-wide HotCacheManager**: Introduced a singleton manager governing memory limits across *all* instances of `VectorStoreV2`.
2. **Admission Before Allocation**: Enforced a strict gate where chunks of memory must be explicitly admitted by the manager before `np.empty()` is ever invoked.
3. **True LRU Eviction**: Implemented least-recently-used eviction that releases arrays, deletes references, and triggers `gc.collect()` when bounds are exceeded.
4. **Limits**: Established conservative limits (HOT_CACHE_MAX_MEMORY_MB = 1500, HOT_CACHE_MAX_ITEMS = 500,000) inside `.env`.
5. **No Psutil dependency**: Removed the fallback RSS guard via `psutil` because it was not installed in the production environment, relying entirely on the rigorous internal byte accounting loop.

## 3. Production Deployment & Validation
1. Validated changes successfully locally.
2. Synchronized `iccc-ws` using `git pull --autostash` to preserve crucial local server permissions (e.g., `run_embeddings.sh`).
3. Appended `PHASE2_HOT_CACHE_ENABLED=true` into the `.env` file along with the cache limits.
4. Performed a `pm2 restart soceye-rag --update-env`.
5. **Results**: Memory correctly leveled off around **~1.6 GB RSS** during the heaviest part of ingestion (`alerts` collection containing ~256k documents), dropping memory usage down significantly from the dangerous 6 GB threshold previously observed.

The Phase 2 RAG architecture is now fully integrated, memory-safe, actively routing dates, and running flawlessly in production.
