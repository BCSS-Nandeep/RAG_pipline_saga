# RAG Memory Fix — Phase 2 Handoff

## 1. Objective

The objective of this workstream was to resolve a severe production memory incident where the `soceye-rag` application consumed excessive RAM, aggressively pushing the server's swap space to its absolute limit and causing massive system load (load average ~30). The goal was to identify the memory leaks/ballooning in the MongoDB vector loading logic and establish safe, bounded search capacities.

## 2. Original Root Cause

The original `VectorStore` implementation recklessly loaded entire MongoDB vector collections into unbounded Python lists and NumPy arrays during initialization. When fetching millions of embeddings into memory at once, Python's overhead and raw matrix allocations rapidly exhausted all available RAM, causing `soceye-rag` to thrash the OS swap file.

## 3. Phase 1 Fix

Phase 1 implemented critical safety rails around the original design:
- **`CACHE_SIZE_LIMIT`**: Imposed a strict 500,000 item ceiling on NumPy array cache loading.
- **Oversized Collection Detection**: Automatically evaluated collection counts and blocked numpy loading if documents exceeded the limit.
- **Server-Side Search Fallback**: Seamlessly redirected oversized collections to execute aggregate queries directly against MongoDB's `$vectorSearch`.
- **Streaming Cache Construction**: Loaded MongoDB cursors incrementally rather than dumping them into a giant Python list.
- **Temporary Files & Atomic Replacement**: Built new `.npy` caches using intermediate temp files (`.tmp`) that were atomically renamed to prevent corruption.
- **Build Locks**: Added `threading.RLock()` to prevent simultaneous threads from triggering redundant cache builds.
- **Cache Invalidation**: Handled `invalidate_cache()` by purging matrices and re-evaluating limits on the next query.

**What Phase 1 solved:** It successfully prevented single, massive collections (e.g., `analyses` with ~2.25M docs) from crashing the server, correctly offloading them to MongoDB server-side search.

## 4. Production Validation

- **Production Commit:** `3456168 updated RAG`
- **Application:** `soceye-rag`
- **Validation:** Successful health/query validation was previously performed. The Phase 1 safeguards were verified to be in place.
- **Deployment Status:** No Phase 2 code was deployed.

## 5. Newly Discovered Production Limitation

While Phase 1 protected against single massive collections, it failed to account for a **multi-collection multiplication problem**.
The `CACHE_SIZE_LIMIT = 500000` is strictly **per-collection**.

    per-collection 500K
        ↓
    multiple collections
        ↓
    multiple NumPy caches
        ↓
    cumulative process memory growth

During its scheduled ingestion cycle, the `soceye-rag` process scans dozens of collections. Observed production collections included:
- `analyses`: ~2.25M (correctly bypasses cache)
- `auditlogs`: ~261K (loads into RAM)
- `alerts`: ~257K (loads into RAM)
- `comments`: ~32K (loads into RAM)

Because many collections sit just underneath the 500K limit, Phase 1 loads them *all* into independent NumPy arrays simultaneously. This aggregates millions of vectors into the same Python process, fundamentally bypassing the intended memory constraint.

## 6. Host-Level Resource Constraint

The server itself is heavily oversubscribed:
- **Total RAM:** 15 GiB
- **Swap Used:** ~3.9 GiB / 4.0 GiB (97% full)
- **Observed RSS:**
  - `soceye-rag`: ~6 GiB
  - `sentiment-api`: ~7.6 GiB
  - `CopWriter`: ~5.3 GiB
- **Load Average:** ~30

**A. RAG's own cache architecture problem:** The lack of a global multi-collection memory manager causes `soceye-rag` to hoard ~6GB of RAM.
**B. Overall host memory oversubscription:** Even if `soceye-rag` is fixed, the host is attempting to run ~19GB worth of active Python applications on a 15GB machine.

## 7. Phase 2 Work Completed

The following items were developed locally as part of the Phase 2 RAG candidate:
- `vector_store_v2.py`: Introduced an initial attempt at memory bounds, memory-mapped historical vectors, and SQLite metadata.
- `store_factory.py`: Centralized VectorStore instantiation and feature flag routing.
- **Deterministic date routing:** Upgraded `intent.py` to extract exact dates/ranges without relying on an LLM.
- **Feature flag:** Bound Phase 2 behind `PHASE2_HOT_CACHE_ENABLED`.
- **Tests performed:** Unit tests for feature flags and date routing passed locally.
- **Commits:** `4230de3` (Freeze Phase 2 candidate) and `b2eb0af` (Integrate Phase 2 VectorStoreV2).

## 8. Phase 2 Audit Result

**NOT APPROVED FOR PRODUCTION**

Phase 2 was formally audited against multi-collection failure modes and rejected for deployment due to the following architectural failures:
- **Per-instance limits:** `HOT_CACHE_MAX_ITEMS` applied per-instance, failing to prevent the multi-collection multiplication problem.
- **No global memory manager:** No centralized entity tracks overall RAM usage across all active `VectorStoreV2` instances.
- **No global item budget:** Item limits are restricted to single collections.
- **No true LRU:** The cache sets an overflow flag to stop growth but implements no eviction mechanism to eject old arrays and free memory.
- **Allocation-before-admission:** The code pre-allocates massive NumPy arrays (e.g., 768 MB) *before* checking the RSS bounds, allowing limits to be vastly overshot.
- **Multiple VectorStore instances:** `_GLOBAL_STORES` caches redundant instances forever, preventing garbage collection.
- **Insufficient concurrency protection:** Simultaneous requests can concurrently allocate massive arrays, bypassing the RSS guard.
- **Inadequate multi-collection testing:** Existing tests mocked single-collection scaling but failed to evaluate cumulative multi-collection ingestion.

## 9. Correct Future Architecture

Future development must pivot to a centralized architecture:
**Process-wide HotCacheManager**
- **Global item budget & cache-owned memory budget:** A singular tracking entity governing total RAM across all collections.
- **Allocation admission BEFORE allocation:** Memory checks must evaluate projected size *before* calling `np.empty`.
- **Deterministic LRU eviction:** A global LRU queue that actively deletes least-recently-used arrays and forces Python GC when space is needed.
- **Global synchronization:** A centralized lock preventing concurrent over-allocation.
- **Process RSS emergency guard:** A fail-safe that strictly denies allocations if global RSS is saturated.
- **Bounded rebuild & Safe invalidation:** Incremental loads that respect global capacity.
- **MongoDB as source of truth.**
- **Historical mmap & SQLite metadata** (Preserve Phase 2's successful historical implementations).
- **Deterministic date routing & Daily rollover** (Preserve Phase 2's successful query routing).

*Note: Do NOT implement this architecture now. It is designated for the next session.*

## 10. Required Future Tests

Before Phase 2 can be reconsidered, the following automated tests must exist:
- Multi-collection memory test (proving total RAM stays flat across N collections).
- Global item limit validation.
- Global memory limit validation.
- Allocation-before-admission prevention.
- LRU eviction (proving older arrays are deleted).
- Actual memory release (verifying RSS/memory tracking drops after eviction).
- Concurrent allocation/rebuild safety (threading tests).
- Ingestion across many collections (simulating the scheduler).
- Rollover functionality.
- Historical mmap and Mongo fallback verification.
- Date routing edge cases.
- Process RSS guard strict enforcement.

## 11. Deployment Status

- **Phase 2 was NOT deployed.**
- **`PHASE2_HOT_CACHE_ENABLED` was NOT enabled.**
- **Production was NOT modified during this workstream.**

## 12. Next Engineer / Next Session

**Where work should resume:**
First redesign and implement a process-wide `HotCacheManager` locally.
Then run realistic multi-collection stress tests.
Then perform code review.
Then test against a genuine staging environment if available.
Only after that consider a controlled production canary.
