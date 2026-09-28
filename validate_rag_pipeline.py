#!/usr/bin/env python3
"""
validate_rag_pipeline.py — end-to-end validation of the vLLM-only RAG stack.

    python validate_rag_pipeline.py

Runs, in order:

    A  embedding health           (vLLM /v1/embeddings reachable, model loaded)
    B  single document embedding
    C  batch embedding
    D  query embedding            (fresh vector per question)
    E  MongoDB vector inventory   (dimensions + provenance, read-only)
    F  vector similarity search
    G  query -> embedding -> retrieval
    H  retrieval -> Qwen3-14B generation
    I  complete RAG answer via the API

Probe questions span the SOC-EYE categories: accidents, grievances,
political/social, traffic, and crime / law-and-order.

Read-only with respect to MongoDB — it never writes or deletes.
Exit code 0 only if every executed check passes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.ERROR)

from embedder import (EXPECTED_DIM, EmbeddingConfigError,   # noqa: E402
                      describe_hardware, detect_hardware, get_embedder)
import llm_client                                           # noqa: E402
from store_factory import get_vector_store as VectorStore                        # noqa: E402

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
DB_NAME = os.getenv("DB_NAME", "test")
VECTOR_PREFIX = os.getenv("VECTOR_COLLECTION", "vector_embeddings")
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8099")

PROBES = [
    ("accidents",   "Give me an accident post which are in alerts"),
    ("grievances",  "Show me grievance posts"),
    ("political",   "Show me political posts"),
    ("traffic",     "Show me traffic related complaints"),
    ("crime",       "Show me crime and law and order incidents"),
]

SEARCH_STORES = [f"{VECTOR_PREFIX}_alerts", f"{VECTOR_PREFIX}_grievances"]

_results: List[tuple] = []


def record(key: str, name: str, passed: Optional[bool], detail: str = "") -> None:
    _results.append((key, name, passed, detail))
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[passed]
    print(f"  [{mark}] {key}. {name}" + (f" — {detail}" if detail else ""))


def fingerprint(vec: List[float]) -> str:
    return hashlib.sha1(",".join(f"{v:.5f}" for v in vec[:32]).encode()).hexdigest()[:12]


def main() -> int:
    print("=" * 78)
    print("SOC-EYE RAG PIPELINE VALIDATION  (self-hosted vLLM only)")
    print("=" * 78)
    hw = detect_hardware()
    print(f"  embedding model    : {os.getenv('EMBED_MODEL', '<unset>')}  (local, in-process)")
    print(f"  expected dimension : {EXPECTED_DIM}")
    print(f"  hardware           : {describe_hardware()}")
    print(f"  device in use      : {'GPU - ' + str(hw['gpu_name']) if hw['cuda_available'] else 'CPU'}")
    print(f"  LLM endpoint       : {llm_client.LLM_BASE_URL}")
    print(f"  LLM model          : {llm_client.LLM_MODEL}")
    print()

    # ---- A: embedding health ---------------------------------------------
    print("-- embedding --")
    try:
        emb = get_embedder()
    except EmbeddingConfigError as exc:
        record("A", "embedding health", False, str(exc))
        emb = None
    else:
        info = emb.probe()
        if info["healthy"]:
            record("A", "embedding health", True,
                   f"{info['model']} on {info['device']} dim={info['dimension']} "
                   f"batch={emb.batch_size}")
        else:
            record("A", "embedding health", False, str(info["error"]))
            emb = None

    vectors: Dict[str, List[float]] = {}

    # ---- B: single document embedding ------------------------------------
    if emb is None:
        for k, n in [("B", "single document embedding"),
                     ("C", "batch embedding"),
                     ("D", "query embedding")]:
            record(k, n, None, "embedding server unavailable")
    else:
        v = emb.embed_documents(["A road accident was reported near Hyderabad."])[0]
        if v and len(v) == EXPECTED_DIM:
            record("B", "single document embedding", True, f"dim={len(v)}")
        else:
            record("B", "single document embedding", False,
                   f"got {len(v) if v else None}")

        # ---- C: batch embedding ------------------------------------------
        batch = [q for _, q in PROBES]
        vs = emb.embed_documents(batch)
        ok = len(vs) == len(batch) and all(x is not None and len(x) == EXPECTED_DIM
                                           for x in vs)
        record("C", "batch embedding", ok,
               f"{sum(1 for x in vs if x)}/{len(batch)} vectors, dim="
               f"{len(vs[0]) if vs and vs[0] else '?'}")

        # prefixes must actually differentiate the two sides
        dq = emb.embed_query(batch[0])
        dd = emb.embed_documents([batch[0]])[0]
        record("C2", "prefix consistency", bool(dq and dd and dq != dd),
               "search_query: vs search_document: produce different vectors")

        # ---- D: query embedding, one fresh vector per question -----------
        for cat, q in PROBES:
            qv = emb.embed_query(q)
            if qv:
                vectors[cat] = qv
        distinct = len({fingerprint(v) for v in vectors.values()})
        record("D", "query embedding", distinct == len(PROBES),
               f"{len(vectors)}/{len(PROBES)} embedded, {distinct} distinct vectors")

    # ---- E: MongoDB vector inventory -------------------------------------
    print("-- storage --")
    from pymongo import MongoClient
    dims, models, total = set(), set(), 0
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        db = client[DB_NAME]
        names = [n for n in db.list_collection_names()
                 if n.startswith(VECTOR_PREFIX) and not n.startswith("oldvec_")]
        for n in names:
            c = db[n].estimated_document_count()
            if not c:
                continue
            total += c
            d = db[n].find_one({}, {"embedding": 1, "metadata": 1})
            if d and d.get("embedding"):
                dims.add(len(d["embedding"]))
            models.update(db[n].distinct("metadata.embed_model") or [])
        client.close()
        consistent = len(dims) == 1 and EXPECTED_DIM in dims
        record("E", "MongoDB vector inventory", consistent,
               f"{total:,} vectors, dims={sorted(dims)}, "
               f"models={sorted(str(m) for m in models) or ['<unrecorded>']}")
    except Exception as exc:
        record("E", "MongoDB vector inventory", False, str(exc)[:80])

    # ---- F/G: similarity search + question-specific retrieval ------------
    print("-- retrieval --")
    retrieved: Dict[str, List[str]] = {}
    if not vectors:
        record("F", "vector similarity search", None, "no query vectors")
        record("G", "query -> embedding -> retrieval", None, "no query vectors")
    else:
        for cat, qv in vectors.items():
            hits = []
            for store_name in SEARCH_STORES:
                st = VectorStore(MONGODB_URI, DB_NAME, store_name)
                try:
                    for r in st.cosine_search(query_vector=qv, top_k=5,
                                              query_text=dict(PROBES)[cat]):
                        m = r.get("metadata", {})
                        hits.append((r.get("score", 0.0), m.get("document_id", "?")))
                except Exception:
                    pass
                finally:
                    st.close()
            hits.sort(reverse=True)
            retrieved[cat] = [d for _, d in hits[:5]]
            if hits:
                print(f"       {cat:<12} top score {hits[0][0]:.4f}  "
                      f"ids {retrieved[cat][:3]}")

        got = sum(1 for v in retrieved.values() if v)
        record("F", "vector similarity search", got == len(vectors),
               f"{got}/{len(vectors)} categories returned hits")

        overlaps = []
        cats = list(retrieved)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = set(retrieved[cats[i]]), set(retrieved[cats[j]])
                if a and b:
                    overlaps.append(len(a & b) / max(1, len(a | b)))
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0
        record("G", "query -> embedding -> retrieval", mean_overlap < 0.5,
               f"mean cross-category overlap {mean_overlap:.0%} (lower is better)")

    # ---- H: generation ----------------------------------------------------
    print("-- generation --")
    if not llm_client.check_health():
        record("H", "retrieval -> Qwen3-14B generation", False,
               f"LLM unhealthy at {llm_client.LLM_BASE_URL}")
    else:
        meta: dict = {}
        out = llm_client.generate("Reply with exactly: GEN OK", max_tokens=16,
                                  temperature=0.0, meta=meta)
        record("H", "retrieval -> Qwen3-14B generation",
               meta.get("status") == 200 and "GEN OK" in out.upper(),
               f"model={meta.get('model')} status={meta.get('status')}")

    # ---- I: complete RAG answer via the API -------------------------------
    print("-- end-to-end --")
    try:
        t0 = time.time()
        r = requests.post(f"{API_URL}/api/rag/query",
                          json={"question": PROBES[0][1], "time_window_days": 3650},
                          timeout=1800)
        d = r.json()
        ans = d.get("answer") or ""
        ok = r.ok and bool(ans) and "LLM generation failed" not in ans
        record("I", "complete RAG answer", ok,
               f"{time.time()-t0:.0f}s intent={d.get('intent')} "
               f"contexts={d.get('context_count')} chars={len(ans)}")
    except Exception as exc:
        record("I", "complete RAG answer", None,
               f"API not running at {API_URL} ({type(exc).__name__})")

    # ---- summary ----------------------------------------------------------
    print()
    print("=" * 78)
    passed = sum(1 for *_, p, _ in [(r[0], r[1], r[2], r[3]) for r in _results] if p is True)
    failed = sum(1 for r in _results if r[2] is False)
    skipped = sum(1 for r in _results if r[2] is None)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("\n  Failures:")
        for k, n, p, det in _results:
            if p is False:
                print(f"    {k}. {n}: {det}")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
