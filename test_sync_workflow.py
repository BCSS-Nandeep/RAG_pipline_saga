#!/usr/bin/env python3
"""
test_sync_workflow.py — end-to-end test of the MongoDB -> embedding sync.

Runs against an ISOLATED database (default: soceye_synctest) with a handful of
synthetic documents. It never reads or writes the production corpus.

    python test_sync_workflow.py

Cases:
    1  fresh collection, no embeddings -> generated automatically
    2  new document added              -> picked up on the next cycle
    3  existing document edited        -> ONLY that document re-embedded
    4  restart                         -> nothing re-embedded
    5  interrupted job                 -> resumes, no duplicate work
    6  a failing document              -> isolated, recorded, retryable
    7  RAG retrieval after each step   -> returns the expected document
"""

from __future__ import annotations

import logging
import os
import sys
import time

os.environ.setdefault("DB_NAME", "soceye_synctest")
os.environ.setdefault("INGEST_SCHEDULER_ENABLED", "false")

from dotenv import load_dotenv

load_dotenv(override=False)
os.environ["DB_NAME"] = os.getenv("SYNC_TEST_DB", "soceye_synctest")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from pymongo import MongoClient                    # noqa: E402

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
TEST_DB = os.environ["DB_NAME"]
COLL = "alerts"                    # name the pipeline already knows how to read
VEC = f"vector_embeddings_{COLL}"

PROD_DB = "test"                   # must never be touched

DOCS = [
    {"_id": "doc-accident-1",
     "title": "Road accident on ORR",
     "description": "A serious road accident occurred near the Outer Ring Road "
                    "involving two cars. Traffic police responded quickly.",
     "priority": "HIGH", "source_category": "traffic"},
    {"_id": "doc-grievance-1",
     "title": "Water supply complaint",
     "description": "Residents complain about irregular water supply in the "
                    "colony for the past two weeks.",
     "priority": "MEDIUM", "source_category": "civic"},
    {"_id": "doc-political-1",
     "title": "Political rally announcement",
     "description": "A large political rally is scheduled at the city centre "
                    "with an expected gathering of five thousand people.",
     "priority": "MEDIUM", "source_category": "political"},
]

_results = []


def check(n, name, ok, detail=""):
    _results.append((n, name, ok, detail))
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"  [{mark}] Case {n}: {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def vec_counts(db):
    """document_id -> number of stored chunks."""
    out = {}
    for row in db[VEC].find({}, {"metadata.document_id": 1}):
        did = (row.get("metadata") or {}).get("document_id")
        out[did] = out.get(did, 0) + 1
    return out


def hashes(db):
    out = {}
    for row in db[VEC].find({}, {"metadata.document_id": 1,
                                 "metadata.content_hash": 1}):
        m = row.get("metadata") or {}
        out[m.get("document_id")] = m.get("content_hash")
    return out


def retrieve(question, db, top_k=1):
    """Embed the question and cosine-search the test vector store."""
    import numpy as np
    from embedder import get_embedder
    qv = np.array(get_embedder().embed_query(question), dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)
    best = []
    for row in db[VEC].find({}, {"embedding": 1, "metadata": 1}):
        v = np.array(row["embedding"], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        best.append((float(v @ qv), (row.get("metadata") or {}).get("document_id")))
    best.sort(reverse=True)
    return best[:top_k]


def main() -> int:
    if TEST_DB == PROD_DB:
        print(f"REFUSING TO RUN: test DB resolves to {PROD_DB!r}.")
        return 2

    print("=" * 76)
    print(f"SYNC WORKFLOW TEST  (isolated database: {TEST_DB})")
    print("=" * 76)

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    db = client[TEST_DB]

    prod_before = client[PROD_DB].command("dbstats")["objects"]

    # clean slate for the test DB only
    client.drop_database(TEST_DB)
    db[COLL].insert_many(DOCS[:2])
    print(f"  seeded {db[COLL].count_documents({})} source documents, "
          f"{db[VEC].estimated_document_count()} vectors\n")

    import api_server
    api_server.DB_NAME = TEST_DB          # point the sync at the test DB

    # ---- Case 1: fresh collection, no embeddings ------------------------
    t0 = time.time()
    r1 = api_server._run_ingest(COLL)
    c1 = vec_counts(db)
    check(1, "fresh collection auto-embedded",
          r1["docs_new"] == 2 and len(c1) == 2 and r1["chunks_stored"] > 0,
          f"new={r1['docs_new']} chunks={r1['chunks_stored']} "
          f"docs_with_vectors={len(c1)} ({time.time()-t0:.1f}s)")

    dim = len((db[VEC].find_one({}) or {}).get("embedding") or [])
    check("1b", "stored vectors are 768-d", dim == 768, f"dim={dim}")

    hit = retrieve("road accident on the ring road", db)
    check("1c", "RAG retrieval after initial embed",
          bool(hit) and hit[0][1] == "doc-accident-1",
          f"top={hit[0][1]} score={hit[0][0]:.4f}" if hit else "no hits")

    # ---- Case 2: new document added -------------------------------------
    db[COLL].insert_one(DOCS[2])
    r2 = api_server._run_ingest(COLL)
    c2 = vec_counts(db)
    check(2, "new document detected and embedded",
          r2["docs_new"] == 1 and r2["docs_skipped"] == 2 and len(c2) == 3,
          f"new={r2['docs_new']} unchanged={r2['docs_skipped']} "
          f"changed={r2['docs_changed']}")

    hit = retrieve("political rally at the city centre", db)
    check("2b", "RAG retrieval finds the new document",
          bool(hit) and hit[0][1] == "doc-political-1",
          f"top={hit[0][1]} score={hit[0][0]:.4f}" if hit else "no hits")

    # ---- Case 3: modify an existing document ----------------------------
    h_before = hashes(db)
    db[COLL].update_one(
        {"_id": "doc-grievance-1"},
        {"$set": {"description":
                  "Residents now report a complete water outage and a burst "
                  "pipeline flooding the main road near the market."}})
    r3 = api_server._run_ingest(COLL)
    h_after = hashes(db)
    changed_ids = [k for k in h_before
                   if h_before.get(k) != h_after.get(k)]
    check(3, "edited document re-embedded, others untouched",
          r3["docs_changed"] == 1 and r3["docs_new"] == 0
          and r3["docs_skipped"] == 2 and changed_ids == ["doc-grievance-1"],
          f"changed={r3['docs_changed']} unchanged={r3['docs_skipped']} "
          f"hash_changed={changed_ids}")

    hit = retrieve("burst pipeline flooding the road", db)
    check("3b", "RAG retrieval reflects the edited content",
          bool(hit) and hit[0][1] == "doc-grievance-1",
          f"top={hit[0][1]} score={hit[0][0]:.4f}" if hit else "no hits")

    # ---- Case 4: restart -> no re-embedding -----------------------------
    before = db[VEC].estimated_document_count()
    r4 = api_server._run_ingest(COLL)
    after = db[VEC].estimated_document_count()
    check(4, "restart does not re-embed",
          r4["docs_new"] == 0 and r4["docs_changed"] == 0
          and r4["docs_skipped"] == 3 and before == after,
          f"new={r4['docs_new']} changed={r4['docs_changed']} "
          f"unchanged={r4['docs_skipped']} vectors {before}->{after}")

    # ---- Case 5: interrupted job resumes --------------------------------
    # Simulate a crash mid-run by deleting one document's chunks, as if it had
    # never completed, then re-running.
    db[VEC].delete_many({"metadata.document_id": "doc-political-1"})
    partial = vec_counts(db)
    r5 = api_server._run_ingest(COLL)
    resumed = vec_counts(db)
    check(5, "interrupted job resumes from last good record",
          "doc-political-1" not in partial
          and "doc-political-1" in resumed
          and r5["docs_new"] == 1 and r5["docs_skipped"] == 2,
          f"re-embedded only the missing doc: new={r5['docs_new']} "
          f"unchanged={r5['docs_skipped']}")

    # ---- Case 6: a failing document is isolated -------------------------
    db[COLL].insert_one({"_id": "doc-broken-1", "title": "breaks the embedder",
                         "description": "this document will fail to embed"})
    from embedder import get_embedder
    emb = get_embedder()
    real = emb.embed_documents

    def flaky(texts):
        if any("will fail to embed" in t for t in texts):
            raise RuntimeError("simulated embedding failure")
        return real(texts)

    emb.embed_documents = flaky
    try:
        r6 = api_server._run_ingest(COLL)
    finally:
        emb.embed_documents = real

    good_still_there = len(vec_counts(db)) >= 3
    check(6, "one failing document does not stop the worker",
          r6["embed_failures"] >= 1
          and "doc-broken-1" in r6.get("failed_document_ids", [])
          and good_still_there,
          f"failures={r6['embed_failures']} "
          f"failed_ids={r6.get('failed_document_ids')} "
          f"other_docs_intact={good_still_there}")

    # retryable: the failure left no hash, so the next cycle tries again
    r6b = api_server._run_ingest(COLL)
    check("6b", "failed document is retried next cycle",
          "doc-broken-1" in vec_counts(db) and r6b["docs_new"] == 1,
          f"new={r6b['docs_new']} (recovered once the embedder worked)")

    # ---- Case 7: final retrieval sanity ---------------------------------
    probes = [("road accident on the ring road", "doc-accident-1"),
              ("burst pipeline flooding the road", "doc-grievance-1"),
              ("political rally at the city centre", "doc-political-1")]
    hits = [(q, retrieve(q, db)[0]) for q, _ in probes]
    correct = sum(1 for (q, exp), (_, (s, got)) in zip(probes, hits) if got == exp)
    for (q, exp), (_, (s, got)) in zip(probes, hits):
        print(f"         {q[:38]:<40} -> {got} ({s:.4f})")
    check(7, "RAG retrieval correct across all categories",
          correct == len(probes), f"{correct}/{len(probes)} returned the expected document")

    # ---- production corpus untouched ------------------------------------
    prod_after = client[PROD_DB].command("dbstats")["objects"]
    check("SAFE", "production corpus untouched",
          prod_before == prod_after,
          f"{PROD_DB}: {prod_before:,} objects before and after")

    client.drop_database(TEST_DB)
    client.close()

    print()
    print("=" * 76)
    failed = sum(1 for r in _results if r[2] is False)
    print(f"  {sum(1 for r in _results if r[2] is True)} passed, {failed} failed")
    if failed:
        for n, name, ok, det in _results:
            if ok is False:
                print(f"    Case {n} ({name}): {det}")
    print("=" * 76)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
