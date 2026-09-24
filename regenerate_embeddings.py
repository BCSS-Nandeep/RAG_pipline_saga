#!/usr/bin/env python3
"""
regenerate_embeddings.py — rebuild every SOC-EYE vector with the local
nomic-ai/nomic-embed-text-v1.5 model.

Source collections are never modified or deleted. Only generated vector
collections are replaced, and the old ones are renamed aside before the new
run so a failed rebuild cannot leave you with nothing.

    python regenerate_embeddings.py --preflight
        Steps 1-4 only: validate Mongo, sources, model load, a test batch,
        dimension, NaN/Inf and write access. Changes nothing.

    python regenerate_embeddings.py --rebuild [--collections a,b] [--limit N]
        Full sequence. Runs the preflight first and refuses to continue if any
        check fails, then retires the old vectors and regenerates.

    python regenerate_embeddings.py --status | --verify
    python regenerate_embeddings.py --drop-old      (after you are satisfied)

Resumable: progress is checkpointed per collection, so an interrupted run
continues where it stopped instead of duplicating work.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, OperationFailure

load_dotenv()

from chunker import TokenAwareChunker                        # noqa: E402
from embedder import (EXPECTED_DIM, DOCUMENT_PREFIX, QUERY_PREFIX,  # noqa: E402
                      EmbeddingConfigError, EmbeddingDimensionError,
                      describe_hardware, detect_hardware, get_embedder)
from processor import DocumentConverter                      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("regen")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
DB_NAME = os.getenv("DB_NAME", "test")
VECTOR_PREFIX = os.getenv("VECTOR_COLLECTION", "vector_embeddings")
RETIRED_PREFIX = "retired_"
CHECKPOINT_COLL = "rag_embed_runs_nomic15"

CHUNK_MIN = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
PAGE_SIZE = int(os.getenv("REGEN_PAGE_SIZE", "200"))

# Collections that carry retrievable narrative text. Override with
# EMBED_COLLECTIONS or --collections. Deliberately excludes machine exhaust
# (auditlogs, counters, retweet edges) that adds cost without helping recall.
_DEFAULT_COLLECTION_LIST = (
    "alerts,grievances,contents,events,dial100incidents,pois,keywords,"
    "sources,dailyprogrammes,telegrammessages,criticism_reports,"
    "suggestion_reports,query_reports,grievance_workflow_reports,"
    "reports,comments,analyses"
)
# `or _DEFAULT...` matters: an empty EMBED_COLLECTIONS= in .env makes getenv
# return "" rather than the default, which would silently reduce the rebuild to
# zero collections.
DEFAULT_COLLECTIONS = [
    c.strip()
    for c in (os.getenv("EMBED_COLLECTIONS", "").strip() or _DEFAULT_COLLECTION_LIST).split(",")
    if c.strip()
]


def connect() -> MongoClient:
    c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    c.admin.command("ping")
    return c


def vector_name(collection: str) -> str:
    return f"{VECTOR_PREFIX}_{collection}"


# --------------------------------------------------------------------------- #
# STEPS 1-4 : preflight
# --------------------------------------------------------------------------- #

def preflight(collections: List[str]) -> Optional[dict]:
    """Validate everything that must hold before any destructive step.

    Returns the loaded embedder's probe info on success, None on failure.
    Nothing here writes to a source collection.
    """
    print("=" * 74)
    print("PREFLIGHT")
    print("=" * 74)
    ok = True

    # -- STEP 1: Mongo + source data ---------------------------------------
    try:
        client = connect()
        db = client[DB_NAME]
        names = set(db.list_collection_names())
        print(f"  [PASS] MongoDB reachable — {DB_NAME}, {len(names)} collections")
    except Exception as exc:
        print(f"  [FAIL] MongoDB unreachable: {exc}")
        return None

    present, missing, total_docs = [], [], 0
    for c in collections:
        if c in names:
            n = db[c].estimated_document_count()
            if n:
                present.append((c, n))
                total_docs += n
            else:
                missing.append(f"{c} (empty)")
        else:
            missing.append(f"{c} (absent)")
    if present:
        print(f"  [PASS] {len(present)} source collections readable, "
              f"{total_docs:,} documents")
    else:
        print("  [FAIL] no source collections with data")
        ok = False
    if missing:
        print(f"         skipping: {', '.join(missing)}")

    # sample readability
    try:
        conv = DocumentConverter()
        sample_txt = ""
        for c, _ in present[:1]:
            d = db[c].find_one({})
            sample_txt = conv.convert(d) if d else ""
        if sample_txt.strip():
            print(f"  [PASS] source text extractable ({len(sample_txt)} chars from "
                  f"{present[0][0]})")
        else:
            print("  [FAIL] could not extract text from a sample document")
            ok = False
    except Exception as exc:
        print(f"  [FAIL] text extraction error: {exc}")
        ok = False

    # -- STEP 2: load the local model --------------------------------------
    hw = detect_hardware()
    print(f"  ---- hardware: {describe_hardware()}")
    if not hw["cuda_available"]:
        print("         NOTE: CUDA unavailable — running on CPU will be slow.")
    try:
        emb = get_embedder()
        t0 = time.time()
        info = emb.probe()
        load_s = time.time() - t0
    except EmbeddingConfigError as exc:
        print(f"  [FAIL] model load: {exc}")
        client.close()
        return None

    if info["healthy"]:
        print(f"  [PASS] model loaded — {info['model']} on {info['device']} "
              f"({load_s:.1f}s), batch={emb.batch_size}")
    else:
        print(f"  [FAIL] model unusable: {info['error']}")
        client.close()
        return None

    # -- STEP 3/4: test batch, dimension, NaN/Inf --------------------------
    probe_texts = ["A road accident was reported near Hyderabad.",
                   "Traffic grievance about potholes on the main road.",
                   "Political rally announcement in the city centre."]
    vecs = emb.embed_documents(probe_texts)
    if all(v is not None for v in vecs) and len(vecs) == len(probe_texts):
        print(f"  [PASS] test batch embedded — {len(vecs)} vectors")
    else:
        print("  [FAIL] test batch failed")
        ok = False

    dims = {len(v) for v in vecs if v}
    if dims == {EXPECTED_DIM}:
        print(f"  [PASS] dimension is exactly {EXPECTED_DIM}")
    else:
        print(f"  [FAIL] dimension {dims}, expected {{{EXPECTED_DIM}}}")
        ok = False

    bad = 0
    for v in vecs:
        for x in (v or []):
            if x != x or x in (float("inf"), float("-inf")):
                bad += 1
    if bad == 0:
        print("  [PASS] no NaN / Inf values")
    else:
        print(f"  [FAIL] {bad} NaN/Inf values present")
        ok = False

    # prefixes must actually change the vector, else they are not applied
    qv = emb.embed_query(probe_texts[0])
    dv = emb.embed_documents([probe_texts[0]])[0]
    if qv and dv and qv != dv:
        print(f"  [PASS] prefixes active — {QUERY_PREFIX!r} vs {DOCUMENT_PREFIX!r} "
              "produce different vectors")
    else:
        print("  [FAIL] query and document prefixes produced identical vectors")
        ok = False

    # -- write access -------------------------------------------------------
    try:
        probe_coll = db["_regen_write_probe"]
        probe_coll.insert_one({"_id": "probe", "at": datetime.now(timezone.utc)})
        probe_coll.delete_one({"_id": "probe"})
        probe_coll.drop()
        print("  [PASS] MongoDB write access confirmed")
    except Exception as exc:
        print(f"  [FAIL] MongoDB write access: {exc}")
        ok = False

    client.close()
    print("=" * 74)
    print(f"  PREFLIGHT {'PASSED' if ok else 'FAILED'}")
    print("=" * 74)
    return info if ok else None


# --------------------------------------------------------------------------- #
# STEP 5 : retire old vectors (rename, never drop)
# --------------------------------------------------------------------------- #

def retire_old_vectors(db) -> int:
    names = [n for n in db.list_collection_names()
             if n.startswith(VECTOR_PREFIX) and not n.startswith(RETIRED_PREFIX)]
    if not names:
        logger.info("No existing vector collections to retire.")
        return 0
    moved = 0
    for name in sorted(names):
        target = f"{RETIRED_PREFIX}{name}"
        count = db[name].estimated_document_count()
        if target in db.list_collection_names():
            db[target].drop()                     # a previous retire, superseded
        try:
            db[name].rename(target)
            moved += count
            logger.info("  retired %-46s -> %s (%s vectors)",
                        name, target, f"{count:,}")
        except OperationFailure as exc:
            logger.error("  could not retire %s: %s", name, exc)
    logger.info("Retired %s old vectors (renamed, not deleted).", f"{moved:,}")
    return moved


def drop_old_vectors() -> int:
    """Permanently remove retired_* collections. Run only after verifying."""
    client = connect()
    db = client[DB_NAME]
    names = [n for n in db.list_collection_names() if n.startswith(RETIRED_PREFIX)]
    if not names:
        print("Nothing retired to drop.")
        client.close()
        return 0
    total = sum(db[n].estimated_document_count() for n in names)
    print(f"About to permanently drop {len(names)} retired collections "
          f"({total:,} vectors). Source data is untouched.")
    for n in names:
        db[n].drop()
        print(f"  dropped {n}")
    client.close()
    return 0


# --------------------------------------------------------------------------- #
# STEPS 6-10 : regeneration
# --------------------------------------------------------------------------- #

def _iter_pages(coll, after_id, page_size: int) -> Iterable[list]:
    """Stream documents by _id so memory stays flat on huge collections."""
    last = after_id
    while True:
        q = {"_id": {"$gt": last}} if last is not None else {}
        page = list(coll.find(q).sort("_id", ASCENDING).limit(page_size))
        if not page:
            return
        yield page
        last = page[-1]["_id"]


def _checkpoint(db, collection: str) -> dict:
    return db[CHECKPOINT_COLL].find_one({"_id": collection}) or {}


def _save(db, collection, last_id, docs, chunks, failures, model, done=False):
    db[CHECKPOINT_COLL].update_one(
        {"_id": collection},
        {"$set": {"last_id_raw": last_id, "docs_processed": docs,
                  "chunks_written": chunks, "embed_failures": failures,
                  "embed_model": model, "embed_dim": EXPECTED_DIM,
                  "done": done, "updated_at": datetime.now(timezone.utc)}},
        upsert=True)


def rebuild(collections: List[str], limit: Optional[int], skip_retire: bool) -> int:
    info = preflight(collections)
    if info is None:
        logger.error("Preflight failed — nothing was changed.")
        return 2

    emb = get_embedder()
    model = info["model"]
    device = info["device"]
    client = connect()
    db = client[DB_NAME]
    names = set(db.list_collection_names())
    targets = [c for c in collections if c in names and db[c].estimated_document_count()]

    print()
    if not skip_retire:
        logger.info("STEP 5 — retiring old vector collections ...")
        retire_old_vectors(db)
    else:
        logger.info("STEP 5 — skipped (--skip-retire)")

    converter = DocumentConverter()
    chunker = TokenAwareChunker(min_tokens=CHUNK_MIN, max_tokens=CHUNK_MAX,
                                overlap=CHUNK_OVERLAP)
    grand = {"docs": 0, "chunks": 0, "failures": 0}
    started = time.time()

    for collection in targets:
        out = db[vector_name(collection)]
        try:
            out.create_index([("metadata.document_id", ASCENDING),
                              ("metadata.chunk_index", ASCENDING)],
                             unique=True, name="upsert_dedup_idx", background=True)
            out.create_index([("metadata.source_collection", ASCENDING)],
                             name="source_collection_idx", background=True)
        except OperationFailure as exc:
            logger.warning("  index on %s: %s", vector_name(collection), exc)

        cp = _checkpoint(db, collection)
        if cp.get("done") and cp.get("embed_model") == model:
            logger.info("%s already complete — skipping.", collection)
            continue

        after = cp.get("last_id_raw")
        docs = cp.get("docs_processed", 0)
        chunks = cp.get("chunks_written", 0)
        failures = cp.get("embed_failures", 0)
        if after is not None:
            logger.info("%s — resuming (%s docs, %s chunks already done)",
                        collection, f"{docs:,}", f"{chunks:,}")

        total = db[collection].estimated_document_count()
        t0 = time.time()
        last_log = t0

        for page in _iter_pages(db[collection], after, PAGE_SIZE):
            texts: List[str] = []
            metas: List[dict] = []
            for doc in page:
                doc_id = str(doc.get("_id", ""))
                text = converter.convert(doc)
                if not text.strip():
                    continue
                # Same fingerprint the API scheduler computes (sha1 of the
                # converted document text). Without it every rebuilt vector
                # counts as "legacy" and the scheduler skips it forever, so an
                # edited document would stay on its original vector for good.
                digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                for ch in chunker.chunk_document(text, collection, doc_id):
                    texts.append(ch.text)
                    metas.append({
                        "source_collection": collection,
                        "document_id": doc_id,
                        "chunk_index": ch.metadata.chunk_index,
                        "chunk_id": f"{doc_id}::{ch.metadata.chunk_index}",
                        "total_chunks": ch.metadata.total_chunks,
                        "source_created_at": doc.get("created_at"),
                        "content_hash": digest,
                        "embed_model": model,
                    })

            ops: List[UpdateOne] = []
            if texts:
                try:
                    # Document side always gets the document prefix.
                    vectors = emb.embed_documents(texts)
                except EmbeddingDimensionError as exc:
                    logger.error("FATAL: %s", exc)
                    _save(db, collection, page[-1]["_id"], docs, chunks,
                          failures, model)
                    client.close()
                    return 3

                now = datetime.now(timezone.utc)
                for text, meta, vec in zip(texts, metas, vectors):
                    if vec is None:
                        failures += 1
                        continue
                    meta = {**meta, "embed_model": model,
                            "embed_dim": len(vec), "embed_device": device,
                            "created_at": now}
                    ops.append(UpdateOne(
                        {"metadata.document_id": meta["document_id"],
                         "metadata.chunk_index": meta["chunk_index"]},
                        {"$set": {"text": text, "embedding": vec,
                                  "metadata": meta}},
                        upsert=True))

            if ops:
                try:
                    r = out.bulk_write(ops, ordered=False)
                    chunks += r.upserted_count + r.modified_count
                except BulkWriteError as exc:
                    logger.warning("  bulk write partial failure: %s",
                                   str(exc)[:160])

            docs += len(page)
            _save(db, collection, page[-1]["_id"], docs, chunks, failures, model)

            now_t = time.time()
            if now_t - last_log >= 10:
                last_log = now_t
                rate = docs / max(1e-6, now_t - t0)
                eta = (total - docs) / rate if rate > 0 else 0
                logger.info("  %s: %s/%s docs (%.1f%%) %s vectors  "
                            "%.0f docs/s  ETA %.0fm",
                            collection, f"{docs:,}", f"{total:,}",
                            100 * docs / max(1, total), f"{chunks:,}", rate, eta / 60)

            if limit and docs >= limit:
                logger.info("  %s: --limit %d reached.", collection, limit)
                break
        else:
            _save(db, collection, None, docs, chunks, failures, model, done=True)

        logger.info("%s complete — %s docs, %s vectors, %s failures (%.1fs)",
                    collection, f"{docs:,}", f"{chunks:,}", f"{failures:,}",
                    time.time() - t0)
        grand["docs"] += docs
        grand["chunks"] += chunks
        grand["failures"] += failures

    elapsed = time.time() - started
    print()
    logger.info("=" * 66)
    logger.info("REBUILD COMPLETE in %.1f min", elapsed / 60)
    logger.info("  model              : %s", model)
    logger.info("  device             : %s", device)
    logger.info("  dimension          : %d", EXPECTED_DIM)
    logger.info("  collections        : %d", len(targets))
    logger.info("  documents processed: %s", f"{grand['docs']:,}")
    logger.info("  vectors stored     : %s", f"{grand['chunks']:,}")
    logger.info("  failed embeddings  : %s", f"{grand['failures']:,}")
    if elapsed > 0:
        logger.info("  throughput         : %.0f vectors/s",
                    grand["chunks"] / elapsed)
    logger.info("=" * 66)
    client.close()
    return 0


# --------------------------------------------------------------------------- #

def verify() -> int:
    client = connect()
    db = client[DB_NAME]
    names = [n for n in db.list_collection_names()
             if n.startswith(VECTOR_PREFIX) and not n.startswith(RETIRED_PREFIX)]
    print(f"\n{'vector collection':<44}{'vectors':>12}{'dim':>6}  model")
    print("-" * 92)
    total, bad = 0, 0
    for n in sorted(names):
        c = db[n].estimated_document_count()
        if not c:
            continue
        total += c
        d = db[n].find_one({}, {"embedding": 1, "metadata": 1})
        dim = len(d.get("embedding") or []) if d else 0
        models = db[n].distinct("metadata.embed_model") or ["<unrecorded>"]
        flag = ""
        if dim != EXPECTED_DIM:
            flag += "  <- DIM MISMATCH"; bad += 1
        if len(models) > 1:
            flag += "  <- MIXED MODELS"; bad += 1
        print(f"{n:<44}{c:>12,}{dim:>6}  {','.join(map(str, models))}{flag}")
    print("-" * 92)
    print(f"{'TOTAL':<44}{total:>12,}\n")
    retired = [n for n in db.list_collection_names() if n.startswith(RETIRED_PREFIX)]
    if retired:
        rt = sum(db[n].estimated_document_count() for n in retired)
        print(f"  {len(retired)} retired collections still held ({rt:,} old vectors) "
              f"— drop with --drop-old once satisfied.\n")
    client.close()
    return 1 if bad else 0


def status() -> int:
    client = connect()
    db = client[DB_NAME]
    rows = list(db[CHECKPOINT_COLL].find({}))
    if not rows:
        print("\nNo rebuild checkpoints recorded yet.\n")
        client.close()
        return 0
    print(f"\n{'collection':<28}{'docs':>12}{'vectors':>12}{'fails':>8}{'done':>7}  model")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: x["_id"]):
        print(f"{r['_id']:<28}{r.get('docs_processed',0):>12,}"
              f"{r.get('chunks_written',0):>12,}{r.get('embed_failures',0):>8,}"
              f"{str(r.get('done',False)):>7}  {r.get('embed_model','?')}")
    print()
    client.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--rebuild", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--drop-old", action="store_true")
    ap.add_argument("--collections")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-retire", action="store_true",
                    help="do not retire existing vectors (resume an interrupted run)")
    args = ap.parse_args()

    cols = ([c.strip() for c in args.collections.split(",") if c.strip()]
            if args.collections else DEFAULT_COLLECTIONS)

    if args.preflight:
        return 0 if preflight(cols) else 1
    if args.verify:
        return verify()
    if args.status:
        return status()
    if args.drop_old:
        return drop_old_vectors()
    return rebuild(cols, args.limit, args.skip_retire)


if __name__ == "__main__":
    sys.exit(main())
