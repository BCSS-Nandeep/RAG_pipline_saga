"""
pipeline.py — CLI entry point and orchestrator for the RAG pipeline.

Usage:
    python pipeline.py --ingest              # Run stages 1–5 (stream → embed → store)
    python pipeline.py --query "question"    # Run stage 6  (ask the AI assistant)
    python pipeline.py --check               # Verify MongoDB + Ollama connectivity
    python pipeline.py --stats               # Show collection & vector store stats
"""

import argparse
import json
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv
from tqdm import tqdm

from processor import MongoStreamProcessor, DocumentConverter
from chunker import TokenAwareChunker
from embedder import OllamaEmbedder
from vector_store import VectorStore
from assistant import Assistant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "your_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "your_collection")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "vector_embeddings")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH_SIZE", "50"))
CHUNK_MIN = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".ingest_checkpoint.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

def run_health_check() -> bool:
    """Verify MongoDB and Ollama (both embedding + LLM models) are reachable."""
    ok = True

    # MongoDB
    logger.info("Checking MongoDB at %s ...", MONGODB_URI)
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[DB_NAME]
        cols = db.list_collection_names()
        logger.info("  MongoDB OK — database '%s' has %d collections.", DB_NAME, len(cols))
        if COLLECTION_NAME not in cols:
            logger.warning("  Source collection '%s' not found in database.", COLLECTION_NAME)
        client.close()
    except Exception as exc:
        logger.error("  MongoDB FAILED: %s", exc)
        ok = False

    # Ollama — embedding model
    logger.info("Checking Ollama embedding model '%s' at %s ...", OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    embed_ok = OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL).check_health()
    if not embed_ok:
        ok = False

    # Ollama — LLM model
    logger.info("Checking Ollama LLM model '%s' at %s ...", OLLAMA_LLM_MODEL, OLLAMA_BASE_URL)
    import requests
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if any(OLLAMA_LLM_MODEL in m for m in models):
            logger.info("  LLM model '%s' available.", OLLAMA_LLM_MODEL)
        else:
            logger.error(
                "  LLM model '%s' NOT found. Available: %s. Run: ollama pull %s",
                OLLAMA_LLM_MODEL, models, OLLAMA_LLM_MODEL,
            )
            ok = False
    except Exception as exc:
        logger.error("  Ollama LLM check FAILED: %s", exc)
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Ingestion (stages 1–5)
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    """Load the last checkpoint (batch number + last _id processed)."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_checkpoint(batch_number: int, last_id: str, docs_done: int, chunks_done: int):
    """Save progress after each successful batch."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "batch_number": batch_number,
            "last_id": last_id,
            "docs_processed": docs_done,
            "chunks_stored": chunks_done,
            "collection": COLLECTION_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def _clear_checkpoint():
    """Remove checkpoint file when ingestion is complete."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def run_ingestion(full: bool = False):
    """Execute the ingest pipeline in discrete batches with checkpoint resumability.

    Each batch:
      1. Fetches BATCH_SIZE docs from MongoDB (sorted by _id, paginated)
      2. Converts + chunks them
      3. Embeds all chunks in one batch call to Ollama
      4. Stores in vector DB
      5. Saves a checkpoint file

    On restart, resumes from the last completed batch.
    """
    from datetime import datetime, timezone

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START (batch mode)")
    logger.info("=" * 60)

    # --- Initialise components ---
    streamer = MongoStreamProcessor(MONGODB_URI, DB_NAME, COLLECTION_NAME, BATCH_SIZE)
    converter = DocumentConverter()
    chunker = TokenAwareChunker(min_tokens=CHUNK_MIN, max_tokens=CHUNK_MAX, overlap=CHUNK_OVERLAP)
    embedder = OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)
    store = VectorStore(MONGODB_URI, DB_NAME, VECTOR_COLLECTION)

    # Pre-flight
    if not embedder.check_health():
        logger.error("Ollama embedding model not available — aborting.")
        sys.exit(1)

    total_docs = streamer.count_documents()
    logger.info("Source collection has ~%d documents.", total_docs)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

    # --- Resume from checkpoint ---
    checkpoint = {} if full else _load_checkpoint()
    start_batch = 1
    after_id = None
    docs_processed = 0
    chunks_stored = 0

    if checkpoint and checkpoint.get("collection") == COLLECTION_NAME:
        start_batch = checkpoint["batch_number"] + 1
        after_id = checkpoint.get("last_id")
        docs_processed = checkpoint.get("docs_processed", 0)
        chunks_stored = checkpoint.get("chunks_stored", 0)
        logger.info(
            "RESUMING from batch %d (last _id: %s, %d docs / %d chunks done)",
            start_batch, after_id, docs_processed, chunks_stored,
        )
    else:
        if full:
            _clear_checkpoint()
        logger.info("Starting fresh — %d batches of %d docs each.", total_batches, BATCH_SIZE)

    # --- Batch loop ---
    embed_failures = 0
    batch_num = start_batch

    pbar = tqdm(
        total=total_docs,
        initial=docs_processed,
        desc="Ingesting",
        unit="doc",
    )

    while True:
        # Fetch one batch from MongoDB
        docs = streamer.fetch_batch(batch_num, after_id=after_id)
        if not docs:
            logger.info("No more documents — all batches complete.")
            break

        batch_doc_count = len(docs)
        last_doc_id = str(docs[-1]["_id"])

        # Process all docs in this batch: convert → chunk
        all_chunks = []
        all_texts = []

        for doc in docs:
            doc_id = str(doc.get("_id", ""))
            doc_created_at = doc.get("created_at")

            text = converter.convert(doc)
            if not text.strip():
                continue

            doc_chunks = chunker.chunk_document(text, COLLECTION_NAME, doc_id)
            for chunk in doc_chunks:
                chunk._source_created_at = doc_created_at
                all_chunks.append(chunk)
                all_texts.append(chunk.text)

        # Embed all chunks in this batch (sub-batched internally by embedder)
        if all_texts:
            embeddings = embedder.embed_texts(all_texts)
            batch_to_store = []
            for chunk, emb in zip(all_chunks, embeddings):
                if emb is None:
                    embed_failures += 1
                    continue
                batch_to_store.append({
                    "text": chunk.text,
                    "embedding": emb,
                    "metadata": {
                        "source_collection": chunk.metadata.source_collection,
                        "document_id": chunk.metadata.document_id,
                        "chunk_index": chunk.metadata.chunk_index,
                        "total_chunks": chunk.metadata.total_chunks,
                        "source_created_at": getattr(chunk, '_source_created_at', None),
                    },
                })
            if batch_to_store:
                written = store.upsert_chunks(batch_to_store)
                chunks_stored += written

        docs_processed += batch_doc_count

        # Save checkpoint after each successful batch
        _save_checkpoint(batch_num, last_doc_id, docs_processed, chunks_stored)

        pbar.update(batch_doc_count)
        pbar.set_postfix(
            batch=f"{batch_num}/{total_batches}",
            stored=chunks_stored,
            failures=embed_failures,
        )

        # Move to next batch
        after_id = last_doc_id
        batch_num += 1

    pbar.close()
    store.close()

    # Clear checkpoint on successful completion
    _clear_checkpoint()

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Batches processed   : %d", batch_num - start_batch)
    logger.info("  Documents processed : %d", docs_processed)
    logger.info("  Chunks stored       : %d", chunks_stored)
    logger.info("  Embed failures      : %d", embed_failures)
    logger.info("  Total in vector DB  : %d", VectorStore(MONGODB_URI, DB_NAME, VECTOR_COLLECTION).total_chunks())
    logger.info("=" * 60)

    # Rebuild search cache after ingestion
    logger.info("Rebuilding vector search cache...")
    cache_store = VectorStore(MONGODB_URI, DB_NAME, VECTOR_COLLECTION)
    cache_store.refresh_cache()
    cache_store.close()
    logger.info("Search cache ready — queries will be fast now.")


# ---------------------------------------------------------------------------
# Query (stage 6)
# ---------------------------------------------------------------------------

def run_query(question: str):
    """Ask the AI assistant a question."""
    bot = Assistant(
        ollama_base_url=OLLAMA_BASE_URL,
        llm_model=OLLAMA_LLM_MODEL,
        embed_model=OLLAMA_EMBED_MODEL,
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME,
        vector_collection=VECTOR_COLLECTION,
        top_k=TOP_K,
    )

    result = bot.ask(question)
    bot.close()

    print("\n" + "=" * 60)
    print("QUESTION:", result["question"])
    print("=" * 60)
    print("\nANSWER:\n")
    print(result["answer"])

    if result["sources"]:
        print("\n" + "-" * 60)
        print("SOURCES (top %d chunks used):\n" % len(result["sources"]))
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] score={src['score']}  collection={src['collection']}  "
                  f"doc={src['document_id']}  chunk={src['chunk_index']}")
            print(f"      {src['preview']}\n")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def run_stats():
    """Print quick stats about source and vector collections."""
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]

    src_count = db[COLLECTION_NAME].estimated_document_count()
    vec_count = db[VECTOR_COLLECTION].estimated_document_count() if VECTOR_COLLECTION in db.list_collection_names() else 0
    unique_docs = len(db[VECTOR_COLLECTION].distinct("metadata.document_id")) if vec_count else 0

    client.close()

    print(f"\nSource collection '{COLLECTION_NAME}': ~{src_count} documents")
    print(f"Vector collection '{VECTOR_COLLECTION}': {vec_count} chunks from {unique_docs} unique documents\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Ingest MongoDB data and query with Ollama Qwen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python pipeline.py --check
              python pipeline.py --ingest
              python pipeline.py --query "How many users signed up last month?"
              python pipeline.py --stats
        """),
    )
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline (stages 1–5)")
    parser.add_argument("--full", action="store_true", help="Force full re-ingestion (skip incremental mode)")
    parser.add_argument("--query", type=str, metavar="QUESTION", help="Ask a question (stage 6)")
    parser.add_argument("--check", action="store_true", help="Health-check MongoDB + Ollama")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    parser.add_argument("--build-cache", action="store_true", help="Build/rebuild the vector search cache")

    args = parser.parse_args()

    if not any([args.ingest, args.query, args.check, args.stats, args.build_cache]):
        parser.print_help()
        sys.exit(0)

    if args.check:
        healthy = run_health_check()
        sys.exit(0 if healthy else 1)

    if args.stats:
        run_stats()

    if args.build_cache:
        logger.info("Building vector search cache...")
        store = VectorStore(MONGODB_URI, DB_NAME, VECTOR_COLLECTION)
        store.refresh_cache()
        store.close()
        logger.info("Cache ready.")

    if args.ingest:
        run_ingestion(full=args.full)

    if args.query:
        run_query(args.query)


if __name__ == "__main__":
    main()
