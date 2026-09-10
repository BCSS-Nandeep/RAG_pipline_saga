"""
api_server.py — FastAPI bridge for the RAG pipeline.

Exposes REST endpoints so the Node.js backend (or any client) can:
  - GET  /api/rag/health          → pipeline + Ollama health check
  - GET  /api/rag/collections     → list all MongoDB collections
  - POST /api/rag/query           → synchronous question (blocks until answer)
  - POST /api/rag/query/async     → enqueue question, returns {job_id} immediately
  - GET  /api/rag/jobs/{job_id}   → poll status/result of an async job
  - GET  /api/rag/jobs            → list recent jobs (optionally by collection)
  - POST /api/rag/ingest          → trigger ingestion for a collection
  - GET  /api/rag/stats           → vector store stats

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8100 --reload
"""

import logging
import os
import re
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
from starlette.middleware.wsgi import WSGIMiddleware

from assistant import Assistant, SYSTEM_PROMPT, CONTEXT_SEPARATOR, _smalltalk_response
from embedder import OllamaEmbedder
from osint_portal.app import app as osint_portal_app
from processor import MongoStreamProcessor, DocumentConverter
from chunker import TokenAwareChunker
from vector_store import VectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Smaller model used as a last-resort fallback when the primary model fails
# with an out-of-memory 500 from Ollama. Pull it once on the Ollama host:
#   ollama pull qwen2.5:1.5b
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:1.5b")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "vector_embeddings")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
CHUNK_MIN = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

# --- Scheduler ---
INGEST_INTERVAL_HOURS = float(os.getenv("INGEST_INTERVAL_HOURS", "6"))
INGEST_COLLECTIONS = [
    c.strip() for c in os.getenv("INGEST_COLLECTIONS", "contents,users").split(",") if c.strip()
]
SCHEDULER_ENABLED = os.getenv("INGEST_SCHEDULER_ENABLED", "true").lower() in ("1", "true", "yes")
INGEST_RUNS_COLLECTION = "rag_ingest_runs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/osint", WSGIMiddleware(osint_portal_app))

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    collection: str | None = None
    top_k: int = 12
    # If set, restrict retrieval to source-doc ids whose timestamp falls within
    # the last N days. Default 7. Set to 0 / None to disable the window.
    time_window_days: int | None = 7
    # When False, skip MongoDB + vector retrieval entirely and answer the
    # question as a pure conversational LLM (general knowledge, casual chat).
    # The frontend toggles this with a "Use database" checkbox.
    use_db: bool = True


class IngestRequest(BaseModel):
    collection: str


# Common timestamp field names found across the BluraSaga collections.
TIMESTAMP_FIELDS = (
    "createdAt", "created_at", "publishedAt", "published_at",
    "timestamp", "ts", "date", "scrapedAt", "fetchedAt", "postedAt",
)


def _allow_ids_within_window(collection: str, days: int) -> Optional[set]:
    """Return a set of stringified _id values whose timestamp is within the
    last *days* days. Returns None if no timestamp field can be detected
    (caller should treat None as "no filter")."""
    if not days or days <= 0:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        if collection not in db.list_collection_names():
            client.close()
            return set()
        col = db[collection]
        sample = col.find_one({}, sort=[("_id", DESCENDING)]) or {}
        ts_field = next((f for f in TIMESTAMP_FIELDS if f in sample), None)
        if not ts_field:
            client.close()
            return None  # cannot filter — fall back to no-window
        cur = col.find({ts_field: {"$gte": cutoff}}, {"_id": 1}).limit(50000)
        ids = {str(d["_id"]) for d in cur}
        client.close()
        return ids
    except Exception as e:
        logger.warning("time-window filter failed for '%s': %s", collection, e)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/rag/health")
def health():
    """Check MongoDB and Ollama connectivity."""
    status = {"mongodb": False, "ollama_embed": False, "ollama_llm": False}
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        status["mongodb"] = True
        client.close()
    except Exception as e:
        logger.error("MongoDB health check failed: %s", e)

    embedder = OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)
    status["ollama_embed"] = embedder.check_health()

    import requests as req
    try:
        resp = req.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        status["ollama_llm"] = any(OLLAMA_LLM_MODEL in m for m in models)
    except Exception:
        pass

    overall = all(status.values())
    return {"healthy": overall, "services": status}


@app.get("/api/rag/collections")
def list_collections():
    """Return all collection names in the configured database."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        names = db.list_collection_names()
        client.close()
        # Filter out system and vector collections
        skip = {"system.profile", "system.js", VECTOR_COLLECTION}
        collections = sorted([n for n in names if n not in skip])
        return {"database": DB_NAME, "collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_GLOBAL_STORES: dict = {}
_GLOBAL_STORES_LOCK = threading.Lock()


def _get_store(vec_col: str) -> VectorStore:
    with _GLOBAL_STORES_LOCK:
        s = _GLOBAL_STORES.get(vec_col)
        if s is None:
            s = VectorStore(uri=MONGODB_URI, db_name=DB_NAME, collection_name=vec_col)
            _GLOBAL_STORES[vec_col] = s
        return s


def _list_vector_collections() -> list:
    """Return all vector collections the chatbot is permitted to search.

    Officers ask about every module in the portal — events, alerts, grievances,
    Dial 100 calls, POIs, monitored profiles, keywords, contents, daily
    programmes, telegram messages, and the reporting collections — so the
    allow-list is intentionally broad. Override via ALLOWED_QUERY_COLLECTIONS.
    """
    allowed = {f"{VECTOR_COLLECTION}_{c}" for c in ALLOWED_QUERY_COLLECTIONS}
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    existing = set(db.list_collection_names())
    client.close()
    return [c for c in allowed if c in existing]


_COUNT_KEYWORDS = {
    "alerts":                      ["alert", "alerts"],
    "grievances":                  ["grievance", "grievances", "greivance", "greivances",
                                    "complaint", "complaints"],
    "events":                      ["event", "events", "festival", "festivals", "rally", "rallies",
                                    "protest", "protests", "procession", "processions"],
    "dial100incidents":            ["dial 100", "dial-100", "dial100", "100 call", "100 calls",
                                    "emergency call", "emergency calls", "incident", "incidents"],
    "pois":                        ["poi", "pois", "person of interest", "persons of interest",
                                    "suspect", "suspects", "history sheeter", "history-sheeter",
                                    "history sheeters", "accused"],
    "keywords":                    ["keyword", "keywords", "watch word", "watch words",
                                    "watchword", "watchwords", "monitored word", "monitored words",
                                    "top keyword", "top keywords"],
    "sources":                     ["source", "sources", "monitored profile", "monitored profiles",
                                    "monitored account", "monitored accounts",
                                    "tracked account", "tracked accounts", "tracked profile",
                                    "tracked profiles"],
    "contents":                    ["content", "contents", "post", "posts", "tweet", "tweets",
                                    "reel", "reels", "story", "stories", "video", "videos"],
    "dailyprogrammes":             ["programme", "programmes", "program", "programs",
                                    "daily programme", "daily programmes", "schedule", "schedules"],
    "telegrammessages":            ["telegram", "telegram message", "telegram messages",
                                    "tg message", "tg messages", "telegram group", "telegram channel"],
    "criticism_reports":           ["criticism", "critique", "critisism", "critisisum",
                                    "criticsm", "critisim", "criticisim", "criticisms",
                                    "criticism report", "criticism reports"],
    "grievance_workflow_reports":  ["workflow report", "grievance workflow"],
    "query_reports":               ["query report", "query reports"],
    "suggestion_reports":          ["suggestion report", "suggestion reports"],
}

_COUNT_TIME_PATTERNS = [
    (re.compile(r"past\s+(\d+)\s*(hour|hr|h)s?", re.I),  lambda m: timedelta(hours=int(m.group(1)))),
    (re.compile(r"last\s+(\d+)\s*(hour|hr|h)s?", re.I),  lambda m: timedelta(hours=int(m.group(1)))),
    (re.compile(r"past\s+(\d+)\s*(day|d)s?", re.I),      lambda m: timedelta(days=int(m.group(1)))),
    (re.compile(r"last\s+(\d+)\s*(day|d)s?", re.I),      lambda m: timedelta(days=int(m.group(1)))),
    (re.compile(r"past\s+(\d+)\s*(week|wk)s?", re.I),    lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"last\s+(\d+)\s*(week|wk)s?", re.I),    lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"yesterday", re.I),                     lambda m: timedelta(days=1)),
    (re.compile(r"today", re.I),                         lambda m: timedelta(hours=24)),
    (re.compile(r"this\s+week", re.I),                   lambda m: timedelta(days=7)),
]


def _count_fast_path(question: str, default_window_days: Optional[int]) -> Optional[dict]:
    """Detect 'how many <thing> in last N days/hours' and answer with a real Mongo count.
    Returns None if the question doesn't fit the count template."""
    q = question.lower().strip()
    if not re.search(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b|\btotal\s+(number\s+of\s+)?\b", q):
        return None
    target_col = None
    matched_kw = None
    for col, kws in _COUNT_KEYWORDS.items():
        for kw in sorted(kws, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", q):
                target_col = col; matched_kw = kw; break
        if target_col: break
    if not target_col:
        return None

    delta = None
    matched_phrase = None
    for pat, mk in _COUNT_TIME_PATTERNS:
        m = pat.search(q)
        if m:
            delta = mk(m); matched_phrase = m.group(0); break
    if delta is None and default_window_days and default_window_days > 0:
        delta = timedelta(days=default_window_days)
        matched_phrase = f"last {default_window_days} day(s)"

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        if target_col not in db.list_collection_names():
            client.close()
            return {"answer": f"The `{target_col}` collection isn't present in the database.",
                    "sources": [], "question": question, "scope": "count"}
        col = db[target_col]
        total = col.estimated_document_count()
        if delta is None:
            client.close()
            return {
                "answer": f"**{total:,}** {matched_kw} in total (all time) in `{target_col}`.",
                "sources": [], "question": question, "scope": "count",
                "count": total, "collection": target_col,
            }
        cutoff = datetime.now(timezone.utc) - delta
        sample = col.find_one({}, sort=[("_id", DESCENDING)]) or {}
        ts_field = next((f for f in TIMESTAMP_FIELDS if f in sample), None)
        if not ts_field:
            client.close()
            return None  # fall back to vector search
        n = col.count_documents({ts_field: {"$gte": cutoff}})
        client.close()
        nice_window = matched_phrase
        return {
            "answer": (
                f"**Bottom line:** {n:,} {matched_kw} in `{target_col}` over the {nice_window}.\n\n"
                f"_(Counted via field `{ts_field}`, cutoff {cutoff.isoformat(timespec='minutes')} UTC.)_"
            ),
            "sources": [], "question": question, "scope": "count",
            "count": n, "collection": target_col, "window": nice_window,
        }
    except Exception as e:
        logger.warning("count fast-path failed: %s", e)
        return None


def _shrink_prompt(prompt: str, target_ctx: int) -> str:
    """Trim the middle of an oversized prompt so it fits a smaller context."""
    approx_chars = max(2000, target_ctx * 3)  # ~3 chars per token, conservative
    if len(prompt) <= approx_chars:
        return prompt
    head = prompt[: approx_chars // 2]
    tail = prompt[-approx_chars // 2:]
    return head + "\n\n[…context truncated for retry…]\n\n" + tail


def _llm_answer(prompt: str) -> str:
    """Call Ollama /api/generate with retry-on-5xx and prompt-shrink fallback.

    Tries 3 times: (16k ctx / 2048 out), then (12k / 1600), then (8k / 1200).
    Sleeps with exponential backoff between retries. Returns the model's text
    on success, or a clearly-labelled error string on terminal failure (so the
    caller can still surface the retrieved evidence to the user).
    """
    import time as _time

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    # Each attempt = (model, num_ctx, num_predict). Later attempts use smaller
    # context AND, eventually, the small fallback model — to survive an OOM
    # host where the primary 7B model can't be loaded.
    attempts = [
        (OLLAMA_LLM_MODEL,      16384, 900),
        (OLLAMA_LLM_MODEL,      12288, 1600),
        (OLLAMA_LLM_MODEL,       8192, 1200),
        (OLLAMA_FALLBACK_MODEL,  8192, 1200),
        (OLLAMA_FALLBACK_MODEL,  4096,  900),
    ]
    last_err: Optional[str] = None
    oom_seen = False
    for i, (model, ctx, predict) in enumerate(attempts):
        # Skip the duplicate fallback if it's the same as the primary
        if model == OLLAMA_LLM_MODEL and i >= 3:
            continue
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt if i == 0 else _shrink_prompt(prompt, ctx),
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "num_ctx": ctx,
                        "num_predict": predict,
                    },
                    "keep_alive": "10m",
                },
                timeout=420,
            )
            if resp.status_code in (500, 502, 503, 504):
                body = ""
                try:
                    body = resp.json().get("error", "") or ""
                except Exception:
                    body = (resp.text or "")[:200]
                logger.warning("Ollama %s on attempt %d (model=%s): %s",
                               resp.status_code, i + 1, model, body)
                last_err = f"{resp.status_code} — {body or resp.reason}"
                if "memory" in body.lower() or "oom" in body.lower():
                    oom_seen = True
                _time.sleep(1.5 * (i + 1))
                continue
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
            if text:
                return text
            logger.warning("Ollama returned empty response on attempt %d", i + 1)
            last_err = "empty response"
        except requests.ConnectionError as e:
            logger.error("Ollama connection error: %s", e)
            return f"Error generating answer: Ollama unreachable at {OLLAMA_BASE_URL} ({e})."
        except requests.Timeout as e:
            logger.warning("Ollama timeout on attempt %d", i + 1)
            last_err = f"timeout ({e})"
        except Exception as e:
            logger.warning("Ollama attempt %d failed: %s", i + 1, e)
            last_err = str(e)
            _time.sleep(1.0 * (i + 1))

    if oom_seen:
        return (
            "_(The Ollama host is out of memory — the model couldn't be loaded. "
            f"Free RAM on `{OLLAMA_BASE_URL}` or pull a smaller model "
            f"(`ollama pull {OLLAMA_FALLBACK_MODEL}`). "
            "Live evidence from MongoDB is shown below.)_"
        )
    return (
        f"_(LLM generation failed after {len(attempts)} attempts: {last_err}. "
        "Live evidence from MongoDB is shown below — please retry.)_"
    )


_URL_RE = re.compile(r"https?://[^\s)\]]+")


CHAT_ONLY_SYSTEM_PROMPT = textwrap.dedent("""\
    You are SOC-EYE, a friendly and highly capable AI assistant — versatile like
    Claude or ChatGPT. In this mode you are NOT querying any internal database.
    Answer the user from general knowledge, help with casual conversation,
    explain concepts, draft text, do reasoning, write code, summarise topics,
    and provide opinions when asked.

    Style guide:
      • Be warm and natural for casual chat ("hi", "thanks", "how are you").
      • Be detailed and well-structured for substantive questions — use
        headings, bullets, code blocks, and examples where they help.
      • Use Markdown formatting. Bold key terms. Code-format `commands`.
      • If the user asks something that would clearly benefit from the live
        Telangana Police database (specific alerts, grievances, POIs, recent
        events, monitored handles, Dial-100 calls), gently note:
        _"Tip: enable 'Use database' to query the live SOC-EYE data."_
      • Never claim to have looked up live data in this mode — you haven't.
      • Knowledge cutoff applies; flag uncertainty rather than inventing facts.
      • Aim for thorough answers (10+ lines) on real questions; keep small-talk
        replies short and friendly.
""")


def _chat_only_answer(question: str) -> dict:
    """Pure LLM call — no DB, no vector search. For casual / general questions."""
    smalltalk = _smalltalk_response(question)
    if smalltalk is not None:
        return {"answer": smalltalk, "sources": [], "question": question,
                "smalltalk": True, "scope": "chat_only"}
    prompt = (
        f"{CHAT_ONLY_SYSTEM_PROMPT}\n\n"
        f"User: {question}\n\n"
        f"Assistant:"
    )
    answer = _llm_answer(prompt)
    return {"answer": answer, "sources": [], "question": question,
            "scope": "chat_only", "use_db": False}


def _ensure_minimum_answer(answer: str, snippets: list, question: str) -> str:
    """Guarantee the user always sees ≥10 lines of useful content with links.

    If the LLM returned a short answer, an error message, or omitted links,
    append a deterministic 'Evidence' section built from the retrieved
    snippets so officers always have the URLs to act on.
    """
    answer = (answer or "").strip()
    line_count = len([ln for ln in answer.splitlines() if ln.strip()])
    has_link = bool(_URL_RE.search(answer)) or "](http" in answer
    needs_evidence = (
        line_count < 10
        or not has_link
        or answer.startswith("_(LLM generation failed")
        or "Error generating answer" in answer
    )
    if not needs_evidence or not snippets:
        return answer

    evidence_lines = ["", "---", "", "### Evidence from live database"]
    for s in snippets[:12]:
        # Pull the first URL from the snippet, if any
        m = _URL_RE.search(s)
        url = m.group(0) if m else ""
        # Take just the header line (before the first newline) for the bullet
        head = s.split("\n", 1)[0].strip()
        if url:
            evidence_lines.append(f"- {head} — [Open]({url})")
        else:
            evidence_lines.append(f"- {head}")
    evidence_lines.append("")
    evidence_lines.append(
        "_Tip: try a more specific question (handle, date range, district) "
        "for a sharper briefing._"
    )
    return answer + "\n" + "\n".join(evidence_lines)


# ---------------------------------------------------------------------------
# Universal DB context builder
#
# For EVERY non-count question we pull a rich context directly from Mongo
# (alerts + grievances), apply the time window, then optionally enrich with
# vector-search results if embeddings exist.  This means the bot always has
# real data regardless of whether ingestion has caught up.
# ---------------------------------------------------------------------------

ALERT_FIELDS = {
    "_id": 1, "id": 1, "title": 1, "platform": 1, "author": 1, "author_handle": 1,
    "content_url": 1, "risk_level": 1, "priority": 1, "alert_type": 1,
    "source_category": 1, "legal_sections": 1, "violated_policies": 1,
    "matched_keywords_normalized": 1, "classification_explanation": 1,
    "velocity_data": 1, "llm_analysis": 1, "threat_details": 1,
    "content_id": 1, "created_at": 1,
}
GRIEVANCE_FIELDS = {
    "_id": 1, "complaint_code": 1, "platform": 1, "posted_by": 1,
    "content": 1, "context": 1, "tagged_account": 1, "status": 1,
    "priority": 1, "created_at": 1,
}


def _safe_join(items) -> str:
    """Join a list that may contain strings or dicts (extract meaningful text)."""
    if not items:
        return "—"
    parts = []
    for it in items:
        if isinstance(it, str):
            parts.append(it)
        elif isinstance(it, dict):
            # common keys: 'section', 'name', 'description', 'policy', 'value'
            val = it.get("section") or it.get("name") or it.get("policy") or it.get("description") or str(it)
            parts.append(str(val))
        else:
            parts.append(str(it))
    return ", ".join(parts) if parts else "—"


def _fmt_alert(d: dict, idx: int) -> str:
    """Format an alert document into a rich text snippet for the LLM."""
    ts = d.get("created_at")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    handle = (d.get("author_handle") or d.get("author") or "?").lstrip("@")
    vdata = d.get("velocity_data") or {}
    llm = d.get("llm_analysis") or {}
    threat = d.get("threat_details") or {}
    legal = _safe_join(d.get("legal_sections"))
    policies = _safe_join(d.get("violated_policies"))
    kw = _safe_join(d.get("matched_keywords_normalized"))
    # Use the real AI reasoning if available
    reasoning = (d.get("classification_explanation") or llm.get("reasoning") or "").strip()
    if "Primary AI analysis unavailable" in reasoning:
        reasoning = ""
    risk_score = threat.get("risk_score") or llm.get("score") or "?"
    velocity_info = (
        f"Viral: {vdata.get('metric','?')} velocity={vdata.get('velocity','?')} "
        f"(threshold={vdata.get('threshold_triggered','?')}, window={vdata.get('time_window_minutes','?')}min)"
        if vdata else ""
    )
    return (
        f"[ALERT {idx} | {d.get('priority','?')}-risk | cat={d.get('source_category','?')} | "
        f"type={d.get('alert_type','?')} | {ts_s}]\n"
        f"  Author: @{handle} on {d.get('platform','?')}\n"
        f"  URL: {d.get('content_url','—')}\n"
        f"  Risk Score: {risk_score}% | Sentiment: {llm.get('sentiment','?')} | "
        f"Intent: {llm.get('intent') or llm.get('category','?')}\n"
        + (f"  {velocity_info}\n" if velocity_info else "")
        + (f"  Analysis: {reasoning[:300]}\n" if reasoning else "")
        + f"  Keywords: {kw} | Policies: {policies} | Legal sections: {legal}"
    )


def _fmt_grievance(d: dict, idx: int) -> str:
    """Format a grievance document — includes the actual tweet text."""
    ts = d.get("created_at")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    pb = d.get("posted_by") or {}
    handle = pb.get("handle") or "?"
    followers = pb.get("follower_count", "?")
    tweet_text = (d.get("content") or {}).get("text") or "—"
    # Original tweet this is replying to
    parent = ((d.get("context") or {}).get("in_reply_to") or {})
    parent_handle = (parent.get("posted_by") or {}).get("handle") or ""
    parent_text = (parent.get("content") or {}).get("text") or ""
    return (
        f"[GRIEVANCE {idx} | code={d.get('complaint_code','?')} | "
        f"status={d.get('status','?')} | {ts_s}]\n"
        f"  Filed by: @{handle} (followers={followers}) on {d.get('platform','?')}\n"
        f"  Tagged: {d.get('tagged_account','—')}\n"
        f"  Tweet: \"{tweet_text[:250]}\"\n"
        + (f"  Replying to @{parent_handle}: \"{parent_text[:200]}\"\n" if parent_text else "")
    )

# Keyword → Mongo field:value filters for smarter retrieval
_CATEGORY_HINTS = [
    (re.compile(r"\bcommunal\b", re.I),          {"source_category": "communal"}),
    (re.compile(r"\bhate.speech\b", re.I),        {"alert_type": {"$regex": "hate", "$options": "i"}}),
    (re.compile(r"\bviral\b", re.I),              {"risk_level": "high"}),
    (re.compile(r"\bviolence\b", re.I),           {"source_category": {"$regex": "violen", "$options": "i"}}),
    (re.compile(r"\bfake.news|misinform\b", re.I),{"alert_type": {"$regex": "fake|misinfo", "$options": "i"}}),
    (re.compile(r"\bfir\b|\blegal\b|\bbnS\b", re.I), {"legal_sections": {"$ne": []}}),
    (re.compile(r"\bopen|pending\b", re.I),       {"status": {"$in": ["open", "pending"]}}),
    (re.compile(r"\bhigh.?risk|urgent|critical\b", re.I), {"priority": "HIGH"}),
    (re.compile(r"\bmedium\b", re.I),             {"priority": "MEDIUM"}),
]

# ---------------------------------------------------------------------------
# Field projections & formatters for ALL additional modules
# ---------------------------------------------------------------------------

CONTENT_FIELDS = {
    "_id": 1, "platform": 1, "content_type": 1, "content_url": 1,
    "text": 1, "author": 1, "author_handle": 1, "risk_score": 1,
    "risk_level": 1, "sentiment": 1, "engagement": 1, "published_at": 1,
    "threat_intent": 1, "threat_reasons": 1, "event_ids": 1,
}
DIAL100_FIELDS = {
    "_id": 1, "date": 1, "category": 1, "incidentDetails": 1,
    "incidentCategory": 1, "location": 1, "psJurisdiction": 1,
    "zoneJurisdiction": 1, "callerName": 1, "status": 1, "priority": 1,
    "remarks": 1, "pcRemarks": 1, "shoRemarks": 1, "createdAt": 1,
}
EVENT_FIELDS = {
    "_id": 1, "name": 1, "description": 1, "start_date": 1, "end_date": 1,
    "location": 1, "keywords": 1, "platforms": 1, "status": 1, "created_at": 1,
}
POI_FIELDS = {
    "_id": 1, "name": 1, "realName": 1, "aliasNames": 1,
    "mobileNumbers": 1, "currentAddress": 1,
    "psLimits": 1, "districtCommisionerate": 1, "firDetails": 1,
    "linkedIncidents": 1, "created_at": 1,
}
KEYWORD_FIELDS = {
    "_id": 1, "keyword": 1, "category": 1, "language": 1,
    "is_active": 1, "weight": 1, "created_at": 1,
}
SOURCE_FIELDS = {
    "_id": 1, "platform": 1, "identifier": 1, "display_name": 1,
    "category": 1, "is_active": 1, "risk_level": 1, "created_at": 1,
    "follower_count": 1, "profile_image_url": 1, "is_verified": 1,
    "platform_user_id": 1,
}


def _build_profile_url(platform: str, identifier: str, platform_user_id: str = "") -> str:
    """Build a public profile URL from a monitored source's platform + handle."""
    if not identifier:
        return ""
    handle = str(identifier).lstrip("@").strip()
    if not handle:
        return ""
    p = (platform or "").lower()
    if p == "x" or p == "twitter":
        return f"https://x.com/{handle}"
    if p == "instagram":
        return f"https://www.instagram.com/{handle}/"
    if p == "facebook":
        return f"https://www.facebook.com/{handle}"
    if p == "youtube":
        # YouTube channels can be addressed via @handle (newer) or channel ID
        if platform_user_id and platform_user_id.startswith("UC"):
            return f"https://www.youtube.com/channel/{platform_user_id}"
        return f"https://www.youtube.com/@{handle}"
    return ""
DAILY_PROGRAMME_FIELDS = {
    "_id": 1, "date": 1, "category": 1, "categoryLabel": 1,
    "programName": 1, "location": 1, "organizer": 1,
    "expectedMembers": 1, "zone": 1,
}
TELEGRAM_FIELDS = {
    "_id": 1, "text": 1, "sender_name": 1, "sender_username": 1,
    "date": 1, "group_id": 1, "links": 1,
}
CRITICISM_REPORT_FIELDS = {
    "_id": 1, "unique_code": 1, "platform": 1, "post_link": 1,
    "post_description": 1, "post_date": 1, "posted_by": 1,
    "category": 1, "remarks": 1, "status": 1, "createdAt": 1,
}
SUGGESTION_REPORT_FIELDS = {
    "_id": 1, "unique_code": 1, "platform": 1, "post_link": 1,
    "post_description": 1, "post_date": 1, "posted_by": 1,
    "category": 1, "remarks": 1, "status": 1, "createdAt": 1,
}


def _fmt_content(d: dict, idx: int) -> str:
    ts = d.get("published_at") or d.get("created_at")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    eng = d.get("engagement") or {}
    text = (d.get("text") or "")[:250]
    return (
        f"[CONTENT {idx} | {d.get('platform','?')} | type={d.get('content_type','?')} | "
        f"risk={d.get('risk_level','?')} | {ts_s}]\n"
        f"  Author: @{d.get('author_handle','?')} | URL: {d.get('content_url','—')}\n"
        f"  Text: \"{text}\"\n"
        f"  Engagement: views={eng.get('views',0)} likes={eng.get('likes',0)} "
        f"comments={eng.get('comments',0)} retweets={eng.get('retweets',0)} | "
        f"Sentiment: {d.get('sentiment','?')} | Risk Score: {d.get('risk_score',0)}%"
    )


def _fmt_dial100(d: dict, idx: int) -> str:
    ts = d.get("date") or d.get("createdAt")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    return (
        f"[DIAL-100 CALL {idx} | cat={d.get('category','?')} | "
        f"incident={d.get('incidentCategory','?')} | status={d.get('status','?')} | {ts_s}]\n"
        f"  Location: {d.get('location','?')} | PS: {d.get('psJurisdiction','?')} "
        f"| Zone: {d.get('zoneJurisdiction','?')}\n"
        f"  Details: {(d.get('incidentDetails') or '')[:250]}\n"
        f"  Priority: {d.get('priority','?')} | Caller: {d.get('callerName','?')}"
        + (f"\n  Remarks: {d.get('remarks','')[:200]}" if d.get('remarks') else "")
    )


def _fmt_event(d: dict, idx: int) -> str:
    start = d.get("start_date")
    end = d.get("end_date")
    start_s = start.strftime("%d-%b %Y") if isinstance(start, datetime) else "?"
    end_s = end.strftime("%d-%b %Y") if isinstance(end, datetime) else "?"
    kws = ", ".join(k.get("keyword", "") for k in (d.get("keywords") or [])[:5])
    return (
        f"[EVENT {idx} | status={d.get('status','?')} | {start_s} to {end_s}]\n"
        f"  Name: {d.get('name','?')}\n"
        f"  Location: {d.get('location','?')} | Platforms: {', '.join(d.get('platforms',[]))}\n"
        f"  Keywords: {kws or '—'}\n"
        f"  Description: {(d.get('description') or '')[:200]}"
    )


def _fmt_poi(d: dict, idx: int) -> str:
    firs = "; ".join(
        f"FIR {f.get('firNo','?')} at {f.get('psLimits','?')}"
        for f in (d.get("firDetails") or [])[:3]
    )
    aliases = ", ".join(d.get("aliasNames") or [])
    return (
        f"[PERSON OF INTEREST {idx}]\n"
        f"  Name: {d.get('name','?')} | Real Name: {d.get('realName','?')}\n"
        f"  Aliases: {aliases or '—'}\n"
        f"  Address: {d.get('currentAddress','?')} | PS: {d.get('psLimits','?')} "
        f"| District: {d.get('districtCommisionerate','?')}\n"
        f"  FIRs: {firs or '—'}\n"
        f"  Linked Incidents: {(d.get('linkedIncidents') or '')[:200]}"
    )


def _fmt_keyword(d: dict, idx: int) -> str:
    return (
        f"[KEYWORD {idx}] \"{d.get('keyword','?')}\" | category={d.get('category','?')} "
        f"| lang={d.get('language','?')} | weight={d.get('weight',0)} "
        f"| active={'yes' if d.get('is_active') else 'no'}"
    )


def _fmt_source(d: dict, idx: int) -> str:
    platform = d.get("platform", "?")
    identifier = d.get("identifier", "?")
    profile_url = _build_profile_url(platform, identifier, d.get("platform_user_id", ""))
    return (
        f"[MONITORED PROFILE {idx}] @{identifier} "
        f"({d.get('display_name','?')}) | platform={platform} "
        f"| category={d.get('category','?')} | risk={d.get('risk_level','?')} "
        f"| followers={d.get('follower_count') or '?'} "
        f"| verified={'yes' if d.get('is_verified') else 'no'}\n"
        f"  Profile URL: {profile_url or 'N/A'}"
    )


def _fmt_daily_programme(d: dict, idx: int) -> str:
    dt = d.get("date")
    dt_s = dt.strftime("%d-%b %Y") if isinstance(dt, datetime) else "?"
    return (
        f"[DAILY PROGRAMME {idx} | {dt_s}]\n"
        f"  Programme: {d.get('programName','?')} | Category: {d.get('categoryLabel') or d.get('category','?')}\n"
        f"  Location: {d.get('location','?')} | Zone: {d.get('zone','?')}\n"
        f"  Organizer: {d.get('organizer','?')} | Expected: {d.get('expectedMembers',0)} members"
    )


def _fmt_telegram(d: dict, idx: int) -> str:
    ts = d.get("date")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    return (
        f"[TELEGRAM MSG {idx} | group={d.get('group_id','?')} | {ts_s}]\n"
        f"  Sender: {d.get('sender_name','?')} (@{d.get('sender_username','?')})\n"
        f"  Text: \"{(d.get('text') or '')[:250]}\""
        + (f"\n  Links: {', '.join(d.get('links',[])[:3])}" if d.get('links') else "")
    )


def _fmt_criticism_report(d: dict, idx: int) -> str:
    ts = d.get("post_date") or d.get("createdAt")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    pb = d.get("posted_by") or {}
    return (
        f"[CRITICISM REPORT {idx} | code={d.get('unique_code','?')} | "
        f"status={d.get('status','?')} | {ts_s}]\n"
        f"  By: @{pb.get('handle','?')} on {d.get('platform','?')} | "
        f"Link: {d.get('post_link','—')}\n"
        f"  Category: {d.get('category','?')}\n"
        f"  Description: {(d.get('post_description') or '')[:200]}"
        + (f"\n  Remarks: {d.get('remarks','')[:150]}" if d.get('remarks') else "")
    )


def _fmt_suggestion_report(d: dict, idx: int) -> str:
    ts = d.get("post_date") or d.get("createdAt")
    ts_s = ts.strftime("%d-%b %H:%M IST") if isinstance(ts, datetime) else "?"
    pb = d.get("posted_by") or {}
    return (
        f"[SUGGESTION REPORT {idx} | code={d.get('unique_code','?')} | "
        f"status={d.get('status','?')} | {ts_s}]\n"
        f"  By: @{pb.get('handle','?')} on {d.get('platform','?')} | "
        f"Link: {d.get('post_link','—')}\n"
        f"  Category: {d.get('category','?')}\n"
        f"  Description: {(d.get('post_description') or '')[:200]}"
        + (f"\n  Remarks: {d.get('remarks','')[:150]}" if d.get('remarks') else "")
    )


# ---------------------------------------------------------------------------
# Question-aware collection routing
#
# For every question we ALWAYS query alerts + grievances.  On top of that,
# we detect which extra modules are relevant based on keywords in the question.
# For general questions with no specific module hints we include a broad set.
# ---------------------------------------------------------------------------
_COLLECTION_ROUTING = [
    (re.compile(r"\bdial[\s-]?100\b|\bemergency\s*calls?\b|\b100\s*calls?\b|\bincidents?\b", re.I),
     ["dial100incidents"]),
    (re.compile(r"\bevents?\b|\bfestivals?\b|\brall(y|ies)\b|\bprotests?\b|\bprocessions?\b|\bgatherings?\b|\bhartaals?\b|\bbandhs?\b", re.I),
     ["events"]),
    (re.compile(r"\bpois?\b|\bpersons?\s*of\s*interest\b|\bsuspects?\b|\baccused\b|\bhistory[\s-]?sheeters?\b|\bcriminals?\b", re.I),
     ["pois"]),
    # "profile/profiles" is ambiguous in the portal — it can mean a Person of
    # Interest record OR a monitored social-media account (sources). Pull both.
    (re.compile(r"\bprofiles?\b", re.I),
     ["pois", "sources"]),
    (re.compile(r"\bkeywords?\b|\bmonitored\s*words?\b|\bwatch[\s-]?words?\b|\btracking\s*terms?\b|\btop\s*keywords?\b", re.I),
     ["keywords"]),
    (re.compile(r"\bsources?\b|\bmonitored\s*(accounts?|profiles?)\b|\btracked\s*(accounts?|profiles?)\b", re.I),
     ["sources"]),
    (re.compile(r"\bprogrammes?\b|\bprograms?\b|\bschedules?\b|\bdaily\s*programmes?\b|\bannouncements?\b", re.I),
     ["dailyprogrammes"]),
    (re.compile(r"\btelegram\b|\btg\s*messages?\b|\btg\s*groups?\b|\bchannel\s*messages?\b", re.I),
     ["telegrammessages"]),
    (re.compile(r"\bcontents?\b|\bposts?\b|\btweets?\b|\breels?\b|\bstor(y|ies)\b|\bvideos?\b|\bsocial\s*media\b", re.I),
     ["contents"]),
    (re.compile(r"\bcriticisms?\b|\bcritiques?\b|\bcritisisms?\b", re.I),
     ["criticismreports"]),
    (re.compile(r"\bsuggestions?\b", re.I),
     ["suggestionreports"]),
    (re.compile(r"\bquery\s*reports?\b", re.I),
     ["queryreports"]),
    (re.compile(r"\bworkflow\b|\bgrievance\s*workflow\b", re.I),
     ["grievanceworkflowreports"]),
]

# Registry: collection_name → (fields, formatter, ts_field, default_limit)
_EXTRA_COLLECTION_REGISTRY = {
    "contents":              (CONTENT_FIELDS,            _fmt_content,            "published_at", 8),
    "dial100incidents":      (DIAL100_FIELDS,            _fmt_dial100,            "date",         8),
    "events":                (EVENT_FIELDS,              _fmt_event,              "start_date",   6),
    "pois":                  (POI_FIELDS,                _fmt_poi,                None,           6),
    "keywords":              (KEYWORD_FIELDS,            _fmt_keyword,            None,          10),
    "sources":               (SOURCE_FIELDS,             _fmt_source,             None,           8),
    "dailyprogrammes":       (DAILY_PROGRAMME_FIELDS,    _fmt_daily_programme,    "date",         6),
    "telegrammessages":      (TELEGRAM_FIELDS,           _fmt_telegram,           "date",         8),
    "criticismreports":      (CRITICISM_REPORT_FIELDS,   _fmt_criticism_report,   "post_date",    6),
    "suggestionreports":     (SUGGESTION_REPORT_FIELDS,  _fmt_suggestion_report,  "post_date",    6),
}


_BROAD_QUESTION_RE = re.compile(
    r"\b(all|every|everything|overall|summary|summari[sz]e|brief(ing)?|overview|"
    r"status|situation|report|round[\s-]?up|digest|dashboard|across\s+modules?)\b",
    re.I,
)

# Modules pulled by default for any question that doesn't specifically target
# one — covers the operational data officers care about most.
_DEFAULT_EXTRA_COLLECTIONS = [
    "contents", "dial100incidents", "events", "dailyprogrammes",
    "pois", "keywords", "sources",
]

# Every supported extra collection — used when the question is broad
# ("all modules", "overall summary", "everything", etc.).
_ALL_EXTRA_COLLECTIONS = [
    "contents", "dial100incidents", "events", "pois", "keywords", "sources",
    "dailyprogrammes", "telegrammessages",
    "criticismreports", "suggestionreports",
]


def _detect_extra_collections(question: str) -> list:
    """Return list of extra collection names (beyond alerts/grievances) to query."""
    extra = set()
    for pat, cols in _COLLECTION_ROUTING:
        if pat.search(question):
            extra.update(cols)

    # Broad / "everything" style questions → pull from every module.
    if _BROAD_QUESTION_RE.search(question):
        extra.update(_ALL_EXTRA_COLLECTIONS)
        return list(extra)

    # No specific module hint → pull the most operationally relevant modules.
    if not extra:
        extra.update(_DEFAULT_EXTRA_COLLECTIONS)
    return list(extra)


def _build_db_context(question: str, window_days: Optional[int], limit_per: int = 15) -> tuple:
    """
    Pull data from ALL relevant modules in Mongo.
    Returns (snippets_list, total_docs_pulled).
    Always returns data — this is the backbone of every non-count answer.
    """
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        base_q: dict = {}
        if window_days and window_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            base_q["created_at"] = {"$gte": cutoff}

        # Keyword/category hint filter
        extra_q: dict = {}
        for pat, filt in _CATEGORY_HINTS:
            if pat.search(question):
                extra_q.update(filt)
                break

        alert_q = {**base_q, **extra_q}

        # Extract named handles/keywords from question for targeted search
        handle_match = re.findall(r"@([\w]+)", question)
        if handle_match:
            handle_re = "|".join(re.escape(h) for h in handle_match)
            alert_q["$or"] = [
                {"author_handle": {"$regex": handle_re, "$options": "i"}},
                {"author": {"$regex": handle_re, "$options": "i"}},
            ]

        # Detect trending/viral questions — sort by velocity instead of recency
        q_lower = question.lower()
        is_trending = any(w in q_lower for w in ["trending", "viral", "hot topic", "hyped", "most popular", "most talked", "top post"])

        if is_trending:
            # Pull HIGH+MEDIUM together sorted by velocity (most viral first)
            all_alerts = list(db.alerts.find(
                {**alert_q, "priority": {"$in": ["HIGH", "MEDIUM"]},
                 "velocity_data.velocity": {"$gt": 0}},
                ALERT_FIELDS
            ).sort("velocity_data.velocity", DESCENDING).limit(limit_per))
            # If not enough velocity data, fall back to recent high-risk
            if len(all_alerts) < 5:
                all_alerts = list(db.alerts.find(
                    {**alert_q, "priority": {"$in": ["HIGH", "MEDIUM"]}}, ALERT_FIELDS
                ).sort("created_at", DESCENDING).limit(limit_per))
        else:
            # Normal: HIGH first, then MEDIUM — weighted 2:1
            high_lim = max(8, limit_per * 2 // 3)
            med_lim  = max(4, limit_per // 3)
            alerts_high = list(db.alerts.find(
                {**alert_q, "priority": "HIGH"}, ALERT_FIELDS
            ).sort("created_at", DESCENDING).limit(high_lim))
            alerts_med = list(db.alerts.find(
                {**alert_q, "priority": "MEDIUM"}, ALERT_FIELDS
            ).sort("created_at", DESCENDING).limit(med_lim))
            all_alerts = alerts_high + alerts_med

        # Grievances — also try keyword match on content text
        grievance_q = {**base_q}
        if handle_match:
            handle_re = "|".join(re.escape(h) for h in handle_match)
            grievance_q["$or"] = [
                {"posted_by.handle": {"$regex": handle_re, "$options": "i"}},
                {"content.text": {"$regex": handle_re, "$options": "i"}},
                {"tagged_account": {"$regex": handle_re, "$options": "i"}},
            ]
        grievances = list(db.grievances.find(
            grievance_q, GRIEVANCE_FIELDS
        ).sort("created_at", DESCENDING).limit(max(6, limit_per // 2)))

        snippets = []
        for i, d in enumerate(all_alerts, 1):
            snippets.append(_fmt_alert(d, i))
        for j, d in enumerate(grievances, 1):
            snippets.append(_fmt_grievance(d, j))

        total = len(all_alerts) + len(grievances)

        # ── Extra modules (beyond alerts + grievances) ────────────────────────
        extra_cols = _detect_extra_collections(question)
        existing_cols = set(db.list_collection_names())

        for col_name in extra_cols:
            if col_name not in existing_cols:
                continue
            reg = _EXTRA_COLLECTION_REGISTRY.get(col_name)
            if not reg:
                continue
            fields, formatter, ts_field, default_limit = reg

            # Build a time-windowed query for this collection
            col_q: dict = {}
            if window_days and window_days > 0 and ts_field:
                cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
                col_q[ts_field] = {"$gte": cutoff}

            # For text-searchable collections, apply keyword filter if present
            if handle_match and col_name == "contents":
                handle_re = "|".join(re.escape(h) for h in handle_match)
                col_q["$or"] = [
                    {"author_handle": {"$regex": handle_re, "$options": "i"}},
                    {"text": {"$regex": handle_re, "$options": "i"}},
                ]
            elif handle_match and col_name == "pois":
                handle_re = "|".join(re.escape(h) for h in handle_match)
                col_q["$or"] = [
                    {"name": {"$regex": handle_re, "$options": "i"}},
                    {"realName": {"$regex": handle_re, "$options": "i"}},
                    {"aliasNames": {"$regex": handle_re, "$options": "i"}},
                ]

            sort_field = ts_field or "created_at" if ts_field else "_id"
            try:
                docs = list(db[col_name].find(
                    col_q, fields
                ).sort(sort_field, DESCENDING).limit(default_limit))
            except Exception:
                # Fallback: sort might fail if field doesn't exist; try without sort
                try:
                    docs = list(db[col_name].find(col_q, fields).limit(default_limit))
                except Exception:
                    docs = []

            for k, d in enumerate(docs, 1):
                snippets.append(formatter(d, k))
            total += len(docs)

        client.close()
        return snippets, total
    except Exception as e:
        logger.warning("_build_db_context failed: %s", e)
        return [], 0


def _global_query(question: str, top_k: int, time_window_days: Optional[int]) -> dict:
    """
    Unified query engine — architecture:

    1. Smalltalk shortcut  → canned greeting, no DB call.
    2. Count fast-path     → real Mongo count, no LLM.
    3. Universal DB context → ALWAYS pulls live alerts + grievances from Mongo.
    4. Vector enrichment   → if vector store has indexed chunks, append the
                             top semantic matches to the context (bonus signal).
    5. LLM                 → Ollama synthesises DB facts + general Telangana
                             knowledge into a police-grade briefing.
    """
    smalltalk = _smalltalk_response(question)
    if smalltalk is not None:
        return {"answer": smalltalk, "sources": [], "question": question, "smalltalk": True}

    fast = _count_fast_path(question, time_window_days)
    if fast is not None:
        return fast

    # ── Step 3: Universal DB context (ALWAYS runs) ───────────────────────────
    days = time_window_days if time_window_days else 7
    db_snippets, db_doc_count = _build_db_context(question, window_days=days, limit_per=10)

    # ── Step 4: Vector enrichment (bonus — only if embeddings exist) ─────────
    vec_extra_parts = []
    try:
        embedder = OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)
        q_vec = embedder.embed_text(question)
        if q_vec is not None:
            vec_cols = _list_vector_collections()
            per_store_k = 4
            vec_hits: list = []
            for vc in vec_cols:
                try:
                    store = _get_store(vc)
                    results = store.cosine_search(query_vector=q_vec, top_k=per_store_k, query_text=question)
                    vec_hits.extend(results)
                except Exception:
                    pass
            vec_hits.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            for r in vec_hits[:6]:
                meta = r.get("metadata", {})
                if r.get("score", 0) >= 0.50:  # only include if reasonably relevant
                    vec_extra_parts.append(
                        f"[VEC · id={meta.get('document_id','')} · src={meta.get('source_collection','')} "
                        f"· score={r['score']:.2f}]\n{r['text'][:400]}"
                    )
    except Exception as e:
        logger.debug("vector enrichment skipped: %s", e)

    # ── Step 5: Build final prompt + call LLM ────────────────────────────────
    context_block = "\n\n---\n\n".join(db_snippets)
    if vec_extra_parts:
        context_block += "\n\n=== ADDITIONAL SEMANTIC MATCHES ===\n\n" + "\n\n".join(vec_extra_parts)

    if not context_block.strip():
        context_block = "(No records found across the queried modules for this time window.)"

    # Tell the LLM exactly which modules are present in the context block
    # so it doesn't ignore non-alert data (events, dial100, POIs, keywords, etc.).
    extra_modules = _detect_extra_collections(question)
    module_label = "alerts, grievances"
    if extra_modules:
        pretty = {
            "contents": "contents", "dial100incidents": "Dial 100 calls",
            "events": "events", "pois": "persons of interest",
            "keywords": "keywords", "sources": "monitored profiles",
            "dailyprogrammes": "daily programmes",
            "telegrammessages": "telegram messages",
            "criticismreports": "criticism reports",
            "suggestionreports": "suggestion reports",
        }
        module_label += ", " + ", ".join(pretty.get(m, m) for m in extra_modules)

    window_label = f"last {days} day(s)" if days else "all time"
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== DATABASE CONTEXT — {module_label} ({window_label}) ===\n"
        f"{context_block}\n"
        f"=== END CONTEXT ===\n\n"
        f"User question: {question}\n\n"
        f"OUTPUT REQUIREMENTS — read carefully:\n"
        f"• Your answer MUST be at least 10 lines (target 15–25 lines, ~250+ words). "
        f"A short reply is a failure — Commissioners need depth.\n"
        f"• Use ALL record types above (ALERT, GRIEVANCE, EVENT, DIAL-100 CALL, "
        f"PERSON OF INTEREST, KEYWORD, MONITORED PROFILE, CONTENT, DAILY PROGRAMME, "
        f"TELEGRAM MSG, CRITICISM/SUGGESTION REPORT) — not just alerts and grievances.\n"
        f"• For EVERY post/alert/grievance/content/telegram you mention, append "
        f"`[View Post](URL)` using the URL/content_url/post_link/Link field from "
        f"the context. If no URL exists for that record, write '_(no URL on file)_'.\n"
        f"• For EVERY monitored profile, POI, or @handle you mention, append "
        f"`[Profile](PROFILE_URL)` using the 'Profile URL' line from the context.\n"
        f"• Always include a final '**Links & profiles**' section that re-lists "
        f"every clickable URL grouped by type (Posts, Profiles, Reports), so the "
        f"officer can copy them in one place.\n"
        f"• Cite exact numbers, handles, FIRs, codes, locations from the context — "
        f"never invent data. If the context has no relevant records for the question, "
        f"say so in one line and then provide a 10+ line briefing using domain "
        f"knowledge labelled _(General:…)_.\n\nAnswer:"
    )
    answer = _llm_answer(prompt)
    # Belt-and-braces: if the LLM still produced too short an answer, append the
    # raw evidence bundle so the officer at least sees the records and links.
    answer = _ensure_minimum_answer(answer, db_snippets, question)

    # Expose previews from every record type, not just the first 6 alerts —
    # so the UI can show officers which module each citation came from.
    sources = []
    for i, s in enumerate(db_snippets[:10]):
        # First token in each snippet is "[ALERT", "[GRIEVANCE", "[EVENT", etc.
        tag = s.split("|", 1)[0].lstrip("[").strip().lower() or "record"
        sources.append({
            "collection": tag,
            "document_id": f"db-{i+1}",
            "score": 1.0,
            "preview": s[:200],
        })

    return {
        "answer": answer, "sources": sources, "question": question,
        "scope": "db_direct+vec", "window_doc_count": db_doc_count,
        "time_window_days": days,
    }


@app.post("/api/rag/query")
def query(req: QueryRequest):
    """Ask a question. If `collection` is omitted (or 'all'), searches across
    every indexed collection — chat-bot style — instead of one specific source.

    When ``use_db`` is False the DB + vector layers are bypassed entirely and
    the question is sent to the LLM as pure conversational chat.
    """
    if not req.use_db:
        return _chat_only_answer(req.question)
    raw_col = (req.collection or "").strip().lower()
    if raw_col in ("", "all", "*", "global", "everything"):
        out = _global_query(req.question, req.top_k, req.time_window_days)
        out["time_window_days"] = req.time_window_days
        return out
    collection = req.collection

    try:
        vec_col, use_source_filter, data_exists = _auto_ingest_if_needed(collection)
        if not data_exists:
            return {
                "answer": f"The collection '{collection}' does not exist in the database.",
                "sources": [],
                "question": req.question,
                "collection": collection,
                "vector_collection": vec_col,
                "ingested": False,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

    bot = Assistant(
        ollama_base_url=OLLAMA_BASE_URL,
        llm_model=OLLAMA_LLM_MODEL,
        embed_model=OLLAMA_EMBED_MODEL,
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME,
        vector_collection=vec_col,
        top_k=req.top_k,
        source_collection=collection if use_source_filter else None,
    )

    allow_ids = _allow_ids_within_window(collection, req.time_window_days or 0)
    result = bot.ask(req.question, doc_id_filter=allow_ids)
    bot.close()
    # Backstop: enforce the 10-line / always-include-links contract even when
    # the per-collection path is used.
    snippet_previews = [s.get("preview", "") for s in result.get("sources", []) if s.get("preview")]
    if snippet_previews:
        result["answer"] = _ensure_minimum_answer(
            result.get("answer", ""), snippet_previews, req.question
        )
    result["collection"] = collection
    result["vector_collection"] = vec_col
    result["scoped_via_metadata"] = use_source_filter
    result["time_window_days"] = req.time_window_days
    if allow_ids is not None:
        result["window_doc_count"] = len(allow_ids)
    return result


# ---------------------------------------------------------------------------
# Async / background queries
#
# Every async query is persisted to the `rag_jobs` collection so:
#   • the UI can poll for completion later (even after refresh / tab close)
#   • answers stay visible across browser sessions and devices
#   • multiple users see the same history per collection
# ---------------------------------------------------------------------------

JOBS_COLLECTION = os.getenv("RAG_JOBS_COLLECTION", "rag_jobs")
_jobs_client_lock = threading.Lock()
_jobs_client: Optional[MongoClient] = None


def _jobs_col():
    """Return a long-lived handle to the rag_jobs collection."""
    global _jobs_client
    with _jobs_client_lock:
        if _jobs_client is None:
            _jobs_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = _jobs_client[DB_NAME]
            db[JOBS_COLLECTION].create_index([("created_at", DESCENDING)])
            db[JOBS_COLLECTION].create_index([("collection", 1), ("created_at", DESCENDING)])
            db[JOBS_COLLECTION].create_index("job_id", unique=True)
        return _jobs_client[DB_NAME][JOBS_COLLECTION]


# Max source docs to auto-ingest inline (blocks the query until done).
# Collections larger than this are ingested in the background so the query
# doesn't hang for minutes while Ollama embeds thousands of docs.
AUTO_INGEST_INLINE_LIMIT = int(os.getenv("AUTO_INGEST_INLINE_LIMIT", "500"))

# Track background ingestion so we don't launch duplicates
_bg_ingest_lock = threading.Lock()
_bg_ingest_running: set = set()  # collection names currently being ingested


def _bg_ingest_worker(collection: str):
    """Background thread that ingests a large collection without blocking queries."""
    try:
        _run_ingest(collection)
        logger.info("Background ingest: '%s' complete.", collection)
    except Exception as exc:
        logger.error("Background ingest failed for '%s': %s", collection, exc)
    finally:
        with _bg_ingest_lock:
            _bg_ingest_running.discard(collection)


def _auto_ingest_if_needed(collection: str) -> tuple:
    """Check if embeddings exist for *collection*; if not, ingest automatically.

    - Small collections (≤ AUTO_INGEST_INLINE_LIMIT docs): ingest inline and block.
    - Large collections: kick off background ingestion so the query can still proceed
      with whatever partial data exists (or return a helpful message).

    Returns (vec_col, use_source_filter, data_exists).
    """
    per_col_vec = f"{VECTOR_COLLECTION}_{collection}"

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    existing = set(db.list_collection_names())

    # Source collection must exist in the database
    if collection not in existing:
        client.close()
        return per_col_vec, False, False  # vec_col, use_source_filter, data_exists

    # Already has a per-collection vector index with data — no ingestion needed
    if per_col_vec in existing and db[per_col_vec].estimated_document_count() > 0:
        client.close()
        return per_col_vec, False, True

    # Global vector collection has chunks for this collection
    if VECTOR_COLLECTION in existing:
        count = db[VECTOR_COLLECTION].count_documents(
            {"metadata.source_collection": collection}, limit=1
        )
        if count > 0:
            client.close()
            return VECTOR_COLLECTION, True, True

    # No embeddings — decide whether to ingest inline or in background
    source_doc_count = db[collection].estimated_document_count()
    client.close()

    if source_doc_count <= AUTO_INGEST_INLINE_LIMIT:
        # Small collection — ingest inline (fast, a few seconds)
        logger.info(
            "Auto-ingest inline: '%s' (%d docs) — ingesting before query...",
            collection, source_doc_count,
        )
        try:
            _run_ingest(collection)
            logger.info("Auto-ingest inline: '%s' complete.", collection)
        except Exception as exc:
            logger.error("Auto-ingest inline failed for '%s': %s", collection, exc)
        return per_col_vec, False, True
    else:
        # Large collection — ingest in background, don't block query
        with _bg_ingest_lock:
            already_running = collection in _bg_ingest_running
            if not already_running:
                _bg_ingest_running.add(collection)
        if not already_running:
            logger.info(
                "Auto-ingest background: '%s' (%d docs) — too large for inline, "
                "spawning background thread.",
                collection, source_doc_count,
            )
            t = threading.Thread(
                target=_bg_ingest_worker,
                args=(collection,),
                name=f"auto-ingest-{collection}",
                daemon=True,
            )
            t.start()
        else:
            logger.info("Auto-ingest background: '%s' already in progress.", collection)

        return per_col_vec, False, True


def _process_job(job_id: str, question: str, collection: str, top_k: int,
                 vec_col: str, use_source_filter: bool, time_window_days: Optional[int] = 7):
    """Background worker — runs the actual RAG query and stores the result."""
    col = _jobs_col()
    started = datetime.now(timezone.utc)
    col.update_one(
        {"job_id": job_id},
        {"$set": {"status": "running", "started_at": started}},
    )
    try:
        # Auto-ingest if this collection has no embeddings yet
        vec_col, use_source_filter, data_exists = _auto_ingest_if_needed(collection)

        if not data_exists:
            finished = datetime.now(timezone.utc)
            col.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "completed",
                    "answer": f"The collection '{collection}' does not exist in the database.",
                    "sources": [],
                    "finished_at": finished,
                    "duration_ms": int((finished - started).total_seconds() * 1000),
                }},
            )
            return

        # Check if the vector collection actually has data after auto-ingest.
        # If not (large collection still ingesting in background), tell the user.
        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            _db = _client[DB_NAME]
            vec_count = _db[vec_col].estimated_document_count() if vec_col in _db.list_collection_names() else 0
            _client.close()
        except Exception:
            vec_count = 0

        if vec_count == 0:
            # Background ingestion in progress — no data yet to search
            with _bg_ingest_lock:
                still_running = collection in _bg_ingest_running
            if still_running:
                answer = (
                    f"This is the first time the **{collection}** collection is being queried. "
                    f"I'm currently indexing the data in the background — this may take a few minutes "
                    f"for large collections.\n\n"
                    f"**Please try again shortly.** Your question will be answered once indexing completes."
                )
            else:
                answer = (
                    f"No indexed data found for the **{collection}** collection. "
                    f"Indexing may have failed. Please check the RAG server logs."
                )
            finished = datetime.now(timezone.utc)
            col.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "completed",
                    "answer": answer,
                    "sources": [],
                    "finished_at": finished,
                    "duration_ms": int((finished - started).total_seconds() * 1000),
                }},
            )
            return

        bot = Assistant(
            ollama_base_url=OLLAMA_BASE_URL,
            llm_model=OLLAMA_LLM_MODEL,
            embed_model=OLLAMA_EMBED_MODEL,
            mongo_uri=MONGODB_URI,
            db_name=DB_NAME,
            vector_collection=vec_col,
            top_k=top_k,
            source_collection=collection if use_source_filter else None,
        )
        allow_ids = _allow_ids_within_window(collection, time_window_days or 0)
        result = bot.ask(question, doc_id_filter=allow_ids)
        bot.close()

        finished = datetime.now(timezone.utc)
        col.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "completed",
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "vector_collection": vec_col,
                "scoped_via_metadata": use_source_filter,
                "finished_at": finished,
                "duration_ms": int((finished - started).total_seconds() * 1000),
            }},
        )
        logger.info("Job %s completed in %ss", job_id, (finished - started).total_seconds())
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        col.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc),
            }},
        )


@app.post("/api/rag/query/async")
def query_async(req: QueryRequest):
    """Enqueue a question for background processing. Returns immediately with a job_id.

    If no embeddings exist for the requested collection, the background worker
    will auto-ingest before answering — no manual ingestion required.
    """
    collection = req.collection or os.getenv("COLLECTION_NAME", "contents")

    # Verify the source collection actually exists in the database
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        existing = set(db.list_collection_names())
        client.close()

        if collection not in existing:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{collection}' does not exist in the database.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

    # Determine initial vec_col (the background worker will re-check and auto-ingest if needed)
    per_col_vec = f"{VECTOR_COLLECTION}_{collection}"
    vec_col = per_col_vec if per_col_vec in existing else VECTOR_COLLECTION
    use_source_filter = vec_col == VECTOR_COLLECTION

    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    _jobs_col().insert_one({
        "job_id": job_id,
        "status": "queued",
        "question": req.question,
        "collection": collection,
        "vector_collection": vec_col,
        "scoped_via_metadata": use_source_filter,
        "top_k": req.top_k,
        "created_at": now,
        "answer": None,
        "sources": [],
    })

    t = threading.Thread(
        target=_process_job,
        args=(job_id, req.question, collection, req.top_k, vec_col, use_source_filter, req.time_window_days),
        daemon=True,
    )
    t.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "question": req.question,
        "collection": collection,
        "created_at": now.isoformat(),
    }


def _serialize_job(doc: dict) -> dict:
    """Convert a stored job document into a JSON-friendly dict."""
    out = {k: v for k, v in doc.items() if k != "_id"}
    for ts_key in ("created_at", "started_at", "finished_at"):
        ts = out.get(ts_key)
        if isinstance(ts, datetime):
            out[ts_key] = ts.isoformat()
    return out


@app.get("/api/rag/jobs/{job_id}")
def get_job(job_id: str):
    """Return the current status / result of a job."""
    doc = _jobs_col().find_one({"job_id": job_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(doc)


@app.get("/api/rag/jobs")
def list_jobs(collection: Optional[str] = None, limit: int = 50, status: Optional[str] = None):
    """Return recent jobs, newest first. Optionally filter by collection / status."""
    q: dict = {}
    if collection:
        q["collection"] = collection
    if status:
        q["status"] = status
    limit = max(1, min(limit, 200))
    docs = list(
        _jobs_col().find(q).sort("created_at", DESCENDING).limit(limit)
    )
    return {"jobs": [_serialize_job(d) for d in docs], "count": len(docs)}


@app.delete("/api/rag/jobs/{job_id}")
def delete_job(job_id: str):
    """Remove a job from history."""
    res = _jobs_col().delete_one({"job_id": job_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True, "job_id": job_id}


# ---------------------------------------------------------------------------
# Ingestion + Scheduler
# ---------------------------------------------------------------------------

_scheduler_state = {
    "running": False,
    "thread": None,
    "stop": threading.Event(),
    "current_collection": None,
    "in_progress": False,
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "next_run_at": None,
    "last_results": [],
}
_scheduler_lock = threading.Lock()


def _ingest_runs_col():
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME][INGEST_RUNS_COLLECTION]


def _record_run(doc: dict):
    try:
        col = _ingest_runs_col()
        col.insert_one(doc)
        col.database.client.close()
    except Exception as e:
        logger.warning("Could not persist ingest run record: %s", e)


def _run_ingest(collection: str) -> dict:
    """Incrementally ingest new docs from a source collection into its vector store.

    Skips documents whose `_id` is already embedded, so it's safe to call repeatedly
    (e.g. on a schedule) — only new data hits Ollama.
    """
    vec_col = f"{VECTOR_COLLECTION}_{collection}"
    logger.info("Ingestion start: '%s' → '%s'", collection, vec_col)

    streamer = MongoStreamProcessor(MONGODB_URI, DB_NAME, collection, BATCH_SIZE)
    converter = DocumentConverter()
    chunker = TokenAwareChunker(min_tokens=CHUNK_MIN, max_tokens=CHUNK_MAX, overlap=CHUNK_OVERLAP)
    embedder = OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)
    store = VectorStore(MONGODB_URI, DB_NAME, vec_col)

    if not embedder.check_health():
        store.close()
        raise RuntimeError("Ollama embedding model not available")

    already_done = store.get_embedded_doc_ids()
    total_docs = streamer.count_documents()

    docs_processed = 0
    chunks_stored = 0
    embed_failures = 0
    pending = []

    for doc in streamer.stream_documents(skip_ids=already_done):
        doc_id = str(doc.get("_id", ""))
        text = converter.convert(doc)
        if not text.strip():
            continue

        doc_chunks = chunker.chunk_document(text, collection, doc_id)
        for chunk in doc_chunks:
            embedding = embedder.embed_text(chunk.text)
            if embedding is None:
                embed_failures += 1
                continue
            pending.append({
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    "source_collection": chunk.metadata.source_collection,
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                },
            })

        docs_processed += 1
        if len(pending) >= BATCH_SIZE:
            chunks_stored += store.upsert_chunks(pending)
            pending = []

    if pending:
        chunks_stored += store.upsert_chunks(pending)

    # Invalidate the local numpy cache so the next query picks up newly ingested chunks.
    if docs_processed > 0:
        store.invalidate_cache()
        logger.info("Invalidated vector cache for '%s' after ingesting %d new docs.", vec_col, docs_processed)

    store.close()

    result = {
        "collection": collection,
        "vector_collection": vec_col,
        "total_source_docs": total_docs,
        "docs_processed": docs_processed,
        "chunks_stored": chunks_stored,
        "embed_failures": embed_failures,
        "skipped_already_done": len(already_done),
    }
    logger.info(
        "Ingestion done: '%s' — new_docs=%d new_chunks=%d failures=%d (skipped %d already-embedded)",
        collection, docs_processed, chunks_stored, embed_failures, len(already_done),
    )
    return result


@app.post("/api/rag/ingest")
def ingest(req: IngestRequest):
    """Trigger incremental ingestion for a specific collection (runs synchronously)."""
    try:
        return _run_ingest(req.collection)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _run_scheduled_cycle() -> list:
    """Run incremental ingestion for every configured collection, sequentially.

    Returns a list of per-collection result dicts. Failures are caught per-collection
    so one bad collection doesn't block the others.
    """
    cycle_results = []
    started = datetime.now(timezone.utc)
    _scheduler_state["in_progress"] = True
    _scheduler_state["last_run_started_at"] = started.isoformat()
    logger.info(
        "Scheduled ingestion cycle starting for collections: %s",
        ", ".join(INGEST_COLLECTIONS),
    )

    for col in INGEST_COLLECTIONS:
        if _scheduler_state["stop"].is_set():
            break
        _scheduler_state["current_collection"] = col
        col_started = datetime.now(timezone.utc)
        try:
            res = _run_ingest(col)
            res["status"] = "ok"
        except Exception as e:
            logger.exception("Scheduled ingestion failed for '%s'", col)
            res = {"collection": col, "status": "error", "error": str(e)}
        col_finished = datetime.now(timezone.utc)
        res["started_at"] = col_started.isoformat()
        res["finished_at"] = col_finished.isoformat()
        res["duration_s"] = (col_finished - col_started).total_seconds()
        cycle_results.append(res)

    _scheduler_state["current_collection"] = None
    _scheduler_state["in_progress"] = False
    finished = datetime.now(timezone.utc)
    _scheduler_state["last_run_finished_at"] = finished.isoformat()
    _scheduler_state["last_results"] = cycle_results

    _record_run({
        "started_at": started,
        "finished_at": finished,
        "duration_s": (finished - started).total_seconds(),
        "trigger": _scheduler_state.pop("_trigger", "scheduled"),
        "results": cycle_results,
    })
    logger.info(
        "Scheduled ingestion cycle complete in %.1fs",
        (finished - started).total_seconds(),
    )
    return cycle_results


def _scheduler_loop():
    """Background loop: every INGEST_INTERVAL_HOURS, run a full ingestion cycle.

    First cycle runs ~30 seconds after startup so the API is responsive immediately.
    """
    interval_s = max(60.0, INGEST_INTERVAL_HOURS * 3600.0)
    # Initial delay so we don't hammer Mongo/Ollama on a cold boot
    initial_delay = 30.0
    _scheduler_state["next_run_at"] = (
        datetime.now(timezone.utc).timestamp() + initial_delay
    )
    if _scheduler_state["stop"].wait(initial_delay):
        return

    while not _scheduler_state["stop"].is_set():
        try:
            _run_scheduled_cycle()
        except Exception:
            logger.exception("Scheduler cycle crashed; will retry next interval")
        _scheduler_state["next_run_at"] = (
            datetime.now(timezone.utc).timestamp() + interval_s
        )
        if _scheduler_state["stop"].wait(interval_s):
            break


def _start_scheduler():
    with _scheduler_lock:
        if _scheduler_state["running"]:
            return
        if not SCHEDULER_ENABLED:
            logger.info("Ingestion scheduler disabled via INGEST_SCHEDULER_ENABLED=false")
            return
        if not INGEST_COLLECTIONS:
            logger.warning("INGEST_COLLECTIONS is empty — scheduler will not run")
            return
        _scheduler_state["stop"].clear()
        t = threading.Thread(target=_scheduler_loop, name="rag-ingest-scheduler", daemon=True)
        t.start()
        _scheduler_state["thread"] = t
        _scheduler_state["running"] = True
        logger.info(
            "Ingestion scheduler started — every %.2fh for collections: %s",
            INGEST_INTERVAL_HOURS, ", ".join(INGEST_COLLECTIONS),
        )


@app.on_event("startup")
def _on_startup_scheduler():
    _start_scheduler()


@app.on_event("shutdown")
def _on_shutdown_scheduler():
    _scheduler_state["stop"].set()


@app.get("/api/rag/scheduler/status")
def scheduler_status():
    """Show scheduler config + current/last run details."""
    next_ts = _scheduler_state.get("next_run_at")
    return {
        "enabled": SCHEDULER_ENABLED,
        "running": _scheduler_state["running"],
        "interval_hours": INGEST_INTERVAL_HOURS,
        "collections": INGEST_COLLECTIONS,
        "in_progress": _scheduler_state["in_progress"],
        "current_collection": _scheduler_state["current_collection"],
        "last_run_started_at": _scheduler_state["last_run_started_at"],
        "last_run_finished_at": _scheduler_state["last_run_finished_at"],
        "next_run_at": (
            datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat() if next_ts else None
        ),
        "last_results": _scheduler_state["last_results"],
    }


@app.post("/api/rag/scheduler/run-now")
def scheduler_run_now():
    """Trigger an immediate ingestion cycle in the background (non-blocking)."""
    if _scheduler_state["in_progress"]:
        return {"queued": False, "message": "A cycle is already in progress"}
    _scheduler_state["_trigger"] = "manual"
    threading.Thread(target=_run_scheduled_cycle, name="rag-ingest-manual", daemon=True).start()
    return {"queued": True, "collections": INGEST_COLLECTIONS}


@app.get("/api/rag/scheduler/runs")
def scheduler_runs(limit: int = 20):
    """List recent scheduled ingestion cycles persisted in MongoDB."""
    try:
        col = _ingest_runs_col()
        cur = col.find({}, {"_id": 0}).sort("started_at", DESCENDING).limit(min(limit, 100))
        runs = []
        for r in cur:
            for k in ("started_at", "finished_at"):
                if isinstance(r.get(k), datetime):
                    r[k] = r[k].isoformat()
            runs.append(r)
        col.database.client.close()
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats")
def stats():
    """Return ingestion stats across all vector collections."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        all_cols = db.list_collection_names()
        vec_cols = [c for c in all_cols if c.startswith(VECTOR_COLLECTION)]

        result = {}
        for vc in vec_cols:
            col = db[vc]
            count = col.estimated_document_count()
            unique = 0
            if count > 0:
                # Aggregation avoids the 16MB cap that `distinct` hits on large collections.
                try:
                    agg = list(col.aggregate([
                        {"$group": {"_id": "$metadata.document_id"}},
                        {"$count": "n"},
                    ], allowDiskUse=True))
                    unique = agg[0]["n"] if agg else 0
                except Exception as e:
                    logger.warning("unique-doc count failed for %s: %s", vc, e)
            result[vc] = {"chunks": count, "unique_documents": unique}

        client.close()
        return {"database": DB_NAME, "vector_collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Daily Status Report (DSR)
#
# Generates a morning summary of the previous 24h of activity across the
# platform's main collections. Cached per UTC date in `rag_dsr` so repeated
# requests on the same day are cheap. Re-generated automatically every
# morning at DSR_HOUR_UTC (default 02:00 UTC ≈ 7:30 AM IST).
# ---------------------------------------------------------------------------

DSR_COLLECTION = os.getenv("DSR_COLLECTION", "rag_dsr")
DSR_HOUR_UTC = int(os.getenv("DSR_HOUR_UTC", "2"))  # 02:00 UTC ≈ 07:30 IST

# Restricts which collections the chatbot is allowed to search via the vector
# store enrichment step. We expose ALL operational modules (events, alerts,
# grievances, Dial 100 calls, POIs, profiles/sources, keywords, contents,
# daily programmes, telegram messages, and the reporting collections) so the
# assistant can answer questions across the entire OSINT portal — not just
# alerts/grievances. Override via the ALLOWED_QUERY_COLLECTIONS env var.
ALLOWED_QUERY_COLLECTIONS = [
    c.strip() for c in os.getenv(
        "ALLOWED_QUERY_COLLECTIONS",
        "alerts,grievances,events,dial100incidents,pois,keywords,sources,contents,"
        "dailyprogrammes,telegrammessages,criticism_reports,grievance_workflow_reports,"
        "query_reports,suggestion_reports",
    ).split(",") if c.strip()
]


def _yesterday_window():
    now = datetime.now(timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    return start, end


def _collect_dsr_data() -> dict:
    """Pull yesterday's alerts + grievances from Mongo for the DSR."""
    start, end = _yesterday_window()
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    aq = {"created_at": {"$gte": start, "$lt": end}}

    # ── Alert counts by category and priority ────────────────────────────────
    total_alerts = db.alerts.count_documents(aq)
    high_alerts  = db.alerts.count_documents({**aq, "priority": "HIGH"})
    med_alerts   = db.alerts.count_documents({**aq, "priority": "MEDIUM"})

    cat_pipeline = [
        {"$match": aq},
        {"$group": {"_id": "$source_category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    categories = {r["_id"] or "unknown": r["count"]
                  for r in db.alerts.aggregate(cat_pipeline)}

    # ── Platform breakdown ───────────────────────────────────────────────────
    plat_pipeline = [
        {"$match": aq},
        {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 8},
    ]
    platforms = {r["_id"] or "unknown": r["count"]
                 for r in db.alerts.aggregate(plat_pipeline)}

    # ── Top HIGH-risk alerts (full fields for formatter) ─────────────────────
    top_alerts = list(db.alerts.find(
        {**aq, "priority": "HIGH"}, ALERT_FIELDS
    ).sort("created_at", DESCENDING).limit(20))

    # Also grab MEDIUM for breadth
    med_sample = list(db.alerts.find(
        {**aq, "priority": "MEDIUM"}, ALERT_FIELDS
    ).sort("created_at", DESCENDING).limit(8))

    # News-category alerts
    news_alerts = list(db.alerts.find(
        {**aq, "source_category": "news"}, ALERT_FIELDS
    ).sort("created_at", DESCENDING).limit(8))

    # ── Grievances (full fields with tweet text) ──────────────────────────────
    total_grievances = db.grievances.count_documents(aq)
    top_grievances = list(db.grievances.find(
        aq, GRIEVANCE_FIELDS
    ).sort("created_at", DESCENDING).limit(10))

    client.close()

    alerts_text = "\n\n".join(_fmt_alert(d, i+1) for i, d in enumerate(top_alerts))
    med_text    = "\n".join(_fmt_alert(d, i+1) for i, d in enumerate(med_sample))
    griev_text  = "\n\n".join(_fmt_grievance(d, i+1) for i, d in enumerate(top_grievances))

    # News-category alerts use the same alert formatter
    news_text   = "\n".join(_fmt_alert(d, i+1) for i, d in enumerate(news_alerts))

    return {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "stats": {
            "total_alerts": total_alerts,
            "high_alerts": high_alerts,
            "medium_alerts": med_alerts,
            "total_grievances": total_grievances,
            "categories": categories,
            "platforms": platforms,
        },
        "alerts_text": alerts_text,
        "medium_alerts_text": med_text,
        "news_text": news_text or "(no news-category alerts)",
        "grievances_text": griev_text or "(no grievances)",
    }


def _build_dsr(force: bool = False) -> dict:
    """Generate (or return cached) DSR for yesterday — alerts & grievances only."""
    start, _ = _yesterday_window()
    date_key = start.strftime("%Y-%m-%d")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    cache = db[DSR_COLLECTION]
    cache.create_index("date", unique=True)
    if not force:
        existing = cache.find_one({"date": date_key})
        if existing:
            existing.pop("_id", None)
            client.close()
            return existing
    client.close()

    raw = _collect_dsr_data()
    s = raw["stats"]

    prompt = textwrap.dedent(f"""\
        You are SOC-EYE Daily Intelligence Briefer for **Telangana Police**
        (CP / DCP / SOC analysts). Produce a crisp morning briefing from the
        alerts and grievances data below (yesterday, {date_key}).

        Data summary:
          - Total alerts: {s['total_alerts']} | HIGH: {s['high_alerts']} | MEDIUM: {s['medium_alerts']}
          - Total grievances: {s['total_grievances']}
          - Alert categories: {s['categories']}
          - Platforms: {s['platforms']}

        Use EXACTLY these five Markdown sections:

        ## High-Risk Alerts
        Top 3 HIGH-priority alerts. For each bullet:
          • **@handle** (platform) — one-line incident description
          • Risk: what law-and-order threat it poses
          • Action: takedown / FIR [BNS section] / escalate to DCP / field unit alert

        ## Most Hyped Topics
        Top 3 most-active themes/handles driving volume. For each:
          • **Theme / @handle** — why it is trending, engagement signal
          • Action: monitor / engage / counter-narrative

        ## Police-Relevant News
        Up to 3 news-category alerts that are directly relevant to police
        operations, law-and-order, or public safety in Telangana. For each:
          • **Headline** (source, platform) — significance to police
          • Action: brief note

        ## Grievances Summary
        - Count, top districts, top categories.
        - Flag any high-priority or unresolved grievances needing attention.
        - Action: assign / escalate / close

        ## Analyst's Note
        2-3 bullets: patterns, coordinated activity, new risky entities/handles,
        or anything the CP should be personally aware of today.

        CRITICAL RULES:
        - Use ONLY real data from the sections below. Do NOT invent handles,
          headlines, or incidents not present in the data.
        - If a section has no data (e.g. zero HIGH alerts), write
          "No items in this category for this period." and move on.
        - Add _(General context: …)_ ONLY as a brief supplement to real data,
          never as a replacement for missing data.
        - Bold real entities, `@real_handles`, `code` for real BNS sections.

        === HIGH-RISK ALERTS ===
        {raw['alerts_text'] or '(none)'}

        === MEDIUM ALERTS (sample) ===
        {raw['medium_alerts_text'] or '(none)'}

        === NEWS-CATEGORY ALERTS ===
        {raw['news_text']}

        === GRIEVANCES ===
        {raw['grievances_text']}
    """)

    llm_summary = ""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
            },
            timeout=600,
        )
        resp.raise_for_status()
        llm_summary = resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("DSR LLM generation failed: %s", exc)
        llm_summary = "_(LLM briefing unavailable — Ollama not reachable.)_"

    # Build the final markdown: stats header + LLM briefing
    header = textwrap.dedent(f"""\
        # Daily Status Report — {date_key}
        _Window: {raw['window']['from'][:10]} (yesterday, IST)_

        | | |
        |---|---|
        | 🚨 Total Alerts | **{s['total_alerts']:,}** |
        | 🔴 HIGH Risk | **{s['high_alerts']:,}** |
        | 🟠 MEDIUM Risk | **{s['medium_alerts']:,}** |
        | 📋 Grievances | **{s['total_grievances']:,}** |

    """)
    full_md = header + llm_summary

    doc = {
        "date": date_key,
        "generated_at": datetime.now(timezone.utc),
        "window": raw["window"],
        "stats": s,
        # keep collections key for backward compat with frontend
        "collections": {
            "alerts": {"count": s["total_alerts"]},
            "alerts_high": {"count": s["high_alerts"]},
            "alerts_medium": {"count": s["medium_alerts"]},
            "grievances": {"count": s["total_grievances"]},
        },
        "total_count": s["total_alerts"] + s["total_grievances"],
        "markdown": full_md,
        "llm_summary": llm_summary,
    }
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    db[DSR_COLLECTION].update_one({"date": date_key}, {"$set": doc}, upsert=True)
    client.close()
    doc.pop("_id", None)
    if isinstance(doc.get("generated_at"), datetime):
        doc["generated_at"] = doc["generated_at"].isoformat()
    return doc


@app.get("/api/rag/dsr")
def get_dsr(force: bool = False):
    """Return today's morning DSR (yesterday's activity)."""
    try:
        doc = _build_dsr(force=force)
        if isinstance(doc.get("generated_at"), datetime):
            doc["generated_at"] = doc["generated_at"].isoformat()
        return doc
    except Exception as e:
        logger.exception("DSR build failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/dsr/history")
def dsr_history(limit: int = 14):
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col = client[DB_NAME][DSR_COLLECTION]
        docs = list(col.find({}, {"samples": 0}).sort("date", DESCENDING).limit(min(limit, 60)))
        client.close()
        for d in docs:
            d.pop("_id", None)
            if isinstance(d.get("generated_at"), datetime):
                d["generated_at"] = d["generated_at"].isoformat()
        return {"reports": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _dsr_scheduler_loop():
    """Generate the DSR once per day at DSR_HOUR_UTC."""
    while not _scheduler_state["stop"].is_set():
        now = datetime.now(timezone.utc)
        target = now.replace(hour=DSR_HOUR_UTC, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        sleep_s = (target - now).total_seconds()
        logger.info("Next DSR generation at %s UTC (in %.0fs)", target.isoformat(), sleep_s)
        if _scheduler_state["stop"].wait(sleep_s):
            return
        try:
            _build_dsr(force=True)
            logger.info("Morning DSR generated.")
        except Exception:
            logger.exception("Morning DSR generation failed")


@app.on_event("startup")
def _start_dsr_scheduler():
    t = threading.Thread(target=_dsr_scheduler_loop, name="rag-dsr-scheduler", daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Top-50 Alerts endpoint
#
# Fetches all alerts from the last 24 hours, sends them to Ollama, which
# ranks and returns the top 50 most important unique alerts for police review.
# Each returned alert carries the original MongoDB document id so the frontend
# can render it with the existing AlertCard flow (acknowledge / escalate / etc.)
# ---------------------------------------------------------------------------

class TopAlertsRequest(BaseModel):
    hours: int = 24          # look-back window in hours (default 24h)
    top_n: int = 50          # number of top alerts to return


@app.post("/api/rag/top-alerts")
def top_alerts(req: TopAlertsRequest):
    """Fetch all alerts from the last N hours, ask Ollama to rank the top-50
    unique most-important ones, and return them with full document data."""
    hours = max(1, min(req.hours, 168))   # clamp 1h–7d
    top_n = max(1, min(req.top_n, 100))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        raw_docs = list(db.alerts.find(
            {"created_at": {"$gte": cutoff}},
            ALERT_FIELDS
        ).sort("created_at", DESCENDING).limit(2000))
        client.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

    if not raw_docs:
        return {"alerts": [], "total_scanned": 0, "hours": hours,
                "message": f"No alerts found in the last {hours} hour(s)."}

    # Build a compact index for Ollama (id → snippet)
    # Deduplicate by author_handle to avoid flooding with the same source
    seen_handles: set = set()
    candidates: list = []   # (doc, snippet) pairs
    for d in raw_docs:
        # Use the UUID 'id' field — that's what the Node backend /api/alerts/bulk expects
        uuid_id = d.get("id") or str(d["_id"])
        handle = (d.get("author_handle") or d.get("author") or "").lstrip("@").lower()
        llm = d.get("llm_analysis") or {}
        threat = d.get("threat_details") or {}
        vdata = d.get("velocity_data") or {}
        score = threat.get("risk_score") or llm.get("score") or 0
        reasoning = (d.get("classification_explanation") or llm.get("reasoning") or "").strip()
        if "Primary AI analysis unavailable" in reasoning:
            reasoning = ""
        dedup_key = f"{handle}_{d.get('source_category','?')}_{d.get('alert_type','?')}"
        if dedup_key in seen_handles:
            continue
        seen_handles.add(dedup_key)
        ts = d.get("created_at")
        ts_s = ts.strftime("%d-%b %H:%M") if isinstance(ts, datetime) else "?"
        vinfo = (f"viral:{vdata.get('metric','?')} velocity={vdata.get('velocity','?')}"
                 if vdata.get("velocity") else "")
        snippet = (
            f"ID:{uuid_id} | pri={d.get('priority','?')} | risk={d.get('risk_level','?')} "
            f"| score={score}% | cat={d.get('source_category','?')} | type={d.get('alert_type','?')}\n"
            f"  @{handle} on {d.get('platform','?')} | {ts_s} {vinfo}\n"
            f"  URL: {d.get('content_url','')}\n"
            + (f"  Analysis: {reasoning[:250]}\n" if reasoning else "")
        )
        candidates.append((d, snippet, uuid_id))

    total_unique = len(candidates)

    # Build Ollama prompt
    candidates_text = "\n".join(s for _, s, _ in candidates)
    prompt = textwrap.dedent(f"""\
        You are a Telangana Police SOC analyst. Below are {total_unique} unique alerts
        from the last {hours} hour(s). Your task:

        1. Select the TOP {top_n} most important alerts that require police attention.
        2. Rank criteria (highest weight first):
           a. Direct threat to public order / communal violence / hate speech with legal implications
           b. High velocity / viral spread (rapid reach = rapid harm)
           c. High risk score (80%+)
           d. Sensitive categories: communal > political > defamation > narcotics > history_sheeters
           e. Novelty — prefer diverse handles over repeated entries from the same author

        3. Output ONLY a JSON array of the selected alert IDs in ranked order, like:
           ["id1","id2","id3",...]
           Do NOT output anything else — no explanation, no markdown, no extra text.
           The IDs must be taken EXACTLY as they appear in "ID:..." lines below.

        ALERTS:
        {candidates_text}
    """)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048},
            },
            timeout=300,
        )
        resp.raise_for_status()
        raw_response = resp.json().get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {e}")

    # Parse the JSON array from the LLM response
    import json as _json
    ranked_ids: list = []
    try:
        # Extract the JSON array — handle LLM wrapping it in ```json ... ```
        match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
        if match:
            ranked_ids = _json.loads(match.group(0))
    except Exception:
        pass

    if not ranked_ids:
        logger.warning("top-alerts LLM parse failed; falling back to rule-based ranking")
        priority_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        candidates.sort(key=lambda x: (
            -priority_weight.get(x[0].get("priority", "LOW"), 0),
            -(x[0].get("threat_details") or {}).get("risk_score", 0),
        ))
        ranked_ids = [uid for _, _, uid in candidates[:top_n]]

    # Build uuid_id → doc map
    doc_map = {uid: d for d, _, uid in candidates}

    # Assemble final list in ranked order, deduplicated
    seen_ids: set = set()
    result_docs = []
    for rid in ranked_ids:
        if rid in seen_ids or rid not in doc_map:
            continue
        seen_ids.add(rid)
        d = doc_map[rid]
        out = {k: v for k, v in d.items() if k != "_id"}
        out["id"] = rid   # always the UUID string from the 'id' field
        if isinstance(out.get("created_at"), datetime):
            out["created_at"] = out["created_at"].isoformat()
        result_docs.append(out)

    # Fill to top_n with highest-scored unseen ones
    if len(result_docs) < top_n:
        priority_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        remaining = sorted(
            [(d, uid) for d, _, uid in candidates if uid not in seen_ids],
            key=lambda x: (
                -priority_weight.get(x[0].get("priority", "LOW"), 0),
                -(x[0].get("threat_details") or {}).get("risk_score", 0),
            )
        )
        for d, uid in remaining[: top_n - len(result_docs)]:
            out = {k: v for k, v in d.items() if k != "_id"}
            out["id"] = uid
            if isinstance(out.get("created_at"), datetime):
                out["created_at"] = out["created_at"].isoformat()
            result_docs.append(out)

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_doc = {
        "date": date_key,
        "generated_at": datetime.now(timezone.utc),
        "hours": hours,
        "total_scanned": len(raw_docs),
        "total_unique": total_unique,
        "top_n": len(result_docs),
        # Store only the alert IDs + lightweight metadata to keep the doc small
        "alert_ids": [a["id"] for a in result_docs],
        "alert_meta": [
            {
                "id": a["id"],
                "priority": a.get("priority"),
                "risk_level": a.get("risk_level"),
                "source_category": a.get("source_category"),
                "platform": a.get("platform"),
                "author_handle": a.get("author_handle") or a.get("author"),
                "content_url": a.get("content_url"),
                "created_at": a.get("created_at"),
                "threat_details": a.get("threat_details"),
                "velocity_data": a.get("velocity_data"),
                "classification_explanation": a.get("classification_explanation"),
            }
            for a in result_docs
        ],
    }
    try:
        client2 = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col2 = client2[DB_NAME]["rag_top_alerts"]
        col2.create_index([("date", 1), ("hours", 1)])
        col2.update_one(
            {"date": date_key, "hours": hours},
            {"$set": run_doc},
            upsert=True,
        )
        client2.close()
    except Exception as e:
        logger.warning("Failed to persist top-alerts run: %s", e)

    return {
        "alerts": result_docs,
        "total_scanned": len(raw_docs),
        "total_unique": total_unique,
        "top_n": len(result_docs),
        "hours": hours,
        "date": date_key,
    }


# ---------------------------------------------------------------------------
# Per-category Top-N Alerts endpoint
#
# Returns up to `top_n_per_category` LLM-ranked alerts for EACH source_category
# (communal, political, defamation, narcotics, history_sheeters, trouble_makers,
# others). Ollama is called once per category in parallel.
# ---------------------------------------------------------------------------

CATEGORY_KEYS = [
    "communal", "political", "defamation",
    "narcotics", "history_sheeters", "trouble_makers", "others",
]


class TopAlertsByCategoryRequest(BaseModel):
    hours: int = 24                  # look-back window
    top_n_per_category: int = 50     # max per category
    categories: Optional[list] = None  # restrict to a subset; default = all


def _rank_category_via_ollama(category: str, candidates: list, top_n: int, hours: int) -> list:
    """Call Ollama once for a single category and return ranked uuid IDs,
    up to min(top_n, len(candidates)). After the LLM picks, pad with
    priority+risk-score sorted candidates not already chosen so the result
    always fills available capacity. Falls back fully to rule-based ranking
    if Ollama is unreachable or returns nothing parseable."""
    if not candidates:
        return []

    target = min(top_n, len(candidates))
    snippets = "\n".join(s for _, s, _ in candidates)

    # Pre-compute the rule-based fallback order once — used both as fallback
    # and to pad short LLM responses.
    priority_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    rule_order = sorted(
        candidates,
        key=lambda x: (
            -priority_weight.get(x[0].get("priority", "LOW"), 0),
            -(x[0].get("threat_details") or {}).get("risk_score", 0),
        ),
    )
    rule_ids = [uid for _, _, uid in rule_order]

    prompt = textwrap.dedent(f"""\
        You are a Telangana Police SOC analyst reviewing alerts in the
        "{category}" category from the last {hours} hour(s).

        TASK: Return the top {target} most important alert IDs from the list below,
        ranked from MOST important to LEAST important.
        - If there are fewer than {target} alerts in the list, return ALL of them
          ranked in order.
        - Do NOT skip alerts: every alert in the list should appear in your
          output unless you are returning the full {target}-item cap.
        - Ranking priority (highest first):
            a. Direct threat to public order / law-and-order incidents
            b. High velocity / viral spread
            c. High risk score (80%+)
            d. Diverse handles — prefer variety over repeating the same author

        Output ONLY a JSON array of the alert IDs in ranked order:
          ["id1","id2","id3",...]
        No explanation. No markdown. IDs must be taken EXACTLY as shown in "ID:" lines.
        Your array MUST contain {target} IDs (or all alerts if fewer than {target} were supplied).

        ALERTS ({len(candidates)} total):
        {snippets}
    """)

    ranked_ids: list = []
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                # 4096 predict tokens fits ~50 UUIDs comfortably
                "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
            },
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        import json as _json
        # Greedy match so a [..] containing newlines is captured whole
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            ranked_ids = _json.loads(match.group(0))
    except Exception as e:
        logger.warning("Ollama ranking failed for category=%s: %s", category, e)

    # Keep only valid IDs (the LLM occasionally hallucinates), de-dup
    valid_ids = {uid for _, _, uid in candidates}
    seen: set = set()
    out: list = []
    for rid in ranked_ids:
        if rid in valid_ids and rid not in seen:
            seen.add(rid)
            out.append(rid)
            if len(out) >= target:
                break

    # Pad with rule-based ranked candidates not yet chosen, until we hit target
    if len(out) < target:
        for rid in rule_ids:
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
                if len(out) >= target:
                    break

    return out


@app.post("/api/rag/top-alerts/by-category")
def top_alerts_by_category(req: TopAlertsByCategoryRequest):
    """For each source_category, ask Ollama to rank the top-N most important
    alerts. Runs categories in parallel for latency. Returns a flat list of
    ranked alerts (preserving per-category internal order) plus per-category
    counts and breakdown."""
    hours = max(1, min(req.hours, 168))
    top_n = max(1, min(req.top_n_per_category, 100))
    requested_subset = {c.strip().lower() for c in (req.categories or []) if c}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        # Larger pool than the single-LLM endpoint since we're partitioning across categories
        raw_docs = list(db.alerts.find(
            {"created_at": {"$gte": cutoff}},
            ALERT_FIELDS,
        ).sort("created_at", DESCENDING).limit(5000))
        client.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {e}")

    if not raw_docs:
        return {"alerts": [], "categories": {}, "total_scanned": 0,
                "total_unique": 0, "top_n_per_category": top_n, "hours": hours,
                "message": f"No alerts found in the last {hours} hour(s)."}

    # Build candidates bucketed by the alert's ACTUAL source_category value
    # (so 'unknown', 'others', or any future category gets its own bucket and
    # matches the frontend chip display). Dedupe per (handle + alert_type)
    # within each bucket.
    buckets: dict = {}
    seen_per_cat: dict = {}
    for d in raw_docs:
        cat = (d.get("source_category") or "others").strip().lower() or "others"
        if requested_subset and cat not in requested_subset:
            continue
        if cat not in buckets:
            buckets[cat] = []
            seen_per_cat[cat] = set()
        uuid_id = d.get("id") or str(d["_id"])
        handle = (d.get("author_handle") or d.get("author") or "").lstrip("@").lower()
        dedup_key = f"{handle}_{d.get('alert_type','?')}"
        if dedup_key in seen_per_cat[cat]:
            continue
        seen_per_cat[cat].add(dedup_key)

        llm = d.get("llm_analysis") or {}
        threat = d.get("threat_details") or {}
        vdata = d.get("velocity_data") or {}
        score = threat.get("risk_score") or llm.get("score") or 0
        reasoning = (d.get("classification_explanation") or llm.get("reasoning") or "").strip()
        if "Primary AI analysis unavailable" in reasoning:
            reasoning = ""
        ts = d.get("created_at")
        ts_s = ts.strftime("%d-%b %H:%M") if isinstance(ts, datetime) else "?"
        vinfo = (f"viral:{vdata.get('metric','?')} velocity={vdata.get('velocity','?')}"
                 if vdata.get("velocity") else "")
        snippet = (
            f"ID:{uuid_id} | pri={d.get('priority','?')} | risk={d.get('risk_level','?')} "
            f"| score={score}% | type={d.get('alert_type','?')}\n"
            f"  @{handle} on {d.get('platform','?')} | {ts_s} {vinfo}\n"
            f"  URL: {d.get('content_url','')}\n"
            + (f"  Analysis: {reasoning[:250]}\n" if reasoning else "")
        )
        buckets[cat].append((d, snippet, uuid_id))

    total_unique = sum(len(b) for b in buckets.values())

    # Rank each category in parallel via Ollama
    ranked_by_cat: dict = {}
    cats_to_rank = [(c, items) for c, items in buckets.items() if items]
    with ThreadPoolExecutor(max_workers=min(4, len(cats_to_rank) or 1)) as ex:
        futures = {
            ex.submit(_rank_category_via_ollama, cat, items, top_n, hours): cat
            for cat, items in cats_to_rank
        }
        for fut in as_completed(futures):
            cat = futures[fut]
            try:
                ranked_by_cat[cat] = fut.result()
            except Exception as e:
                logger.warning("category rank failed for %s: %s", cat, e)
                ranked_by_cat[cat] = []

    # Assemble flat result. Iterate categories in CATEGORY_KEYS order first
    # (so known/important categories appear first), then any extras alphabetically.
    ordered_cats = [c for c in CATEGORY_KEYS if c in buckets] + \
                   sorted([c for c in buckets.keys() if c not in CATEGORY_KEYS])

    seen_ids: set = set()
    result_docs: list = []
    category_summary: dict = {}
    for cat in ordered_cats:
        bucket_items = buckets[cat]
        if not bucket_items:
            category_summary[cat] = {"count": 0, "candidates": 0}
            continue
        doc_map = {uid: d for d, _, uid in bucket_items}
        per_cat = []
        for rid in ranked_by_cat.get(cat, []):
            if rid in seen_ids or rid not in doc_map:
                continue
            seen_ids.add(rid)
            d = doc_map[rid]
            out = {k: v for k, v in d.items() if k != "_id"}
            out["id"] = rid
            if isinstance(out.get("created_at"), datetime):
                out["created_at"] = out["created_at"].isoformat()
            per_cat.append(out)
        result_docs.extend(per_cat)
        category_summary[cat] = {"count": len(per_cat), "candidates": len(bucket_items)}

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_doc = {
        "date": date_key,
        "mode": "by_category",
        "generated_at": datetime.now(timezone.utc),
        "hours": hours,
        "total_scanned": len(raw_docs),
        "total_unique": total_unique,
        "top_n_per_category": top_n,
        "categories": category_summary,
        "alert_ids": [a["id"] for a in result_docs],
        "alert_meta": [
            {
                "id": a["id"],
                "priority": a.get("priority"),
                "risk_level": a.get("risk_level"),
                "source_category": a.get("source_category"),
                "platform": a.get("platform"),
                "author_handle": a.get("author_handle") or a.get("author"),
                "content_url": a.get("content_url"),
                "created_at": a.get("created_at"),
                "threat_details": a.get("threat_details"),
                "velocity_data": a.get("velocity_data"),
                "classification_explanation": a.get("classification_explanation"),
            }
            for a in result_docs
        ],
    }
    try:
        client2 = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col2 = client2[DB_NAME]["rag_top_alerts"]
        col2.create_index([("date", 1), ("hours", 1), ("mode", 1)])
        col2.update_one(
            {"date": date_key, "hours": hours, "mode": "by_category"},
            {"$set": run_doc},
            upsert=True,
        )
        client2.close()
    except Exception as e:
        logger.warning("Failed to persist per-category top-alerts run: %s", e)

    return {
        "alerts": result_docs,
        "categories": category_summary,
        "total_scanned": len(raw_docs),
        "total_unique": total_unique,
        "top_n_per_category": top_n,
        "hours": hours,
        "date": date_key,
    }


# ---------------------------------------------------------------------------
# Daily Intelligence Report (DIR)
#
# A comprehensive daily social-media intelligence digest covering:
#   • Trending topics & keywords
#   • Viral/high-velocity posts with links
#   • Platform-wise and region-wise analysis
#   • Sentiment breakdown
#   • Most active accounts
#   • AI-generated narrative summary
#   • Threat/relevance scores
#   • Category classification (Politics, Crime, Entertainment, etc.)
#
# Stored in `rag_dir` collection, auto-generated every 24h.
# ---------------------------------------------------------------------------

DIR_COLLECTION = os.getenv("DIR_COLLECTION", "rag_dir")
DIR_HOUR_UTC    = int(os.getenv("DIR_HOUR_UTC", "3"))   # 03:00 UTC ≈ 08:30 IST

# Map internal source_category values to human-readable labels
_CATEGORY_LABELS = {
    "communal":        "Communal / Religious",
    "political":       "Politics",
    "crime":           "Crime",
    "narcotics":       "Narcotics",
    "defamation":      "Defamation",
    "hate_speech":     "Hate Speech",
    "public_order":    "Public Order",
    "news":            "News",
    "entertainment":   "Entertainment",
    "misinformation":  "Misinformation",
    "violence":        "Violence",
    "terrorism":       "Terrorism",
    "cybercrime":      "Cybercrime",
    "other":           "Other",
    "unknown":         "Unclassified",
}


def _sentiment_label(s: str) -> str:
    if not s:
        return "neutral"
    sl = s.lower()
    if any(w in sl for w in ["negative", "anger", "hostile", "hate", "threat"]):
        return "negative"
    if any(w in sl for w in ["positive", "support", "praise"]):
        return "positive"
    return "neutral"


def _enrich_with_content(db, alerts_list: list) -> list:
    """Join alerts with the `contents` collection to add full post text and media URLs.

    Each alert gets new keys:
      • text          — the full post text
      • media         — list of {type, url, video_url, thumbnail_url, s3_url}
      • quoted_text   — text of the quoted/retweeted post (X only)
      • quoted_media  — media of the quoted post
    """
    if not alerts_list:
        return alerts_list
    content_ids = list({a.get("content_id") for a in alerts_list if a.get("content_id")})
    if not content_ids:
        return alerts_list
    try:
        content_docs = list(db.contents.find(
            {"id": {"$in": content_ids}},
            {"_id": 0, "id": 1, "text": 1, "media": 1, "quoted_content": 1,
             "engagement": 1, "published_at": 1, "scraped_content": 1},
        ))
        cmap = {c["id"]: c for c in content_docs}
        for a in alerts_list:
            cid = a.get("content_id")
            c = cmap.get(cid) if cid else None
            if not c:
                a["text"] = ""
                a["media"] = []
                continue
            a["text"] = (c.get("text") or c.get("scraped_content") or "")[:2000]
            # Normalise media: prefer s3_url for cross-origin reliability, fall back to url
            media_items = []
            for m in (c.get("media") or [])[:10]:
                if not isinstance(m, dict):
                    continue
                mtype = m.get("type") or "photo"
                url = m.get("s3_url") or m.get("url") or ""
                video_url = m.get("s3_url") if mtype in ("video", "animated_gif") else m.get("video_url")
                if not url and not video_url:
                    continue
                media_items.append({
                    "type": mtype,
                    "url": url,
                    "video_url": video_url or "",
                    "thumbnail_url": m.get("thumbnail_url") or url,
                })
            a["media"] = media_items
            qc = c.get("quoted_content") or {}
            if qc:
                a["quoted_text"] = (qc.get("text") or "")[:1000]
                qmedia = []
                for m in (qc.get("media") or [])[:6]:
                    if not isinstance(m, dict):
                        continue
                    mtype = m.get("type") or "photo"
                    url = m.get("s3_url") or m.get("url") or ""
                    video_url = m.get("s3_url") if mtype in ("video", "animated_gif") else m.get("video_url")
                    if not url and not video_url:
                        continue
                    qmedia.append({
                        "type": mtype,
                        "url": url,
                        "video_url": video_url or "",
                        "thumbnail_url": m.get("thumbnail_url") or url,
                    })
                a["quoted_media"] = qmedia
            eng = c.get("engagement") or {}
            if eng:
                a["engagement"] = {
                    "likes": eng.get("likes") or eng.get("like_count") or 0,
                    "shares": eng.get("retweets") or eng.get("shares") or eng.get("share_count") or 0,
                    "comments": eng.get("replies") or eng.get("comments") or eng.get("comment_count") or 0,
                    "views": eng.get("views") or eng.get("view_count") or 0,
                }
    except Exception as e:
        logger.warning("Failed to enrich alerts with content: %s", e)
    return alerts_list


def _collect_dir_data(hours: int = 24) -> dict:
    """Pull comprehensive social-media intelligence data for the DIR."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    aq = {"created_at": {"$gte": cutoff}}

    # ── 1. Trending topics (from matched keywords) ───────────────────────────
    kw_pipeline = [
        {"$match": {**aq, "matched_keywords_normalized": {"$exists": True, "$not": {"$size": 0}}}},
        {"$unwind": "$matched_keywords_normalized"},
        {"$match": {"matched_keywords_normalized": {"$ne": None, "$ne": ""}}},
        {"$group": {
            "_id": "$matched_keywords_normalized",
            "count": {"$sum": 1},
            "platforms": {"$addToSet": "$platform"},
            "high_count": {"$sum": {"$cond": [{"$eq": ["$priority", "HIGH"]}, 1, 0]}},
            "categories": {"$addToSet": "$source_category"},
        }},
        {"$sort": {"count": -1}},
        {"$limit": 25},
    ]
    trending_keywords = list(db.alerts.aggregate(kw_pipeline))

    # ── 2. Platform breakdown with sentiment distribution ────────────────────
    plat_pipeline = [
        {"$match": aq},
        {"$group": {
            "_id": "$platform",
            "total":      {"$sum": 1},
            "high_risk":  {"$sum": {"$cond": [{"$eq": ["$priority", "HIGH"]}, 1, 0]}},
            "medium_risk":{"$sum": {"$cond": [{"$eq": ["$priority", "MEDIUM"]}, 1, 0]}},
            "avg_risk_score": {"$avg": "$threat_details.risk_score"},
            "sentiments": {"$push": "$llm_analysis.sentiment"},
        }},
        {"$sort": {"total": -1}},
        {"$limit": 10},
    ]
    platform_data_raw = list(db.alerts.aggregate(plat_pipeline))
    # Compute sentiment ratios per platform
    platform_data = []
    for p in platform_data_raw:
        sents = [_sentiment_label(s) for s in (p.get("sentiments") or []) if s]
        total_s = len(sents) or 1
        p["sentiment_breakdown"] = {
            "negative": round(sents.count("negative") / total_s * 100),
            "positive": round(sents.count("positive") / total_s * 100),
            "neutral":  round(sents.count("neutral")  / total_s * 100),
        }
        p.pop("sentiments", None)
        p["avg_risk_score"] = round(p.get("avg_risk_score") or 0, 1)
        platform_data.append(p)

    # ── 3. Viral / high-velocity posts ──────────────────────────────────────
    viral_posts_raw = list(db.alerts.find(
        {**aq, "velocity_data.velocity": {"$gt": 0}},
        {**ALERT_FIELDS, "velocity_data": 1, "content_url": 1, "title": 1},
    ).sort("velocity_data.velocity", DESCENDING).limit(15))
    viral_posts = []
    for d in viral_posts_raw:
        vdata = d.get("velocity_data") or {}
        llm = d.get("llm_analysis") or {}
        threat = d.get("threat_details") or {}
        ts = d.get("created_at")
        viral_posts.append({
            "id": str(d.get("id") or d["_id"]),
            "author_handle": (d.get("author_handle") or d.get("author") or "?").lstrip("@"),
            "platform": d.get("platform", "?"),
            "content_url": d.get("content_url", ""),
            "priority": d.get("priority", "?"),
            "source_category": d.get("source_category", "?"),
            "alert_type": d.get("alert_type", "?"),
            "velocity": vdata.get("velocity", 0),
            "velocity_metric": vdata.get("metric", ""),
            "velocity_window_min": vdata.get("time_window_minutes", "?"),
            "risk_score": threat.get("risk_score") or llm.get("score") or 0,
            "sentiment": llm.get("sentiment", ""),
            "reasoning": (d.get("classification_explanation") or llm.get("reasoning") or "")[:200],
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts or ""),
        })

    # ── 4. Most active accounts ──────────────────────────────────────────────
    account_pipeline = [
        {"$match": aq},
        {"$group": {
            "_id": {"$ifNull": ["$author_handle", "$author"]},
            "count": {"$sum": 1},
            "high_count": {"$sum": {"$cond": [{"$eq": ["$priority", "HIGH"]}, 1, 0]}},
            "categories": {"$addToSet": "$source_category"},
            "platforms":  {"$addToSet": "$platform"},
            "max_risk":   {"$max": "$threat_details.risk_score"},
            "last_seen":  {"$max": "$created_at"},
            "sample_url": {"$first": "$content_url"},
        }},
        {"$match": {"_id": {"$ne": None, "$ne": ""}}},
        {"$sort": {"count": -1}},
        {"$limit": 15},
    ]
    active_accounts_raw = list(db.alerts.aggregate(account_pipeline))
    active_accounts = []
    for a in active_accounts_raw:
        ls = a.get("last_seen")
        active_accounts.append({
            "handle": (a.get("_id") or "?").lstrip("@"),
            "alert_count": a["count"],
            "high_risk_count": a.get("high_count", 0),
            "categories": [c for c in (a.get("categories") or []) if c],
            "platforms": [p for p in (a.get("platforms") or []) if p],
            "max_risk_score": a.get("max_risk") or 0,
            "last_seen": ls.isoformat() if isinstance(ls, datetime) else str(ls or ""),
            "sample_url": a.get("sample_url", ""),
        })

    # ── 5. Category breakdown ────────────────────────────────────────────────
    cat_pipeline = [
        {"$match": aq},
        {"$group": {
            "_id": "$source_category",
            "count": {"$sum": 1},
            "high_risk": {"$sum": {"$cond": [{"$eq": ["$priority", "HIGH"]}, 1, 0]}},
            "avg_risk_score": {"$avg": "$threat_details.risk_score"},
            "platforms": {"$addToSet": "$platform"},
        }},
        {"$sort": {"count": -1}},
    ]
    categories_raw = list(db.alerts.aggregate(cat_pipeline))
    categories = []
    for c in categories_raw:
        raw_cat = c.get("_id") or "unknown"
        categories.append({
            "category": raw_cat,
            "label": _CATEGORY_LABELS.get(raw_cat, raw_cat.replace("_", " ").title()),
            "count": c["count"],
            "high_risk": c.get("high_risk", 0),
            "avg_risk_score": round(c.get("avg_risk_score") or 0, 1),
            "platforms": [p for p in (c.get("platforms") or []) if p],
        })

    # ── 6. Overall sentiment analysis ────────────────────────────────────────
    sent_pipeline = [
        {"$match": {**aq, "llm_analysis.sentiment": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$llm_analysis.sentiment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    sentiment_raw = list(db.alerts.aggregate(sent_pipeline))
    # Normalise to negative / positive / neutral buckets
    sent_buckets: dict = {"negative": 0, "positive": 0, "neutral": 0, "others": {}}
    for s in sentiment_raw:
        label = _sentiment_label(s.get("_id", ""))
        if label in sent_buckets:
            sent_buckets[label] += s["count"]
        else:
            sent_buckets["others"][s.get("_id", "?")] = s["count"]

    # ── 7. High-threat posts with links ──────────────────────────────────────
    threat_posts_raw = list(db.alerts.find(
        {**aq, "priority": "HIGH"},
        ALERT_FIELDS,
    ).sort([("threat_details.risk_score", DESCENDING), ("created_at", DESCENDING)]).limit(20))
    threat_posts = []
    for d in threat_posts_raw:
        llm = d.get("llm_analysis") or {}
        threat = d.get("threat_details") or {}
        ts = d.get("created_at")
        vdata = d.get("velocity_data") or {}
        legal = _safe_join(d.get("legal_sections"))
        reasoning = (d.get("classification_explanation") or llm.get("reasoning") or "").strip()
        if "Primary AI analysis unavailable" in reasoning:
            reasoning = ""
        threat_posts.append({
            "id": str(d.get("id") or d["_id"]),
            "author_handle": (d.get("author_handle") or d.get("author") or "?").lstrip("@"),
            "platform": d.get("platform", "?"),
            "content_url": d.get("content_url", ""),
            "alert_type": d.get("alert_type", "?"),
            "source_category": d.get("source_category", "?"),
            "risk_score": threat.get("risk_score") or llm.get("score") or 0,
            "priority": d.get("priority", "?"),
            "sentiment": llm.get("sentiment", ""),
            "legal_sections": legal,
            "reasoning": reasoning[:250],
            "velocity": vdata.get("velocity", 0),
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts or ""),
            "keywords": d.get("matched_keywords_normalized") or [],
        })

    # ── 8. Summary stats ─────────────────────────────────────────────────────
    total_alerts     = db.alerts.count_documents(aq)
    high_alerts      = db.alerts.count_documents({**aq, "priority": "HIGH"})
    med_alerts       = db.alerts.count_documents({**aq, "priority": "MEDIUM"})
    active_alerts    = db.alerts.count_documents({**aq, "status": "active"})
    escalated_alerts = db.alerts.count_documents({**aq, "status": "escalated"})
    total_grievances = db.grievances.count_documents(aq)

    # ── 9. Dial 100 calls in window ──────────────────────────────────────────
    # Real records use either `date` (call timestamp) or `createdAt`.
    dial100_q = {"$or": [
        {"date":      {"$gte": cutoff}},
        {"createdAt": {"$gte": cutoff}},
    ]}
    dial100_total = (
        db.dial100incidents.count_documents(dial100_q)
        if "dial100incidents" in db.list_collection_names() else 0
    )

    # ── 10. Grievance-family breakdown ───────────────────────────────────────
    existing_cols = set(db.list_collection_names())
    grievance_breakdown = {
        "grievances":                  db.grievances.count_documents(aq) if "grievances" in existing_cols else 0,
        "criticism_reports":           0,
        "suggestion_reports":          0,
        "grievance_workflow_reports":  0,
        "workflow_escalated":          0,
        "workflow_pending":            0,
        "workflow_closed":             0,
    }
    if "criticism_reports" in existing_cols:
        grievance_breakdown["criticism_reports"] = db.criticism_reports.count_documents(
            {"$or": [{"created_at": {"$gte": cutoff}}, {"createdAt": {"$gte": cutoff}}]}
        )
    if "suggestion_reports" in existing_cols:
        grievance_breakdown["suggestion_reports"] = db.suggestion_reports.count_documents(
            {"$or": [{"created_at": {"$gte": cutoff}}, {"createdAt": {"$gte": cutoff}}]}
        )
    if "grievance_workflow_reports" in existing_cols:
        wf_q = {"$or": [{"created_at": {"$gte": cutoff}}, {"createdAt": {"$gte": cutoff}}]}
        grievance_breakdown["grievance_workflow_reports"] = db.grievance_workflow_reports.count_documents(wf_q)
        grievance_breakdown["workflow_escalated"] = db.grievance_workflow_reports.count_documents(
            {**wf_q, "status": "ESCALATED"}
        )
        grievance_breakdown["workflow_pending"] = db.grievance_workflow_reports.count_documents(
            {**wf_q, "status": "PENDING"}
        )
        grievance_breakdown["workflow_closed"] = db.grievance_workflow_reports.count_documents(
            {**wf_q, "status": "CLOSED"}
        )

    # ── 11. Events: fetched vs relevant ──────────────────────────────────────
    events_breakdown = {"fetched": 0, "relevant": 0, "active": 0, "recent": []}
    if "events" in existing_cols:
        ev_q = {"$or": [{"created_at": {"$gte": cutoff}}, {"start_date": {"$gte": cutoff}}]}
        events_breakdown["fetched"] = db.events.count_documents(ev_q)
        # "Relevant" = events that produced alerts in the same window (linked via event_ids)
        # OR are currently active.
        events_breakdown["active"] = db.events.count_documents(
            {**ev_q, "status": {"$in": ["active", "planned"]}}
        )
        try:
            relevant_event_ids = db.alerts.distinct("event_ids", aq) or []
            relevant_event_ids = [e for e in relevant_event_ids if e]
            events_breakdown["relevant"] = (
                db.events.count_documents({"_id": {"$in": relevant_event_ids}})
                if relevant_event_ids else events_breakdown["active"]
            )
        except Exception:
            events_breakdown["relevant"] = events_breakdown["active"]
        # Top 10 recent events
        recent_ev = list(db.events.find(
            ev_q,
            {"_id": 1, "name": 1, "start_date": 1, "end_date": 1, "location": 1,
             "status": 1, "platforms": 1, "keywords": 1},
        ).sort("start_date", DESCENDING).limit(10))
        for e in recent_ev:
            kws = ", ".join((k.get("keyword") or "") for k in (e.get("keywords") or [])[:5])
            events_breakdown["recent"].append({
                "id":         str(e.get("_id")),
                "name":       e.get("name", "?"),
                "status":     e.get("status", "?"),
                "location":   e.get("location", "?"),
                "platforms":  e.get("platforms") or [],
                "keywords":   kws,
                "start_date": e["start_date"].isoformat()
                              if isinstance(e.get("start_date"), datetime) else "",
                "end_date":   e["end_date"].isoformat()
                              if isinstance(e.get("end_date"), datetime) else "",
            })

    # ── 12. Top 50 alerts (full list with URLs) ──────────────────────────────
    top_50_raw = list(db.alerts.find(
        aq, ALERT_FIELDS,
    ).sort([
        ("priority", DESCENDING),  # HIGH > MEDIUM > LOW alphabetically too
        ("threat_details.risk_score", DESCENDING),
        ("created_at", DESCENDING),
    ]).limit(50))
    top_50_alerts = []
    for d in top_50_raw:
        llm = d.get("llm_analysis") or {}
        threat = d.get("threat_details") or {}
        ts = d.get("created_at")
        top_50_alerts.append({
            "id":             str(d.get("id") or d["_id"]),
            "author_handle":  (d.get("author_handle") or d.get("author") or "?").lstrip("@"),
            "platform":       d.get("platform", "?"),
            "content_url":    d.get("content_url", ""),
            "alert_type":     d.get("alert_type", "?"),
            "source_category": d.get("source_category", "?"),
            "priority":       d.get("priority", "?"),
            "status":         d.get("status", "?"),
            "risk_score":     threat.get("risk_score") or llm.get("score") or 0,
            "sentiment":      llm.get("sentiment", ""),
            "title":          (d.get("title") or "")[:120],
            "timestamp":      ts.isoformat() if isinstance(ts, datetime) else str(ts or ""),
        })

    # ── 13. Top 5 concepts (largest source_categories with sample handles) ────
    concept_pipeline = [
        {"$match": {**aq, "source_category": {"$ne": None, "$ne": ""}}},
        {"$group": {
            "_id": "$source_category",
            "count":     {"$sum": 1},
            "high":      {"$sum": {"$cond": [{"$eq": ["$priority", "HIGH"]}, 1, 0]}},
            "platforms": {"$addToSet": "$platform"},
            "handles":   {"$addToSet": {"$ifNull": ["$author_handle", "$author"]}},
            "sample_url": {"$first": "$content_url"},
            "avg_risk":   {"$avg": "$threat_details.risk_score"},
        }},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    top_concepts_raw = list(db.alerts.aggregate(concept_pipeline))
    top_concepts = []
    for c in top_concepts_raw:
        raw_cat = c.get("_id") or "unknown"
        handles = [h for h in (c.get("handles") or []) if h][:5]
        top_concepts.append({
            "concept":        raw_cat,
            "label":          _CATEGORY_LABELS.get(raw_cat, raw_cat.replace("_", " ").title()),
            "alert_count":    c["count"],
            "high_risk":      c.get("high", 0),
            "avg_risk_score": round(c.get("avg_risk") or 0, 1),
            "platforms":      [p for p in (c.get("platforms") or []) if p],
            "sample_handles": handles,
            "sample_url":     c.get("sample_url", ""),
        })

    # ── 14. Profiles (monitored sources) ─────────────────────────────────────
    profiles_breakdown = {
        "monitored":     0,
        "active":        0,
        "added_24h":     0,
        "deleted_24h":   0,
        "high_risk":     0,
    }
    if "sources" in existing_cols:
        profiles_breakdown["monitored"] = db.sources.estimated_document_count()
        profiles_breakdown["active"]    = db.sources.count_documents({"is_active": True})
        profiles_breakdown["added_24h"] = db.sources.count_documents({"created_at": {"$gte": cutoff}})
        profiles_breakdown["high_risk"] = db.sources.count_documents(
            {"risk_level": {"$in": ["high", "critical"]}}
        )
    # Deletions tracked via audit_logs (resource_type='source', action containing 'delete')
    if "audit_logs" in existing_cols:
        try:
            profiles_breakdown["deleted_24h"] = db.audit_logs.count_documents({
                "$and": [
                    {"$or": [{"created_at": {"$gte": cutoff}}, {"createdAt": {"$gte": cutoff}}, {"timestamp": {"$gte": cutoff}}]},
                    {"resource_type": {"$regex": "source", "$options": "i"}},
                    {"action":        {"$regex": "delete", "$options": "i"}},
                ]
            })
        except Exception:
            pass

    # ── 15. Top 10 monitored keywords (configured watch-words) ───────────────
    top_keywords_10 = []
    if "keywords" in existing_cols:
        kw_docs = list(db.keywords.find(
            {"is_active": {"$ne": False}},
            {"_id": 0, "keyword": 1, "category": 1, "language": 1, "weight": 1,
             "is_active": 1, "created_at": 1},
        ).sort([("weight", DESCENDING), ("created_at", DESCENDING)]).limit(10))
        for k in kw_docs:
            top_keywords_10.append({
                "keyword":  k.get("keyword", "?"),
                "category": k.get("category", "?"),
                "language": k.get("language", "?"),
                "weight":   k.get("weight", 0),
                "active":   k.get("is_active", True),
            })

    client.close()

    return {
        "window_hours": hours,
        "window_start": cutoff.isoformat(),
        "window_end": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_alerts":     total_alerts,
            "high_alerts":      high_alerts,
            "medium_alerts":    med_alerts,
            "active_alerts":    active_alerts,
            "escalated_alerts": escalated_alerts,
            "total_grievances": total_grievances,
            "dial100_total":    dial100_total,
            "threat_rate_pct":  round(high_alerts / max(total_alerts, 1) * 100, 1),
        },
        "trending_keywords":   trending_keywords,
        "platform_data":       platform_data,
        "viral_posts":         viral_posts,
        "active_accounts":     active_accounts,
        "categories":          categories,
        "sentiment":           sent_buckets,
        "threat_posts":        threat_posts,
        # New sections (officer-requested)
        "dial100_total":       dial100_total,
        "grievance_breakdown": grievance_breakdown,
        "events_breakdown":    events_breakdown,
        "top_50_alerts":       top_50_alerts,
        "top_concepts":        top_concepts,
        "profiles_breakdown":  profiles_breakdown,
        "top_keywords_10":     top_keywords_10,
    }


def _build_dir(hours: int = 24, force: bool = False) -> dict:
    """Generate (or return cached) Daily Intelligence Report."""
    now = datetime.now(timezone.utc)
    date_key = now.strftime("%Y-%m-%d")
    cache_key = f"{date_key}_{hours}h"

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    cache = db[DIR_COLLECTION]
    cache.create_index([("cache_key", 1)], unique=True, background=True)
    if not force:
        existing = cache.find_one({"cache_key": cache_key})
        if existing:
            existing.pop("_id", None)
            if isinstance(existing.get("generated_at"), datetime):
                existing["generated_at"] = existing["generated_at"].isoformat()
            client.close()
            return existing
    client.close()

    raw = _collect_dir_data(hours=hours)
    s = raw["stats"]
    top_kw = ", ".join(
        f"{kw['_id']} ({kw['count']})" for kw in raw["trending_keywords"][:10]
    ) or "(none)"
    top_accounts = ", ".join(
        f"@{a['handle']} ({a['alert_count']} alerts)" for a in raw["active_accounts"][:8]
    ) or "(none)"
    top_cats = ", ".join(
        f"{c['label']} ({c['count']})" for c in raw["categories"][:8]
    ) or "(none)"
    plat_summary = ", ".join(
        f"{p['_id']} ({p['total']})" for p in raw["platform_data"][:6]
    ) or "(none)"
    viral_summary = "\n".join(
        f"  • @{v['author_handle']} on {v['platform']}: {v['velocity_metric']} velocity={v['velocity']} | "
        f"cat={v['source_category']} | URL: {v['content_url'] or 'N/A'}"
        for v in raw["viral_posts"][:8]
    ) or "(none)"
    threat_summary = "\n".join(
        f"  • @{t['author_handle']} on {t['platform']} | risk={t['risk_score']}% "
        f"| type={t['alert_type']} | {t['reasoning'][:150]} | URL: {t['content_url'] or 'N/A'}"
        for t in raw["threat_posts"][:10]
    ) or "(none)"
    sent_info = (
        f"Negative: {s.get('negative', raw['sentiment'].get('negative', 0))} | "
        f"Positive: {raw['sentiment'].get('positive', 0)} | "
        f"Neutral: {raw['sentiment'].get('neutral', 0)}"
    )

    prompt = textwrap.dedent(f"""\
        You are SOC-EYE Daily Intelligence Briefer for Telangana Police
        (CP, DCP, SOC analysts). Generate a comprehensive Daily Intelligence Report
        for the last {hours} hours as of {now.strftime('%d-%b-%Y %H:%M')} IST.

        === RAW INTELLIGENCE DATA ===
        PERIOD: Last {hours} hours
        Total Alerts: {s['total_alerts']} | HIGH: {s['high_alerts']} | MEDIUM: {s['medium_alerts']}
        Grievances: {s['total_grievances']}
        Threat Rate: {s['threat_rate_pct']}%

        TOP TRENDING KEYWORDS/TOPICS: {top_kw}
        MOST ACTIVE ACCOUNTS: {top_accounts}
        CATEGORY BREAKDOWN: {top_cats}
        PLATFORM BREAKDOWN: {plat_summary}
        SENTIMENT: {sent_info}

        VIRAL/HIGH-VELOCITY POSTS:
        {viral_summary}

        HIGH-THREAT POSTS (with URLs):
        {threat_summary}

        === END DATA ===

        Write the report using EXACTLY these 7 sections (Markdown ## headings):

        ## Executive Summary
        3-4 sentences. Overall threat landscape for the period. Highlight the most
        critical development that CP needs to know immediately. Include key numbers.

        ## Trending Topics & Keywords
        List top 8 trending topics/keywords. For each:
        • **[keyword]** — X alerts | platforms: ... | risk level (HIGH/MEDIUM/LOW)
        • Brief note on why it is trending or its significance.

        ## Viral & High-Velocity Posts
        List the top 5 viral posts. For each:
        • **@handle** on [platform] — [metric] velocity=[value] | [category]
        • [View Post](URL) | Risk: X% | Significance: one line

        ## Platform & Sentiment Analysis
        For each major platform: alert count, % high-risk, dominant sentiment.
        Overall sentiment breakdown (negative/positive/neutral %).
        Pattern note: which platform is highest-risk this period.

        ## High-Threat Intelligence
        Top 5 HIGH-priority alerts requiring immediate attention. For each:
        • **@handle** — [alert_type] | Risk: X% | [View Post](URL)
        • Threat: what law-and-order risk it poses
        • Action: specific recommendation (FIR / takedown / escalate / monitor)

        ## Active Threat Actors
        Top 5 most active accounts generating alerts. For each:
        • **@handle** — X alerts | categories | platforms
        • Threat level: HIGH/MEDIUM/LOW | Recommended action

        ## Analyst's Assessment
        3-4 bullets covering:
        • Dominant threat category this period and why
        • Any coordinated activity or emerging patterns
        • Jurisdiction/district-specific risks if detectable
        • One forward-looking note (watch-list for next 24h)

        CRITICAL RULES:
        - Use ONLY real data from the sections above. NEVER invent handles/URLs/incidents.
        - For every post mentioned: include [View Post](URL) if URL is available.
        - If a section has no data, write "No significant items in this period."
        - Be specific with numbers. Officers need hard data, not vague summaries.
        - Bold **@handles** and key figures. Use `code` for legal sections.
    """)

    llm_summary = ""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.15, "num_ctx": 8192, "num_predict": 1800},
            },
            timeout=600,
        )
        resp.raise_for_status()
        llm_summary = resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("DIR LLM generation failed: %s", exc)
        llm_summary = "_(AI narrative unavailable — Ollama not reachable.)_"

    doc = {
        "cache_key":   cache_key,
        "date":        date_key,
        "hours":       hours,
        "generated_at": now,
        "window_start": raw["window_start"],
        "window_end":   raw["window_end"],
        "stats":        raw["stats"],
        "trending_keywords": raw["trending_keywords"],
        "platform_data":     raw["platform_data"],
        "viral_posts":       raw["viral_posts"],
        "active_accounts":   raw["active_accounts"],
        "categories":        raw["categories"],
        "sentiment":         raw["sentiment"],
        "threat_posts":      raw["threat_posts"],
        "llm_summary":       llm_summary,
    }

    try:
        c2 = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        c2[DB_NAME][DIR_COLLECTION].update_one(
            {"cache_key": cache_key}, {"$set": doc}, upsert=True
        )
        c2.close()
    except Exception as e:
        logger.warning("DIR cache write failed: %s", e)

    doc.pop("_id", None)
    doc["generated_at"] = doc["generated_at"].isoformat()
    return doc


@app.get("/api/rag/dir")
def get_dir(hours: int = 24, force: bool = False):
    """Return the Daily Intelligence Report for the last N hours."""
    try:
        doc = _build_dir(hours=hours, force=force)
        return doc
    except Exception as e:
        logger.exception("DIR build failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/dir/history")
def dir_history(limit: int = 14):
    """List historical Daily Intelligence Reports, newest first."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col = client[DB_NAME][DIR_COLLECTION]
        docs = list(
            col.find({}, {
                "trending_keywords": 0, "viral_posts": 0,
                "active_accounts": 0, "threat_posts": 0,
            }).sort("generated_at", DESCENDING).limit(min(limit, 60))
        )
        client.close()
        for d in docs:
            d.pop("_id", None)
            if isinstance(d.get("generated_at"), datetime):
                d["generated_at"] = d["generated_at"].isoformat()
        return {"reports": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _dir_scheduler_loop():
    """Generate DIR once per day at DIR_HOUR_UTC."""
    while not _scheduler_state["stop"].is_set():
        now = datetime.now(timezone.utc)
        target = now.replace(hour=DIR_HOUR_UTC, minute=30, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        sleep_s = (target - now).total_seconds()
        logger.info("Next DIR generation at %s UTC (in %.0fs)", target.isoformat(), sleep_s)
        if _scheduler_state["stop"].wait(sleep_s):
            return
        try:
            _build_dir(hours=24, force=True)
            logger.info("Daily Intelligence Report generated.")
        except Exception:
            logger.exception("DIR generation failed")


@app.on_event("startup")
def _start_dir_scheduler():
    t = threading.Thread(target=_dir_scheduler_loop, name="rag-dir-scheduler", daemon=True)
    t.start()


@app.get("/api/rag/top-alerts/history")
def top_alerts_history(limit: int = 14):
    """List past top-alert runs stored in rag_top_alerts, newest first."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col = client[DB_NAME]["rag_top_alerts"]
        docs = list(col.find({}, {"alert_meta": 0}).sort("generated_at", DESCENDING).limit(min(limit, 60)))
        client.close()
        for d in docs:
            d.pop("_id", None)
            if isinstance(d.get("generated_at"), datetime):
                d["generated_at"] = d["generated_at"].isoformat()
        return {"runs": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/top-alerts/cached")
def top_alerts_cached(date: Optional[str] = None, hours: int = 24, mode: Optional[str] = None):
    """Return the cached top-alerts for a given date (default=today).
    `mode` can be omitted (legacy single-list cache) or "by_category" for the
    per-category cache. Returns alert_ids + alert_meta so frontend can fetch
    full cards via /api/alerts/bulk."""
    date_key = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        col = client[DB_NAME]["rag_top_alerts"]
        if mode == "by_category":
            query = {"date": date_key, "hours": hours, "mode": "by_category"}
        else:
            # Legacy cache docs were written without a `mode` field; match those.
            query = {"date": date_key, "hours": hours, "mode": {"$exists": False}}
        doc = col.find_one(query, {"_id": 0})
        client.close()
        if not doc:
            return {"found": False, "date": date_key}
        if isinstance(doc.get("generated_at"), datetime):
            doc["generated_at"] = doc["generated_at"].isoformat()
        doc["found"] = True
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/refresh-cache")
def refresh_cache():
    """Rebuild the local vector search cache from MongoDB."""
    try:
        store = VectorStore(MONGODB_URI, DB_NAME, VECTOR_COLLECTION)
        store.refresh_cache()
        count = len(store._texts)
        store.close()
        return {"message": f"Cache rebuilt with {count} vectors."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
