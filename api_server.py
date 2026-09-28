"""
api_server.py — FastAPI bridge for the RAG pipeline.

Exposes REST endpoints so the Node.js backend (or any client) can:
  - GET  /api/rag/health          → PostgreSQL + vLLM embedding + vLLM LLM health
  - GET  /api/rag/collections     → list all PostgreSQL collections
  - POST /api/rag/query           → synchronous question (blocks until answer)
  - POST /api/rag/query/async     → enqueue question, returns {job_id} immediately
  - GET  /api/rag/jobs/{job_id}   → poll status/result of an async job
  - GET  /api/rag/jobs            → list recent jobs (optionally by collection)
  - POST /api/rag/ingest          → trigger ingestion for a collection
  - GET  /api/rag/stats           → vector store stats

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8100 --reload
"""

import hashlib
import logging
import os
import re
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.wsgi import WSGIMiddleware

from assistant import Assistant, SYSTEM_PROMPT, CONTEXT_SEPARATOR, _smalltalk_response
from embedder import get_embedder
from llm_client import generate as llm_generate, LLM_BASE_URL, LLM_MODEL
import llm_client
import intent as intent_router
try:
    from osint_portal.app import app as osint_portal_app
except ModuleNotFoundError:          # sub-app not vendored in this repo
    osint_portal_app = None
from processor import PostgresStreamProcessor, DocumentConverter
from chunker import TokenAwareChunker
from vector_store import VectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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
# Browser test console (static/index.html) — served from the API itself so it
# shares an origin and needs no separate web server.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def _root_redirect():
        return RedirectResponse(url="/ui/")

if osint_portal_app is not None:
    app.mount("/osint", WSGIMiddleware(osint_portal_app))
else:
    logger.warning("osint_portal not installed — /osint not mounted.")

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
    # When False, skip PostgreSQL + vector retrieval entirely and answer the
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


def _get_cutoff_date(days: int) -> Optional[datetime]:
    if not days or days <= 0:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/rag/health")
def health():
    """Check PostgreSQL, the embedding host and the LLM endpoint."""
    status = {"postgresql": False, "embedding": False, "llm": False}
    try:
        from db import get_pool
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        status["postgresql"] = True
    except Exception as e:
        logger.error("PostgreSQL health check failed: %s", e)

    embedding = get_embedder().probe()
    status["embedding"] = embedding["healthy"]

    import llm_client
    status["llm"] = llm_client.check_health()

    overall = all(status.values())
    return {
        "healthy": overall,
        "services": status,
        "llm": {
            "status": "HEALTHY" if status["llm"] else "UNHEALTHY",
            "model": llm_client.LLM_MODEL,
            "endpoint": llm_client.LLM_BASE_URL,
        },
    }


@app.get("/api/rag/collections")
def list_collections():
    """Return all collection names in the configured database."""
    try:
        from db import get_pool
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT collection_name FROM source_documents")
                collections = sorted([row[0] for row in cur.fetchall()])
        return {"database": "postgresql", "collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Filled in by the startup warm-up thread; surfaced via /api/rag/health.
_EMBED_WARMUP: dict = {"ok": None, "error": None}

_GLOBAL_STORES: dict = {}
_GLOBAL_STORES_LOCK = threading.Lock()


def _get_store(vec_col: str) -> VectorStore:
    with _GLOBAL_STORES_LOCK:
        s = _GLOBAL_STORES.get(vec_col)
        if s is None:
            s = VectorStore()
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
    from db import get_pool
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT metadata->>'source_collection' FROM vector_embeddings")
            existing = {f"{VECTOR_COLLECTION}_{row[0]}" for row in cur.fetchall() if row[0]}
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


# Qualifiers a count question can carry ("how many HIGH alerts", "how many
# ESCALATED grievances"). Without these the count ignored every adjective in
# the question and reported the collection total instead, so "how many high
# alerts" answered 130,356 when the real figure was 17,857.
#
# Each entry is (pattern, mongo clause, label shown to the user). Order
# matters: the most specific phrasing has to be tried first, which is why
# "high priority" precedes the bare "high".
#
# `priority` and `risk_level` are DIFFERENT fields here and they disagree:
# 13,039 docs are priority=LOW but risk_level=high, and no document with a
# "HIGH Risk" title has priority=HIGH. Alert titles are generated from
# risk_level, so that is what an officer reading the portal means by "high
# alert". "High priority" is matched separately and the two are never merged.
_COUNT_FILTERS: dict = {
    "alerts": [
        (r"\bhigh\s+priorit(y|ies)\b|\bpriority\s*[:=]?\s*high\b",
         {"priority": "HIGH"}, "priority=HIGH"),
        (r"\bmedium\s+priorit(y|ies)\b|\bpriority\s*[:=]?\s*medium\b",
         {"priority": "MEDIUM"}, "priority=MEDIUM"),
        (r"\blow\s+priorit(y|ies)\b|\bpriority\s*[:=]?\s*low\b",
         {"priority": "LOW"}, "priority=LOW"),
        (r"\b(high|critical|severe)(\s+(risk|severity|level))?\b",
         {"risk_level": "high"}, "risk_level=high"),
        (r"\bmedium(\s+(risk|severity|level))?\b|\bmoderate\b",
         {"risk_level": "medium"}, "risk_level=medium"),
        (r"\blow(\s+(risk|severity|level))?\b",
         {"risk_level": "low"}, "risk_level=low"),
        (r"\bunread\b", {"is_read": False}, "is_read=false"),
        (r"\backnowledged\b", {"status": "acknowledged"}, "status=acknowledged"),
        (r"\bescalated\b", {"status": "escalated"}, "status=escalated"),
        (r"\bfalse\s+positives?\b", {"status": "false_positive"},
         "status=false_positive"),
        (r"\bactive\b|\bopen\b|\bunresolved\b", {"status": "active"},
         "status=active"),
        (r"\bunder\s+investigation\b|\binvestigations?\b",
         {"is_investigation": True}, "is_investigation=true"),
        (r"\b(twitter|x\.com)\b", {"platform": "x"}, "platform=x"),
        (r"\binstagram\b|\binsta\b", {"platform": "instagram"},
         "platform=instagram"),
        (r"\byoutube\b", {"platform": "youtube"}, "platform=youtube"),
        (r"\bfacebook\b|\bfb\b", {"platform": "facebook"}, "platform=facebook"),
        (r"\bai\s*[-_]?\s*risk\b", {"alert_type": "ai_risk"},
         "alert_type=ai_risk"),
        (r"\bvelocity\b", {"alert_type": "velocity"}, "alert_type=velocity"),
        (r"\bkeyword\s*[-_]?\s*risk\b", {"alert_type": "keyword_risk"},
         "alert_type=keyword_risk"),
    ],
}

# Words that carry no filtering meaning, so their presence must not trigger
# the "could not interpret" caveat below.
_COUNT_FILLER = frozenset("""
a an the how many much count number total there is are was were do does did
of in on at for from by with to and or not no me us i you we show tell give
list please currently right now over during within between all any each
some whats what which who whose why when where have has had been being
post posts posting record records row rows entry entries item items
doc docs document documents thing things data database db collection
collections so far up till until as per about regarding across overall
altogether alert alerts
""".split())


def _extract_count_filters(q: str, target_col: str, consumed: str) -> tuple:
    """Turn the qualifiers in a count question into a PostgreSQL filter.

    Returns (clauses, labels, unknown_terms). *consumed* is the text already
    accounted for -- the matched collection keyword and the time phrase -- and
    is removed before scanning for leftovers, so only genuinely uninterpreted
    words land in unknown_terms.
    """
    clauses: dict = {}
    labels: list = []
    remaining = " %s " % q
    for token in consumed.lower().split():
        remaining = remaining.replace(token, " ")
    # Every way of naming this collection is part of the question's subject,
    # not a qualifier: "how many dial 100 incidents" matched on "incidents",
    # which would otherwise leave "dial" looking uninterpreted.
    for kw in sorted(_COUNT_KEYWORDS.get(target_col, []), key=len, reverse=True):
        remaining = re.sub(r"\b%s\b" % re.escape(kw), " ", remaining, flags=re.I)
    for pattern, clause, label in _COUNT_FILTERS.get(target_col, []):
        # A field already pinned by a more specific phrase wins: "high
        # priority" must not then also be read as risk_level=high.
        if any(k in clauses for k in clause):
            continue
        m = re.search(pattern, remaining, re.I)
        if not m:
            continue
        clauses.update(clause)
        labels.append(label)
        remaining = remaining[:m.start()] + " " + remaining[m.end():]
    # Removing whole phrases can still leave a fragment of the collection's
    # own name ("dial 100 calls" matches "100 calls", leaving "dial"), so any
    # word used in any of its keywords counts as subject, not qualifier.
    subject = {t for kw in _COUNT_KEYWORDS.get(target_col, [])
               for t in kw.lower().split()}
    unknown = [w for w in re.findall(r"[a-z][a-z0-9_'-]{2,}", remaining.lower())
               if w not in _COUNT_FILLER and w not in subject]
    return clauses, labels, unknown


def _count_noun(kw: str, n: int) -> str:
    """Agree the matched keyword with the number in front of it.

    The keyword is whatever the question happened to use, so "how many high
    alert posts" matched the singular and read as "17,857 alert". Only simple
    one-word keywords are touched; multi-word ones like "dial 100" are left
    exactly as matched.
    """
    if n == 1 or " " in kw or kw.endswith("s"):
        return kw
    return kw + ("ies" if kw.endswith("y") and kw[-2:-1] not in "aeiou" else "s")


def _count_fast_path(question: str, default_window_days: Optional[int]) -> Optional[dict]:
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
    """Generate an answer on the configured LLM endpoint (see llm_client.py).

    Retries, context fitting and error formatting live in llm_client; on
    terminal failure this returns a marked "_(...)_" string so callers can
    still surface the retrieved PostgreSQL evidence to the user.
    """
    return llm_generate(prompt, temperature=0.2, top_p=0.9, max_tokens=2048)


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
    verdict = intent_router.classify(question)
    if verdict.intent is intent_router.Intent.GREETING:
        return {"answer": intent_router.greeting_answer(question), "sources": [],
                "question": question, "smalltalk": True, "scope": "chat_only",
                "intent": verdict.intent.value, "rag_used": False}
    if verdict.intent is intent_router.Intent.CAPABILITY:
        return {"answer": intent_router.CAPABILITY_ANSWER, "sources": [],
                "question": question, "scope": "chat_only",
                "intent": verdict.intent.value, "rag_used": False}
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
    # Only rescue a genuinely failed generation. Padding short answers used to
    # force a 10-line briefing even when the context did not support one, which
    # is exactly the hallucination pressure the intent router removes.
    needs_evidence = (
        not answer
        or answer.startswith("_(LLM generation failed")
        or answer.startswith("_(Error generating answer")
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
# For EVERY non-count question we pull a rich context directly from PostgreSQL
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

# Keyword → PostgreSQL field:value filters for smarter retrieval
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
# Relevance filtering for the DB-recency pull
#
# _CATEGORY_HINTS above only covers ~9 fixed phrasings. Every other question
# ("potholes", "political rally", "telegram messages", ...) fell through to
# an unfiltered "most recent HIGH/MEDIUM alerts" query, so the SAME handful of
# recent alerts appeared as context/sources regardless of what was asked —
# they then tied with genuine vector hits in the RRF fusion below and rode
# along into the answer. _extract_relevance_terms + _relevance_or_clause keep
# that recency pull on-topic by requiring the question's own content words to
# appear somewhere in the candidate record; the unfiltered pull remains as a
# fallback only when nothing on-topic exists, so a broad or truly generic
# question still gets real data instead of an empty context block.
# ---------------------------------------------------------------------------
_RELEVANCE_STOPWORDS = _COUNT_FILLER | frozenset(
    "recent latest current please related relating".split()
)


def _extract_relevance_terms(question: str) -> list:
    """Meaningful content words from the question (order-preserved, deduped)."""
    seen: list = []
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", question.lower()):
        if w in _RELEVANCE_STOPWORDS or w in seen:
            continue
        seen.append(w)
    return seen


ALERT_RELEVANCE_FIELDS = [
    "title", "source_category", "alert_type", "classification_explanation",
    "matched_keywords_normalized", "author_handle", "author",
]
GRIEVANCE_RELEVANCE_FIELDS = [
    "content.text", "context.in_reply_to.content.text", "tagged_account",
    "posted_by.handle", "complaint_code",
]


def _relevance_or_clause(terms: list, fields: list) -> Optional[dict]:
    """`$or` regex clause matching any *fields* against any *terms*, or None."""
    if not terms:
        return None
    term_re = "|".join(re.escape(t) for t in terms[:12])  # cap — very long questions
    return {"$or": [{f: {"$regex": term_re, "$options": "i"}} for f in fields]}


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





def _vector_candidates(question: str, cutoff_date: Optional[datetime] = None) -> list:
    """Semantic hits for *question*, globally across all PostgreSQL collections."""
    try:
        embedder = get_embedder()
        q_vec = embedder.embed_text(question)
        if q_vec is None:
            return []
        
        store = _get_store("global")
        hits = store.cosine_search(
            query_vector=q_vec, 
            top_k=25, 
            query_text=question,
            cutoff_date=cutoff_date
        )
        return hits
    except Exception as exc:
        logger.debug("vector enrichment skipped: %s", exc)
        return []


# Completion size for a grounded RAG answer. Smaller than the old 2048: the
# answer should be as long as the evidence warrants, and a smaller completion
# leaves more of the (currently tight) server context for actual evidence.
_RAG_COMPLETION_TOKENS = 1600


def _global_query(question: str, top_k: int, time_window_days: Optional[int]) -> dict:
    """Route by intent, then retrieve only what that intent actually needs.

    Previously every question — "hi" included — pulled alerts + grievances from
    PostgreSQL, appended vector hits, and asked for a 10+ line briefing. Now:

      greeting / capability / unsupported -> answered without touching PostgreSQL
      general knowledge                   -> LLM only, no SOC-EYE context
      data query                          -> retrieve, rank, budget, ground

    Retrieval itself is unchanged; what changed is whether it runs and how much
    of its output reaches the prompt.
    """
    started = time.time()
    verdict = intent_router.classify(question)

    log = {"intent": verdict.intent.value, "rag_used": False, "candidates": 0,
           "contexts": 0, "context_tokens": 0, "prompt_tokens": 0,
           "model": llm_client.LLM_MODEL, "llm_status": None}

    def _finish(payload: dict) -> dict:
        payload.setdefault("intent", verdict.intent.value)
        payload.setdefault("rag_used", log["rag_used"])
        payload.setdefault("context_count", log["contexts"])
        payload.setdefault("context_tokens", log["context_tokens"])
        payload.setdefault("prompt_tokens", log["prompt_tokens"])
        if verdict.unsupported:
            payload.setdefault("unsupported", [k for k, _ in verdict.unsupported])
        logger.info(
            "RAG q=%r intent=%s rag=%s candidates=%d (db=%s vec=%s) contexts=%d "
            "ctx_tok=%d prompt_tok=%s model=%s llm=%s %.1fs",
            question[:80], log["intent"], log["rag_used"], log["candidates"],
            log.get("db_candidates", 0), log.get("vector_candidates", 0),
            log["contexts"], log["context_tokens"], log["prompt_tokens"],
            log["model"], log["llm_status"], time.time() - started,
        )
        return payload

    # ── routes that need no retrieval ────────────────────────────────────────
    if verdict.intent is intent_router.Intent.GREETING:
        return _finish({"answer": intent_router.greeting_answer(question),
                        "sources": [], "question": question,
                        "scope": "greeting", "smalltalk": True})

    if verdict.intent is intent_router.Intent.CAPABILITY:
        return _finish({"answer": intent_router.CAPABILITY_ANSWER, "sources": [],
                        "question": question, "scope": "capability"})

    if verdict.intent is intent_router.Intent.UNSUPPORTED:
        return _finish({"answer": intent_router.limitation_notice(
                            verdict.unsupported, data_follows=False),
                        "sources": [], "question": question,
                        "scope": "unsupported"})

    if verdict.intent is intent_router.Intent.GENERAL:
        meta: dict = {}
        answer = llm_generate(
            f"{CHAT_ONLY_SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:",
            temperature=0.3, max_tokens=800, meta=meta)
        log["llm_status"] = meta.get("status")
        log["prompt_tokens"] = meta.get("prompt_tokens") or 0
        return _finish({"answer": answer, "sources": [], "question": question,
                        "scope": "general"})

    # ── counts answer exactly, without an LLM ────────────────────────────────
    fast = _count_fast_path(question, time_window_days)
    if fast is not None:
        log["rag_used"] = True
        return _finish(fast)

    # ── retrieve candidates (PostgreSQL native ONLY) ─────────────────────────
    log["rag_used"] = True
    days = time_window_days if time_window_days else 7

    vec_list = []
    # Assert safety to prevent silent fallback to localhost PostgreSQL for RAG
    for i, hit in enumerate(_vector_candidates(question, _get_cutoff_date(days))):
        meta_h = hit.get("metadata", {})
        vec_list.append({
            "text": (f"[VEC · id={meta_h.get('document_id', '')} "
                     f"· src={meta_h.get('source_collection', '')} "
                     f"· score={hit.get('score', 0):.2f}]\n"
                     f"{hit.get('text', '')[:900]}"),
            "origin": "vector",
            "rank": i,
            "score": float(hit.get("score") or 0.0),
            "document_id": meta_h.get("document_id", ""),
            "source_collection": meta_h.get("source_collection", ""),
        })

    candidates = vec_list
    candidates.sort(key=lambda c: c["score"], reverse=True)
    log["db_candidates"] = 0
    log["vector_candidates"] = len(vec_list)
    log["candidates"] = len(candidates)

    # ── budget: the RAG ceiling, or whatever the endpoint can take today ─────
    instructions_allowance = 600
    fixed = (intent_router.count_tokens(SYSTEM_PROMPT)
             + intent_router.count_tokens(question)
             + instructions_allowance)
    ctx_budget = max(512, min(
        intent_router.MAX_CONTEXT_TOKENS,
        llm_client.prompt_budget(_RAG_COMPLETION_TOKENS) - fixed,
    ))

    selection = intent_router.select_context(
        candidates, max_tokens=ctx_budget,
        complex_question=verdict.complex_question,
        soft_max=verdict.suggested_contexts,
    )
    log["contexts"] = len(selection.items)
    log["context_tokens"] = selection.tokens

    if selection.items:
        context_block = "\n\n---\n\n".join(c["text"] for c in selection.items)
    else:
        context_block = "(No records found across the queried modules for this time window.)"

    window_label = f"last {days} day(s)" if days else "all time"
    guardrail = (intent_router.capability_guardrail(verdict.unsupported)
                 if verdict.unsupported else "")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current date and time: {now_utc}\n\n"
        f"=== DATABASE CONTEXT ({window_label}) ===\n"
        f"{context_block}\n"
        f"=== END CONTEXT ===\n\n"
        f"User question: {question}\n"
        f"{guardrail}\n"
        f"OUTPUT REQUIREMENTS — read carefully:\n"
        f"• Answer ONLY from the DATABASE CONTEXT above. Every number, handle, "
        f"FIR, location and URL must appear in that context.\n"
        f"• Never invent records, counts, names or links. Do not fill gaps with "
        f"general knowledge or assumptions.\n"
        f"• If the context does not contain enough information to answer, say so "
        f"plainly in one or two lines — that is a correct answer, not a failure. "
        f"Do not pad it out.\n"
        f"• For every record you cite, append `[View Post](URL)` using that "
        f"record's URL field. If it has none, write `_(no URL on file)_`.\n"
        f"• Length should match the evidence: brief when the context is thin, "
        f"fuller when it is rich.\n\nAnswer:"
    )
    log["prompt_tokens"] = intent_router.count_tokens(prompt)

    meta = {}
    answer = llm_generate(prompt, temperature=0.2, top_p=0.9,
                          max_tokens=_RAG_COMPLETION_TOKENS, meta=meta)
    log["llm_status"] = meta.get("status")
    if meta.get("prompt_tokens"):
        log["prompt_tokens"] = meta["prompt_tokens"]

    # Capability boundary is stated in code, not left to the model — it cannot
    # talk its way into claiming it emailed or exported anything.
    if verdict.unsupported:
        answer = (intent_router.limitation_notice(verdict.unsupported,
                                                  data_follows=True)
                  + "\n\n" + answer)

    # Only rescue the answer when the LLM actually failed. A short, grounded
    # reply is now a valid outcome, so it is no longer padded.
    answer = _ensure_minimum_answer(answer, [c["text"] for c in selection.items], question)

    sources = []
    for i, item in enumerate(selection.items):
        text = item["text"]
        if item["origin"] == "vector":
            collection = item.get("source_collection") or "vector"
            document_id = item.get("document_id") or f"vec-{i + 1}"
        else:
            collection = text.split("|", 1)[0].lstrip("[").strip().lower() or "record"
            document_id = f"db-{i + 1}"
        sources.append({
            "collection": collection,
            "document_id": document_id,
            "score": round(float(item.get("score") or 0.0), 4),
            "preview": text[:200],
        })

    return _finish({
        "answer": answer, "sources": sources, "question": question,
        "scope": "vec", "window_doc_count": log["vector_candidates"],
        "time_window_days": days,
        "candidates_considered": selection.considered,
        "dropped_by_budget": selection.dropped_by_budget,
        "dropped_by_relevance": selection.dropped_by_relevance,
    })


@app.post("/api/rag/query")
def query(req: QueryRequest):
    """Ask a question. If `collection` is omitted (or 'all'), searches across
    every indexed collection — chat-bot style — instead of one specific source.

    When ``use_db`` is False the DB + vector layers are bypassed entirely and
    the question is sent to the LLM as pure conversational chat.
    """
    if not req.use_db:
        return _chat_only_answer(req.question)

    # Greetings, capability questions and pure unsupported-action requests are
    # answered the same way whatever collection is selected — and must never
    # trigger retrieval. _global_query owns those branches.
    if not intent_router.classify(req.question).needs_rag:
        out = _global_query(req.question, req.top_k, req.time_window_days)
        out["time_window_days"] = req.time_window_days
        return out

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
        raise HTTPException(status_code=500, detail=f"PostgreSQL error: {e}")

    bot = Assistant(
        llm_model=LLM_MODEL,
        top_k=req.top_k,
        source_collection=collection if use_source_filter else None,
    )

    cutoff_date = _get_cutoff_date(req.time_window_days or 0)
    result = bot.ask(req.question, cutoff_date=cutoff_date)
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
    if cutoff_date is not None:
        result["window_applied"] = True
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



def _save_job(job_id: str, data: dict):
    from db import get_pool
    import json
    import uuid
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Upsert
            cur.execute("SELECT id FROM source_documents WHERE collection_name='rag_jobs' AND document_data->>'job_id' = %s", (job_id,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE source_documents SET document_data = %s WHERE id = %s", (json.dumps(data, default=str), row[0]))
            else:
                cur.execute("INSERT INTO source_documents (id, collection_name, created_at, document_data) VALUES (%s, %s, %s, %s)", (str(uuid.uuid4()), 'rag_jobs', datetime.now(timezone.utc), json.dumps(data, default=str)))
            conn.commit()

def _get_job(job_id: str) -> dict:
    from db import get_pool
    import json
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT document_data FROM source_documents WHERE collection_name='rag_jobs' AND document_data->>'job_id' = %s", (job_id,))
            row = cur.fetchone()
            if row:
                return row['document_data']
    return None

def _run_query_job(job_id: str, collection: str, question: str, top_k: int, use_source_filter: bool, time_window_days: Optional[int]):
    started = datetime.now(timezone.utc)
    try:
        from db import get_pool
        pool = get_pool()
        vec_count = 0
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings")
                vec_count = cur.fetchone()[0]

        if vec_count == 0:
            answer = "No indexed data found. Indexing may be in progress or failed."
            finished = datetime.now(timezone.utc)
            _save_job(job_id, {
                "job_id": job_id, "status": "completed", "answer": answer, "sources": [],
                "finished_at": finished, "duration_ms": int((finished - started).total_seconds() * 1000)
            })
            return

        bot = Assistant(llm_model=LLM_MODEL, top_k=top_k, source_collection=collection if use_source_filter else None)
        cutoff_date = _get_cutoff_date(time_window_days or 0)
        result = bot.ask(question, cutoff_date=cutoff_date)
        bot.close()

        finished = datetime.now(timezone.utc)
        _save_job(job_id, {
            "job_id": job_id, "status": "completed",
            "answer": result.get("answer", ""), "sources": result.get("sources", []),
            "finished_at": finished, "duration_ms": int((finished - started).total_seconds() * 1000)
        })
        logger.info("Job %s completed in %ss", job_id, (finished - started).total_seconds())
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _save_job(job_id, {
            "job_id": job_id, "status": "failed", "error": str(exc), "finished_at": datetime.now(timezone.utc)
        })

@app.post("/api/rag/query/async")
def query_async(req: QueryRequest):
    """Enqueue a question for background processing. Returns immediately with a job_id.

    If no embeddings exist for the requested collection, the background worker
    will auto-ingest before answering — no manual ingestion required.
    """
    collection = req.collection or os.getenv("COLLECTION_NAME", "contents")

    # Verify the source collection actually exists in the database
    try:
        from source_store import SourceStore
        store = SourceStore()
        if store.count_documents(collection) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{collection}' does not exist in the database.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PostgreSQL error: {e}")

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
    from db import get_pool
    from fastapi import HTTPException
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM source_documents WHERE collection_name='rag_jobs' AND document_data->>'job_id' = %s", (job_id,))
            if cur.rowcount == 0:
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


def _ingest_runs_col(): return

def _dummy_func(raw_docs, top_n, hours, requested_subset):

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

    # Rank each category in parallel via the LLM
    ranked_by_cat: dict = {}
    cats_to_rank = [(c, items) for c, items in buckets.items() if items]
    with ThreadPoolExecutor(max_workers=min(4, len(cats_to_rank) or 1)) as ex:
        futures = {
            ex.submit(_rank_category_via_llm, cat, items, top_n, hours): cat
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
    pass

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
    ctx = []

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

    pass

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
        llm_summary = llm_generate(
            prompt,
            temperature=0.15,
            max_tokens=1800,
            timeout=600,
        )
    except Exception as exc:
        logger.warning("DIR LLM generation failed: %s", exc)
        llm_summary = "_(AI narrative unavailable — vLLM not reachable.)_"

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

    pass

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
        ctx = []
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
        ctx = []
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
        ctx = []
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
    """Rebuild the local vector search cache from PostgreSQL."""
    try:
        store = VectorStore()
        store.refresh_cache()
        store.close()
        return {"message": "Cache refreshed. (PostgreSQL queries dynamically)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
