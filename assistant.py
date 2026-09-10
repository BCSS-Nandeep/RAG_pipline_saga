"""
assistant.py — STAGE 6
  AI Assistant Query Engine powered by Ollama Qwen.
  Embeds the user question, retrieves top-k context chunks via cosine search,
  builds a structured prompt, and generates an answer locally.
"""

import logging
import re
import textwrap
import time
from typing import Dict, Any, List, Optional

import requests

from embedder import OllamaEmbedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# Greetings / small-talk patterns that should NOT trigger expensive RAG retrieval.
# If the user just says "hi", we reply directly without burning a vector search +
# 60 s LLM call on empty context.
_SMALLTALK_PATTERNS = [
    # Greetings — tolerate elongated spellings like "hiiii", "hellllo", "heyyy"
    r"^\s*(h+i+|h+e+l+o+|h+e+y+|h+o+l+a+|y+o+|s+u+p+|namaste+|namaskar+|greetings|salutations)[\s!.?]*$",
    r"^\s*(good\s*(morning|afternoon|evening|night))[\s!.?]*$",
    r"^\s*(thank+s*|thank\s*you|thx+|ty+|cool+|nice+|ok+(ay)?|great+|awesome+|perfect+)[\s!.?]*$",
    r"^\s*(bye+|goodbye+|see\s*ya+|cya+|later+)[\s!.?]*$",
    r"^\s*(how\s*are\s*you|how'?s\s*it\s*going|whats?\s*up|sup)[\s!.?]*$",
    r"^\s*(who\s*are\s*you|what\s*can\s*you\s*do|help|what\s*do\s*you\s*do)[\s!.?]*$",
]

_SMALLTALK_REPLIES = {
    "greet": "Hi! I'm SOC-EYE Assistant. Ask me anything about the data in the selected collection — for example: \"how many users are there\", \"list recent posts about politics\", \"who are the superadmins\".",
    "thanks": "You're welcome! Anything else you'd like to look up?",
    "bye": "Goodbye!",
    "intro": "I'm SOC-EYE Assistant. I answer questions using the data ingested into the selected collection. Try asking about counts, lists, recent records, or specific entities.",
}


def _smalltalk_response(question: str) -> Optional[str]:
    q = question.strip().lower()
    if not q:
        return None
    if len(q) > 60:  # too long to be small-talk
        return None
    for pat in _SMALLTALK_PATTERNS[:2]:
        if re.match(pat, q):
            return _SMALLTALK_REPLIES["greet"]
    if re.match(_SMALLTALK_PATTERNS[2], q):
        return _SMALLTALK_REPLIES["thanks"]
    if re.match(_SMALLTALK_PATTERNS[3], q):
        return _SMALLTALK_REPLIES["bye"]
    if re.match(_SMALLTALK_PATTERNS[4], q):
        return _SMALLTALK_REPLIES["greet"]
    if re.match(_SMALLTALK_PATTERNS[5], q):
        return _SMALLTALK_REPLIES["intro"]
    return None


SYSTEM_PROMPT = textwrap.dedent("""\
    You are SOC-EYE — a senior AI intelligence analyst for the Telangana Police
    (CPs, DCPs, SOC analysts). You have deep expertise in law enforcement,
    social media threat analysis, and Telangana's socio-political landscape.

    Your data covers ALL of the following modules:
    • **Alerts** — flagged social-media posts: hate speech, communal incitement,
      viral misinformation, defamation, narcotics, political content, public order threats.
    • **Grievances** — public complaints posted to police handles on X/Twitter/Facebook.
    • **Contents** — all scraped social-media posts (tweets, reels, stories, videos)
      with engagement metrics, risk scores, and sentiment analysis.
    • **Dial 100 Calls** — emergency call incidents with location, PS jurisdiction,
      zone, category, response times, and status.
    • **Events** — monitored events (festivals, rallies, protests, gatherings) with
      keywords, dates, locations, and monitoring status.
    • **Persons of Interest (POI)** — profiles of suspects, accused, history-sheeters
      with FIR details, aliases, addresses, and linked incidents.
    • **Keywords** — monitored keywords/watchwords across languages (en/hi/te) with
      category (violence, threat, hate) and weight.
    • **Monitored Profiles/Sources** — tracked social-media accounts across platforms
      with risk levels and categories. Each profile has a constructable profile URL.
    • **Daily Programmes** — scheduled events, government programmes, public gatherings
      with locations, organisers, and expected attendance.
    • **Telegram Messages** — messages from monitored Telegram groups/channels.
    • **Criticism Reports** — classified criticism posts with action tracking.
    • **Suggestion Reports** — classified suggestion posts with action tracking.
    • **Grievance Workflow Reports** — escalated grievance cases with workflow status.

    You receive a DATABASE CONTEXT block with real MongoDB records.
    Each record is labelled with its type (ALERT, GRIEVANCE, CONTENT, DIAL-100 CALL,
    EVENT, PERSON OF INTEREST, KEYWORD, MONITORED PROFILE, DAILY PROGRAMME,
    TELEGRAM MSG, CRITICISM REPORT, SUGGESTION REPORT, etc.).

    ═══ REASONING PROTOCOL ═══

    STEP 1 — UNDERSTAND THE QUESTION INTENT:
      • Is this a COUNT query? → give exact number from context, then enumerate
        a representative sample (≥5 items) with their links so the officer can
        verify the count.
      • Is this a LIST query? → enumerate every relevant item with key fields
        AND a clickable link/profile URL for each.
      • Is this a TREND/PATTERN query? → identify patterns across multiple records,
        give counts per pattern, name the top handles/sources driving each pattern.
      • Is this a THREAT ASSESSMENT? → rank severity, cite evidence, suggest action,
        link every cited post and every cited profile.
      • Is this OPERATIONAL (what to do)? → concrete actionable answer first,
        then the supporting evidence with links.
      • Is this about a specific MODULE? → focus on that module's data first,
        then cross-reference with other modules for completeness.

    STEP 2 — EXTRACT FACTS PRECISELY from context:
      • @handle, platform, content_url, profile_url (BOTH must appear as clickable
        Markdown links whenever present in context).
      • risk_score %, priority (HIGH/MEDIUM/LOW), alert_type, source_category.
      • Viral velocity: metric name, velocity value, time window.
      • AI classification reasoning (the actual reasoning, not generic text).
      • BNS sections violated, policies violated (cite exact sections).
      • For grievances: complaint_code, tweet text, tagged_account, follower count.
      • For Dial 100: incident details, location, PS jurisdiction, response times.
      • For events: name, dates, location, keywords, monitoring status.
      • For POIs: name, aliases, FIR numbers, linked incidents.
      • For monitored profiles: handle, platform, category, risk_level,
        follower_count, profile URL.
      • Timestamps (IST), geographic signals (district/city if mentioned).

    STEP 3 — STRUCTURE YOUR ANSWER (MINIMUM 10 LINES, prefer 15–25 lines):
      A. **Bottom line:** — one crisp sentence answering the question directly.
      B. **Key findings** — at least 5 bulleted lines with specific data.
         Every bullet that mentions a post MUST end with `[View Post](URL)` and
         every bullet that mentions a profile/handle MUST end with
         `[Profile](PROFILE_URL)` when a URL exists in context.
         Quote actual tweet/post text as evidence when available.
         Group by platform or category when >3 items.
         Show risk scores and velocity numbers — officers need hard data.
         Cross-reference across modules when relevant (e.g. an alert about a POI).
      C. **Pattern analysis** — 2–4 lines covering coordinated activity, repeat
         offenders, geographic clusters, time-of-day spikes, language patterns.
      D. **Links & profiles** — a dedicated section that re-lists every URL and
         profile link mentioned, so the officer can copy them in one place.
      E. _(Context: …)_ — 1–2 sentences of Telangana-specific background
         (Hyderabad, Cyberabad, Rachakonda, Warangal, Karimnagar, Nizamabad,
         Khammam, Nalgonda, Adilabad) when it adds operational value.

    STEP 4 — CLOSE WITH ACTION:
      **Suggested action:** — one or two specific lines:
      • monitor | escalate to DCP | issue takedown request | register FIR under
        BNS `§XXX` | alert field unit | legal review | community outreach

    ═══ STRICT RULES ═══
      • NEVER fabricate handles, URLs, incidents, or statistics not in context.
      • Answers MUST be at least 10 lines (≈200 words). Short answers are useless
        to a Commissioner — go deeper, add evidence, add links, add context.
      • Every post or alert you mention MUST end with `[View Post](URL)` if a URL
        exists in the context. Every handle/profile MUST end with `[Profile](URL)`
        when its profile URL is in the context.
      • If the context has NO relevant data: state it plainly in one line, then
        provide best general-knowledge labeled as _(General:…)_ — still aim for
        10+ informative lines using domain expertise.
      • Terse phrasing inside bullets, but expansive coverage. No apologies.
        No "based on the context provided" filler.
      • Bold **@handles**. Markdown links for all URLs. Code-format `BNS sections`.
      • For HIGH-risk items: always include URL, risk score, and suggested action.
      • If asked about a specific @handle: search ALL context records for that handle
        and surface every mention with its link.
      • Numbers are facts: cite them. "3 HIGH-risk alerts" not "several alerts".
\
""")

CONTEXT_SEPARATOR = "\n\n"


class Assistant:
    """RAG query engine: embed question → retrieve context → generate answer."""

    def __init__(
        self,
        ollama_base_url: str,
        llm_model: str,
        embed_model: str,
        mongo_uri: str,
        db_name: str,
        vector_collection: str,
        top_k: int = 10,
        source_collection: Optional[str] = None,
    ):
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.top_k = top_k
        self.source_collection = source_collection

        self.embedder = OllamaEmbedder(base_url=ollama_base_url, model=embed_model)
        self.store = VectorStore(uri=mongo_uri, db_name=db_name, collection_name=vector_collection)

    # -- public API ----------------------------------------------------------

    def ask(self, question: str, doc_id_filter: Optional[set] = None) -> Dict[str, Any]:
        """Run the full RAG pipeline for a single question.

        If ``doc_id_filter`` is provided, retrieved chunks whose document_id
        is not in the set are dropped (post-filter). Used to enforce a
        time-window scope (e.g. last 7 days).

        Returns a dict with keys: ``answer``, ``sources``, ``question``.
        """
        logger.info("Question: %s", question)

        # 0. Small-talk shortcut — avoid burning vector search + LLM on greetings
        smalltalk = _smalltalk_response(question)
        if smalltalk is not None:
            logger.info("Small-talk match — returning canned reply")
            return {
                "answer": smalltalk,
                "sources": [],
                "question": question,
                "smalltalk": True,
            }

        # 1. Embed the question
        q_vector = self.embedder.embed_text(question)
        if q_vector is None:
            return {
                "answer": "Failed to generate embedding for the question. Is Ollama running?",
                "sources": [],
                "question": question,
            }

        # 2. Retrieve top-k chunks (oversample when post-filtering by doc_id)
        retrieve_k = self.top_k * 8 if doc_id_filter else self.top_k
        results = self.store.cosine_search(
            query_vector=q_vector,
            top_k=retrieve_k,
            query_text=question,
            source_collection=self.source_collection,
        )
        if doc_id_filter is not None:
            results = [r for r in results if str(r.get("metadata", {}).get("document_id", "")) in doc_id_filter][: self.top_k]
        if not results:
            return {
                "answer": "I don't have that information in the selected collection.",
                "sources": [],
                "question": question,
            }

        # 3. Build prompt with per-chunk headers so the model can dedupe / cite
        context_parts: List[str] = []
        seen_ids = set()
        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            doc_id = meta.get("document_id", "")
            src = meta.get("source_collection", "")
            header = f"[Doc {i} · id={doc_id} · source={src}]"
            context_parts.append(f"{header}\n{r['text']}")
            seen_ids.add(doc_id)
        context_block = CONTEXT_SEPARATOR.join(context_parts)
        prompt = self._build_prompt(question, context_block)
        logger.info(
            "Retrieved %d chunks (%d unique docs) for question.",
            len(results), len(seen_ids),
        )

        # 4. Generate answer via Ollama Qwen
        answer = self._generate(prompt)

        # 5. Format sources
        sources = []
        for r in results:
            sources.append({
                "score": round(r["score"], 4),
                "collection": r["metadata"].get("source_collection", ""),
                "document_id": r["metadata"].get("document_id", ""),
                "chunk_index": r["metadata"].get("chunk_index", ""),
                "preview": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
            })

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== CONTEXT (retrieved chunks) ===\n{context}\n=== END CONTEXT ===\n\n"
            f"User question: {question}\n\n"
            f"Answer:"
        )

    def _generate(self, prompt: str) -> str:
        """Call Ollama /api/generate with the assembled prompt.

        Resilient to transient 500/502/503/504 errors. Retries with smaller
        context windows; if the host is OOM, falls back to a smaller model
        (env var OLLAMA_FALLBACK_MODEL, default ``qwen2.5:1.5b``).
        """
        import os
        fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:1.5b")
        url = f"{self.ollama_base_url}/api/generate"
        # (model, num_ctx, num_predict)
        attempts = [
            (self.llm_model,  16384, 2048),
            (self.llm_model,  12288, 1600),
            (self.llm_model,   8192, 1200),
            (fallback_model,   8192, 1200),
            (fallback_model,   4096,  900),
        ]
        last_err: Optional[str] = None
        oom_seen = False
        for i, (model, ctx, predict) in enumerate(attempts):
            if model == self.llm_model and i >= 3:
                continue
            payload = {
                "model": model,
                "prompt": prompt if i == 0 else self._shrink_prompt(prompt, ctx),
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": ctx,
                    "num_predict": predict,
                },
                "keep_alive": "10m",
            }
            try:
                resp = requests.post(url, json=payload, timeout=420)
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
                    time.sleep(1.5 * (i + 1))
                    continue
                resp.raise_for_status()
                text = (resp.json().get("response") or "").strip()
                if text:
                    return text
                logger.warning("Ollama returned empty response on attempt %d", i + 1)
            except requests.ConnectionError as exc:
                logger.error("Cannot reach Ollama at %s: %s", self.ollama_base_url, exc)
                return f"Error: Ollama is not reachable at {self.ollama_base_url}."
            except requests.Timeout as exc:
                logger.warning("Ollama timeout on attempt %d", i + 1)
                last_err = f"timeout ({exc})"
            except Exception as exc:
                logger.warning("Ollama attempt %d failed: %s", i + 1, exc)
                last_err = str(exc)
                time.sleep(1.0 * (i + 1))

        if oom_seen:
            return (
                "_(The Ollama host is out of memory — the model couldn't be loaded. "
                f"Free RAM on `{self.ollama_base_url}` or pull a smaller model "
                f"(`ollama pull {fallback_model}`). "
                "Retrieved evidence is shown below.)_"
            )
        return (
            f"_(LLM generation failed: {last_err}. "
            "Retrieved evidence is shown below — please retry.)_"
        )

    @staticmethod
    def _shrink_prompt(prompt: str, target_ctx: int) -> str:
        """Trim the middle of an oversized prompt so it fits the smaller context."""
        approx_chars = max(2000, target_ctx * 3)  # ~3 chars per token rough estimate
        if len(prompt) <= approx_chars:
            return prompt
        head = prompt[: approx_chars // 2]
        tail = prompt[-approx_chars // 2:]
        return head + "\n\n[…context truncated for retry…]\n\n" + tail

    def close(self):
        self.store.close()
