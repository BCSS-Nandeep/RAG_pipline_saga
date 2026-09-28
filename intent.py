"""
intent.py — request routing, capability boundaries and context budgeting.

The assistant used to run the same pipeline for every question: always pull
alerts + grievances from Mongo, always append vector hits, always demand a
long briefing from the LLM. That wasted retrieval on "hi", and — worse — the
prompt told the model to pad thin context with general knowledge, which is a
licence to hallucinate.

This module decides three things before any retrieval happens:

  1. what the user actually wants          -> classify()
  2. whether the backend can do it         -> UNSUPPORTED_ACTIONS
  3. how much context the prompt can hold  -> select_context()

Classification is rule-based on purpose: it adds no latency, costs no tokens,
and does not depend on the (intermittently unavailable) LLM endpoint.

Nothing here reads or writes MongoDB.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Context budget for retrieved evidence only — system prompt, question and
# output instructions are budgeted separately by the caller.
MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "6500"))

# Soft targets. These are preferences, not quotas: a simple question that is
# well answered by two chunks stops at two, a genuinely broad one may take more.
SOFT_MIN_CONTEXTS = 2
SOFT_MAX_CONTEXTS = 4
HARD_MAX_CONTEXTS = 10

# Once we have SOFT_MIN, keep taking chunks only while they remain close in
# relevance to the best one — this is what makes the count adaptive.
RELATIVE_SCORE_FLOOR = 0.80


class Intent(str, Enum):
    GREETING = "greeting"                     # hi / thanks / bye — no retrieval
    CAPABILITY = "capability"                 # "what can you do" — no retrieval
    GENERAL = "general"                       # general knowledge — no retrieval
    DATA_QUERY = "data_query"                 # normal RAG
    DATA_PLUS_UNSUPPORTED = "data_plus_unsupported"   # RAG + explain the limit
    UNSUPPORTED = "unsupported"               # explain the limit, no retrieval


# --------------------------------------------------------------------------- #
# capability boundary
# --------------------------------------------------------------------------- #
# Each entry: (key, pattern, human description of what is NOT possible).
# The assistant may never claim to have done any of these — there is no backend
# implementation behind them.

UNSUPPORTED_ACTIONS: List[Tuple[str, str, str]] = [
    ("pdf",
     r"\b(pdf|word\s+document|docx|powerpoint|pptx)\b",
     "generate or export PDF/Office documents"),
    ("file_export",
     r"\b(export|download|csv|excel|xlsx|spreadsheet)\b"
     r"|\bsave\s+(it|this|them|these)\b"
     r"|\bwrite\s+(it|this|them)?\s*to\s+a?\s*file\b",
     "export, download or write files"),
    ("email",
     r"\be-?mail\b|\bmail\s+(it|this|them|these)\b",
     "send email"),
    ("notify",
     r"\b(notification|push\s+notify|sms|text\s+message)\b"
     r"|\bnotify\s+\w+"
     r"|\balert\s+(him|her|them|the\s+(commissioner|cp|dcp|officer|team))\b",
     "send notifications, SMS or push alerts"),
    ("phone_call",
     r"\b(make|place|initiate)\s+(a\s+)?(phone\s+)?call\b"
     r"|\bcall\s+(him|her|them|the\s+(commissioner|cp|dcp|officer))\b",
     "place phone calls"),
    ("send_to_person",
     # "send <anything> to <someone>" — the object may sit between verb and
     # "to" ("send these alerts to the commissioner"), so allow a short gap.
     r"\b(send|forward|share|deliver|dispatch)\b[^.?!]{0,60}?\bto\b"
     r"|\b(forward|send)\s+(it|this|these|them)\b"
     r"|\bshare\s+(it|this|these|them)\s+with\b",
     "send or forward data to people or external systems"),
    ("mutate",
     r"\b(update|insert|delete|remove|modify|edit)\b[^.?!]{0,40}"
     r"\b(record|records|database|collection|entry|document|ticket|case|status)\b"
     r"|\bfile\s+an?\s+fir\b|\bregister\s+a\s+case\b",
     "create, update or delete records in SOC-EYE or any external system"),
    ("schedule",
     r"\b(schedule\s+(a|an|this|it)|set\s+a\s+reminder|remind\s+me)\b",
     "schedule tasks, jobs or reminders"),
]

# Vocabulary that means "this question is about SOC-EYE's own data".
_DATA_TERMS = r"""
    alert|alerts|grievance|grievances|complaint|complaints|content|contents|post|posts|
    tweet|tweets|reel|reels|story|stories|event|events|incident|incidents|
    dial\s*100|poi|pois|person\s+of\s+interest|persons\s+of\s+interest|suspect|suspects|
    profile|profiles|handle|handles|account|accounts|keyword|keywords|
    telegram|instagram|facebook|twitter|youtube|
    report|reports|criticism|suggestion|programme|programmes|
    audit|auditlog|user|users|source|sources|record|records|
    risk|threat|sentiment|engagement|viral|hashtag|
    protest|protests|rally|rallies|festival|communal|misinformation|defamation
"""
_DATA_RE = re.compile(rf"\b({_DATA_TERMS})\b", re.I | re.X)

_GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|howdy|greetings"
    r"|good\s*(morning|afternoon|evening|day|night)"
    r"|how\s+(are|r)\s+(you|u)\b.*"
    r"|what'?s\s+up\b.*"
    r"|thanks?|thank\s+you|thx|ty"
    r"|bye|goodbye|see\s+you|good\s*night"
    r"|ok(ay)?|cool|nice|great)"
    r"[\s!.,?~]*$",
    re.I,
)

_CAPABILITY_RE = re.compile(
    r"\b(what\s+can\s+you\s+do"
    r"|what\s+(are\s+)?your\s+(capabilities|features|abilities|functions)"
    r"|who\s+are\s+you|what\s+are\s+you"
    r"|how\s+(do|can)\s+i\s+use\s+(you|this)"
    r"|what\s+(data|collections?|modules?)\s+(do\s+you\s+have|can\s+you\s+access)"
    r"|help\s*(me)?$)\b",
    re.I,
)

# Clearly general-knowledge / non-SOC-EYE phrasing. Deliberately narrow: when
# in doubt the question is treated as a data query, because that is what this
# assistant is for.
_GENERAL_RE = re.compile(
    r"^\s*(what\s+is\s+(a|an|the)?\s*\b(?!alert|grievance|risk|threat)"
    r"|define\s+|explain\s+(what\s+)?(is\s+)?(a|an|the)?\s*\b(?!alert|grievance)"
    r"|who\s+(invented|wrote|discovered|founded)"
    r"|translate\s+|write\s+(a|an)\s+(poem|song|story|essay|code|script|program)"
    r"|how\s+do\s+i\s+(code|program|write\s+code|install|configure)"
    r"|capital\s+of\s+|meaning\s+of\s+life)",
    re.I,
)

# Enumeration-style questions ("list...", "show me...", "which...") want
# breadth — several short records — where a thematic question wants a few rich
# chunks. Both stay under the same token budget.
_LIST_RE = re.compile(
    r"\b(list|show\s+me|give\s+me|display|enumerate|which\s+\w+|"
    r"what\s+are\s+the|recent|latest|top\s+\d+|last\s+\d+)\b",
    re.I,
)

_COMPLEXITY_RE = re.compile(
    r"\b(all|every|across|compare|comparison|trend|trends|summar\w+|overview|"
    r"breakdown|analyse|analyze|analysis|correlat\w+|pattern|patterns|"
    r"and\s+also|as\s+well\s+as|relationship|between)\b",
    re.I,
)


@dataclass
class IntentResult:
    intent: Intent
    needs_rag: bool
    unsupported: List[Tuple[str, str]] = field(default_factory=list)  # (key, description)
    complex_question: bool = False
    reason: str = ""
    # Soft ceiling on context chunks for this question — a preference the
    # relevance floor and token budget can still cut short.
    suggested_contexts: int = SOFT_MAX_CONTEXTS

    @property
    def unsupported_descriptions(self) -> List[str]:
        return [d for _, d in self.unsupported]


def detect_unsupported(question: str) -> List[Tuple[str, str]]:
    """Return the unsupported actions this question asks for, if any."""
    found: List[Tuple[str, str]] = []
    for key, pattern, description in UNSUPPORTED_ACTIONS:
        if re.search(pattern, question, re.I | re.X):
            found.append((key, description))
    return found


def mentions_data(question: str) -> bool:
    return bool(_DATA_RE.search(question))


def classify(question: str) -> IntentResult:
    """Route a question. Rule-based, deterministic, no network calls."""
    q = (question or "").strip()
    if not q:
        return IntentResult(Intent.GREETING, False, reason="empty question")

    # Greetings are short by nature; a long message that merely opens with
    # "hi" is a real request.
    if len(q) <= 64 and _GREETING_RE.match(q):
        return IntentResult(Intent.GREETING, False, reason="greeting pattern")

    if _CAPABILITY_RE.search(q):
        return IntentResult(Intent.CAPABILITY, False, reason="capability question")

    unsupported = detect_unsupported(q)
    complex_q = bool(_COMPLEXITY_RE.search(q))
    wants_list = bool(_LIST_RE.search(q))
    suggested = SOFT_MAX_CONTEXTS
    if wants_list:
        suggested = 8
    if complex_q:
        suggested = HARD_MAX_CONTEXTS

    if unsupported:
        if mentions_data(q):
            # e.g. "create a PDF of the alerts from the last 24 hours" — do the
            # retrievable half, refuse the rest explicitly.
            return IntentResult(
                Intent.DATA_PLUS_UNSUPPORTED, True, unsupported, complex_q,
                reason=f"unsupported action(s) {[k for k, _ in unsupported]} + data reference",
                suggested_contexts=suggested,
            )
        return IntentResult(
            Intent.UNSUPPORTED, False, unsupported, complex_q,
            reason=f"unsupported action(s) {[k for k, _ in unsupported]}, no data reference",
        )

    if _GENERAL_RE.match(q) and not mentions_data(q):
        return IntentResult(Intent.GENERAL, False, complex_question=complex_q,
                            reason="general-knowledge phrasing, no data terms")

    # Default: this is a SOC-EYE assistant, so assume the question is about the
    # data unless something above proved otherwise.
    return IntentResult(Intent.DATA_QUERY, True, complex_question=complex_q,
                        reason="default — treated as data query",
                        suggested_contexts=suggested)


# --------------------------------------------------------------------------- #
# token counting + context selection
# --------------------------------------------------------------------------- #

from datetime import datetime, timezone, timedelta

def extract_dates(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (date_from, date_to) as YYYY-MM-DD strings based on deterministic parsing.
    Returns (None, None) if no explicit date is found.
    """
    import re
    try:
        from dateutil import parser
    except ImportError:
        parser = None

    q = (question or "").lower()
    now = datetime.now(timezone.utc)
    today = now.date()
    yesterday = today - timedelta(days=1)
    
    # Check explicit format: "from September 10 to September 15"
    if parser:
        month_re = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
        m_range = re.search(r'from\s+(' + month_re + r'\s+\d{1,2}(?:st|nd|rd|th)?)\s+to\s+(' + month_re + r'\s+\d{1,2}(?:st|nd|rd|th)?)', q)
        if m_range:
            try:
                d1 = parser.parse(m_range.group(1)).date()
                d2 = parser.parse(m_range.group(2)).date()
                if d1.month > today.month and d1.year == today.year: d1 = d1.replace(year=today.year - 1)
                if d2.month > today.month and d2.year == today.year: d2 = d2.replace(year=today.year - 1)
                return (d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d"))
            except Exception:
                pass

        # Check explicit date: "on september 10" or "september 10th"
        m_single = re.search(r'(?:on\s+)?(' + month_re + r'\s+\d{1,2}(?:st|nd|rd|th)?)', q)
        if m_single:
            try:
                d = parser.parse(m_single.group(1)).date()
                if d.month > today.month and d.year == today.year: d = d.replace(year=today.year - 1)
                return (d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
            except Exception:
                pass

    # YYYY-MM-DD
    m_iso = re.search(r'(\d{4}-\d{2}-\d{2})(?:\s+to\s+(\d{4}-\d{2}-\d{2}))?', q)
    if m_iso:
        if m_iso.group(2):
            return (m_iso.group(1), m_iso.group(2))
        return (m_iso.group(1), m_iso.group(1))

    if "today" in q:
        return (today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        
    if "yesterday" in q:
        return (yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"))

    return None, None

_ENC = None



def _encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:                      # pragma: no cover
            _ENC = False
    return _ENC or None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)          # rough fallback
    return len(enc.encode(text))


@dataclass
class ContextSelection:
    items: List[Dict[str, Any]]
    tokens: int
    considered: int
    dropped_by_budget: int = 0
    dropped_by_relevance: int = 0


def select_context(
    candidates: Sequence[Dict[str, Any]],
    *,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    text_key: str = "text",
    score_key: str = "score",
    complex_question: bool = False,
    soft_max: Optional[int] = None,
) -> ContextSelection:
    """Pick the most useful subset of *candidates* that fits the token budget.

    Whole chunks only — a half-cut record loses the URL or the identifier that
    makes it actionable, so a chunk that does not fit is skipped rather than
    truncated.

    The count is driven by relevance and budget, not by a fixed number: after
    the soft minimum, chunks are taken only while they stay within
    RELATIVE_SCORE_FLOOR of the best score, so a question answered well by two
    chunks gets two, and a broad one can take more.
    """
    if not candidates:
        return ContextSelection([], 0, 0)

    soft_cap = soft_max or (HARD_MAX_CONTEXTS if complex_question else SOFT_MAX_CONTEXTS)
    soft_cap = min(soft_cap, HARD_MAX_CONTEXTS)

    ranked = sorted(candidates, key=lambda c: c.get(score_key) or 0.0, reverse=True)
    top_score = ranked[0].get(score_key) or 0.0
    floor = top_score * RELATIVE_SCORE_FLOOR if top_score > 0 else 0.0

    chosen: List[Dict[str, Any]] = []
    used = 0
    by_budget = by_relevance = 0

    for cand in ranked:
        text = cand.get(text_key) or ""
        if not text.strip():
            continue
        n = len(chosen)

        if n >= HARD_MAX_CONTEXTS:
            break
        # Past the soft minimum, stop pulling in weaker material.
        if n >= SOFT_MIN_CONTEXTS and (cand.get(score_key) or 0.0) < floor:
            by_relevance += 1
            continue
        # Past the soft cap, only a genuinely strong chunk earns a slot.
        if n >= soft_cap and (cand.get(score_key) or 0.0) < top_score * 0.95:
            by_relevance += 1
            continue

        cost = count_tokens(text)
        if used + cost > max_tokens:
            by_budget += 1
            continue                            # try the next, smaller chunk

        chosen.append(cand)
        used += cost

    return ContextSelection(chosen, used, len(candidates), by_budget, by_relevance)


# --------------------------------------------------------------------------- #
# capability-aware response text
# --------------------------------------------------------------------------- #

CAPABILITY_ANSWER = (
    "I'm the **SOC-EYE Assistant**. I answer questions using the SOC-EYE "
    "database and its vector index.\n\n"
    "**What I can do**\n"
    "- Look up and summarise **alerts**, **grievances**, **contents/posts**, "
    "**events**, **Dial 100 incidents**, **persons of interest**, **keywords**, "
    "**monitored profiles**, **daily programmes**, **Telegram messages**, and "
    "**criticism/suggestion reports**\n"
    "- Count records over a time window (e.g. \"how many grievances this month\")\n"
    "- Find semantically related records and cite the source post or profile URL\n"
    "- Summarise themes, risks and trends across the retrieved records\n\n"
    "**What I cannot do**\n"
    "- Generate or export files (PDF, CSV, Excel, Word)\n"
    "- Send email, SMS, notifications, or place calls\n"
    "- Forward data to people or external systems\n"
    "- Create, update or delete any record\n"
    "- Schedule jobs or reminders\n\n"
    "Ask me something like *\"how many high-risk alerts were reported?\"* or "
    "*\"what are the recent grievances about?\"*"
)


def greeting_answer(question: str) -> str:
    """Short, varied-by-time-of-day greeting. No database context attached."""
    q = (question or "").strip().lower()
    if re.match(r"^\s*(thanks?|thank\s+you|thx|ty)\b", q):
        return "You're welcome! Anything else you'd like to look up in SOC-EYE?"
    if re.match(r"^\s*(bye|goodbye|see\s+you|good\s*night)\b", q):
        return "Goodbye! 👋"
    if re.match(r"^\s*(ok(ay)?|cool|nice|great)\b", q):
        return "👍 Let me know what you'd like to look up."

    m = re.match(r"^\s*good\s*(morning|afternoon|evening|day)", q)
    opener = f"Good {m.group(1)}!" if m else "Hello!"
    if re.search(r"how\s+(are|r)\s+(you|u)", q):
        opener = "I'm doing well, thanks!"
    return (f"{opener} 👋 I'm the SOC-EYE Assistant. "
            "How can I help you with the available SOC-EYE data?")


def limitation_notice(unsupported: Sequence[Tuple[str, str]], *,
                      data_follows: bool) -> str:
    """Deterministic statement of what the backend cannot do.

    Written in code rather than left to the LLM so the boundary is guaranteed:
    the model cannot talk its way into claiming it emailed or exported anything.
    """
    descriptions = [d for _, d in unsupported]
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        what = descriptions[0]
    else:
        what = ", ".join(descriptions[:-1]) + f", or {descriptions[-1]}"

    if data_follows:
        return (f"> ⚠️ **Note:** I can retrieve and present this data, but the "
                f"current SOC-EYE RAG system cannot {what}. "
                f"The requested information is below — nothing was sent, "
                f"exported or created.")
    return (f"I can retrieve and analyse the relevant SOC-EYE data, but the "
            f"current SOC-EYE RAG system cannot {what}. "
            f"Nothing was sent, exported or created.\n\n"
            f"If you tell me what you're looking for, I can pull the underlying "
            f"records and summarise them here instead.")


# Appended to the LLM prompt whenever an unsupported action was requested.
def capability_guardrail(unsupported: Sequence[Tuple[str, str]]) -> str:
    descriptions = [d for _, d in unsupported]
    return (
        "\nCAPABILITY BOUNDARY — this is mandatory:\n"
        f"• The user asked you to: {'; '.join(descriptions)}.\n"
        "• You CANNOT do any of that. There is no such integration.\n"
        "• Never say you have created, exported, sent, emailed, scheduled or "
        "delivered anything. Never promise to do it later.\n"
        "• Answer the part you CAN answer — presenting the retrieved records — "
        "and state plainly that the rest is not supported.\n"
    )
