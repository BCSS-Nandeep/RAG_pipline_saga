"""
llm_client.py — text generation against the OpenAI-compatible LLM endpoint.

Chat runs on our self-hosted vLLM server via /v1/chat/completions. Embeddings
run on a separate self-hosted vLLM embedding server (see embedder.py). No
external API and no Ollama is used anywhere in this project.

Configured entirely from .env, which previously defined these but was never
read by any code:

    LLM_BASE_URL          e.g. http://host/v1
    LLM_API_KEY           bearer token
    LLM_MODEL             e.g. qwen3-14b
    LLM_MAX_CONTEXT       prompt + completion budget (vLLM max_model_len)
    LLM_TIMEOUT           seconds
    LLM_ENABLE_THINKING   Qwen3 emits <think> blocks when true
"""

import logging
import os
import re
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-14b")
LLM_MAX_CONTEXT = int(os.getenv("LLM_MAX_CONTEXT", "16384"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "420"))
LLM_ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "false").lower() in (
    "1", "true", "yes", "on",
)

# Leave headroom for the chat template's own tokens.
_CTX_MARGIN = 512
# cl100k_base approximates Qwen tokenisation well on this corpus (measured
# ratio 1.022), but leave a margin rather than billing the full context.
_CTX_SAFETY = 0.88
# The shared vLLM endpoint answers 500 ("EngineCore encountered an issue")
# when a request exceeds the capacity it currently has free. That capacity
# moves with other tenants' load, not with our content: an identical prompt
# succeeded at 14k total tokens at one point and failed at 8k an hour later.
# So the reliable recovery is to SHRINK, not to wait — back off only briefly.
_ENGINE_BACKOFF = (3.0, 5.0, 8.0, 12.0, 15.0)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_HEADERS = {"Content-Type": "application/json"}
if LLM_API_KEY:
    _HEADERS["Authorization"] = f"Bearer {LLM_API_KEY}"


# ---------------------------------------------------------------------------
# prompt sizing
# ---------------------------------------------------------------------------

def _encoder():
    """cl100k_base is close enough to Qwen's tokenizer for budgeting."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def prompt_budget(max_tokens: int = 2048, scale: float = 1.0) -> int:
    """Tokens available for the whole prompt given a target completion size.

    Callers building a prompt from retrieved context use this to size the
    context *before* assembling it, rather than having fit_prompt cut it
    afterwards — a mid-record truncation loses the URL that makes the record
    actionable.
    """
    budget = int((LLM_MAX_CONTEXT - max_tokens - _CTX_MARGIN) * _CTX_SAFETY * scale)
    return max(512, budget)


def fit_prompt(prompt: str, max_tokens: int, scale: float = 1.0) -> str:
    """Trim the middle of *prompt* so prompt + completion fits the context.

    vLLM's max_model_len covers prompt AND completion, so an oversized prompt
    is rejected outright rather than silently truncated. ``scale`` shrinks the
    budget further on retries — an oversized prompt can take the engine down
    with a 500 rather than a clean 400, so retries back off on prompt size too.

    Budgeting uses cl100k_base, which only approximates Qwen's tokenizer (it
    diverges most on the Telugu content in this corpus), hence the safety
    factor rather than using the full nominal context.
    """
    budget = int((LLM_MAX_CONTEXT - max_tokens - _CTX_MARGIN) * _CTX_SAFETY * scale)
    if budget <= 0:
        budget = max(512, LLM_MAX_CONTEXT // 4)

    enc = _encoder()
    if enc is None:                                   # conservative char fallback
        approx = budget * 3
        if len(prompt) <= approx:
            return prompt
        head, tail = prompt[: approx // 2], prompt[-approx // 2:]
        return head + "\n\n[…context truncated to fit model context…]\n\n" + tail

    tokens = enc.encode(prompt)
    if len(tokens) <= budget:
        return prompt
    half = budget // 2
    head = enc.decode(tokens[:half])
    tail = enc.decode(tokens[-(budget - half):])
    logger.info("Prompt trimmed %d -> %d tokens to fit context.",
                len(tokens), budget)
    return head + "\n\n[…context truncated to fit model context…]\n\n" + tail


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.1,
    timeout: Optional[int] = None,
    model: Optional[str] = None,
    meta: Optional[dict] = None,
) -> str:
    """Generate text for *prompt*. Returns the model's reply.

    On terminal failure returns a clearly-marked ``_(...)_`` string rather than
    raising, so callers can still surface retrieved evidence to the user —
    this mirrors the behaviour the previous helpers had.

    Pass a dict as ``meta`` to receive call diagnostics (status, attempts,
    model, prompt/completion tokens) for logging.
    """
    if meta is None:
        meta = {}
    meta.update(model=model or LLM_MODEL, status=None, attempts=0,
                prompt_tokens=None, completion_tokens=None)
    model = model or LLM_MODEL
    timeout = timeout or LLM_TIMEOUT
    url = f"{LLM_BASE_URL}/chat/completions"

    # Each retry asks for a shorter completion AND a smaller prompt: an
    # oversized request takes the engine down with a 500 rather than a 400.
    # Each retry asks for materially less, so a request that overshot the
    # server's currently-free capacity fits on the next pass. Quality degrades
    # gracefully instead of the query failing outright.
    attempts = [
        (max_tokens, 1.0),
        (max_tokens, 0.70),
        (max(768, max_tokens // 2), 0.50),
        (max(512, max_tokens // 2), 0.35),
        (512, 0.25),
    ]
    last_err: Optional[str] = None

    for i, (want, scale) in enumerate(attempts):
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": fit_prompt(prompt, want, scale)})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": want,
            "repetition_penalty": repetition_penalty,
            # Qwen3 reasoning toggle, honoured by vLLM's chat template.
            "chat_template_kwargs": {"enable_thinking": LLM_ENABLE_THINKING},
        }

        try:
            meta["attempts"] = i + 1
            resp = requests.post(url, headers=_HEADERS, json=payload, timeout=timeout)
            meta["status"] = resp.status_code

            if resp.status_code in (500, 502, 503, 504):
                body = (resp.text or "")[:200]
                logger.warning("LLM %s on attempt %d: %s", resp.status_code, i + 1, body)
                last_err = f"{resp.status_code} — {body or resp.reason}"
                # EngineCore crashes (usually an oversized request) take ~45s
                # to self-heal; a 1.5s backoff just burns the retry budget.
                time.sleep(_ENGINE_BACKOFF[min(i, len(_ENGINE_BACKOFF) - 1)]
                           if resp.status_code == 500 else 2.0 * (i + 1))
                continue

            if resp.status_code == 400 and "context" in (resp.text or "").lower():
                logger.warning("Context overflow on attempt %d — shrinking.", i + 1)
                last_err = (resp.text or "")[:200]
                continue

            resp.raise_for_status()
            _body = resp.json()
            _usage = _body.get("usage") or {}
            meta["prompt_tokens"] = _usage.get("prompt_tokens")
            meta["completion_tokens"] = _usage.get("completion_tokens")
            choices = _body.get("choices") or []
            if not choices:
                last_err = "no choices in response"
                continue

            text = (choices[0].get("message", {}).get("content") or "").strip()
            if not LLM_ENABLE_THINKING:
                text = _THINK_RE.sub("", text).strip()
            if text:
                return text

            logger.warning("LLM returned empty content on attempt %d.", i + 1)
            last_err = "empty response"

        except requests.ConnectionError as exc:
            meta["status"] = "connection_error"
            logger.error("Cannot reach LLM at %s: %s", LLM_BASE_URL, exc)
            return (f"_(Error generating answer: LLM unreachable at "
                    f"{LLM_BASE_URL} — {exc}.)_")
        except requests.Timeout as exc:
            meta["status"] = "timeout"
            logger.warning("LLM timeout on attempt %d: %s", i + 1, exc)
            last_err = f"timeout ({exc})"
        except Exception as exc:
            logger.warning("LLM attempt %d failed: %s", i + 1, exc)
            last_err = str(exc)
            time.sleep(1.0 * (i + 1))

    return (f"_(LLM generation failed after {len(attempts)} attempts: "
            f"{last_err}. Live evidence from MongoDB is shown below — "
            "please retry.)_")


def check_health() -> bool:
    """True if the endpoint is reachable and LLM_MODEL is loaded."""
    try:
        resp = requests.get(f"{LLM_BASE_URL}/models", headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", [])]
        if LLM_MODEL in ids:
            logger.info("LLM healthy — model '%s' available at %s.",
                        LLM_MODEL, LLM_BASE_URL)
            return True
        logger.error("LLM reachable but model '%s' not loaded. Available: %s",
                     LLM_MODEL, ids)
        return False
    except Exception as exc:
        logger.error("LLM health-check failed at %s: %s", LLM_BASE_URL, exc)
        return False
