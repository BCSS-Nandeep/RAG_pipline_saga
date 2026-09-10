"""
embedder.py — STAGE 4
  Generate 768-dim embeddings via Ollama REST API (nomic-embed-text).
  Uses the batch /api/embed endpoint for high throughput.
  Includes retry logic with exponential backoff and pre-flight health check.
"""

import logging
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EXPECTED_DIM = 768
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
BATCH_EMBED_SIZE = 50  # texts per batch call


class OllamaEmbedder:
    """Thin wrapper around Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, base_url: str, model: str = DEFAULT_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._embed_url = f"{self.base_url}/api/embed"       # batch endpoint
        self._embed_url_single = f"{self.base_url}/api/embeddings"  # single fallback

    # -- health check --------------------------------------------------------

    def check_health(self) -> bool:
        """Return True if Ollama is reachable and the embedding model is available."""
        try:
            # Check server
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Model names may include tag (e.g. "nomic-embed-text:latest")
            available = any(self.model in m for m in models)
            if not available:
                logger.error(
                    "Ollama is running but model '%s' not found. "
                    "Available: %s. Pull it with: ollama pull %s",
                    self.model,
                    models,
                    self.model,
                )
                return False
            logger.info("Ollama healthy — model '%s' available.", self.model)
            return True
        except requests.ConnectionError:
            logger.error(
                "Cannot reach Ollama at %s. Is it running?", self.base_url
            )
            return False
        except Exception as exc:
            logger.error("Ollama health-check failed: %s", exc)
            return False

    # -- embedding -----------------------------------------------------------

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text string. Returns a 768-dim vector or None."""
        results = self.embed_texts([text])
        return results[0] if results else None

    def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Embed multiple texts in one call using Ollama /api/embed.
        Returns a list of vectors (or None for failures), same order as input.
        """
        if not texts:
            return []

        all_results: List[Optional[List[float]]] = []

        # Process in sub-batches to avoid overloading
        for i in range(0, len(texts), BATCH_EMBED_SIZE):
            batch = texts[i : i + BATCH_EMBED_SIZE]
            embeddings = self._embed_batch_call(batch)
            all_results.extend(embeddings)

        return all_results

    def _embed_batch_call(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Call the /api/embed endpoint with a batch of texts."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._embed_url,
                    json={"model": self.model, "input": texts},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts):
                    return embeddings
                logger.warning(
                    "Batch returned %d embeddings for %d texts",
                    len(embeddings), len(texts),
                )
                # Pad with None for missing
                return embeddings + [None] * (len(texts) - len(embeddings))
            except (requests.ConnectionError, requests.Timeout) as exc:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama batch embed error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            except requests.HTTPError as exc:
                # If batch endpoint not supported, fall back to single
                if resp.status_code == 404:
                    logger.warning("Batch /api/embed not available, falling back to single calls.")
                    return [self._embed_single(t) for t in texts]
                logger.error("Ollama HTTP error: %s", exc)
                return [None] * len(texts)

        logger.error("All %d retries exhausted for batch embedding.", MAX_RETRIES)
        return [None] * len(texts)

    def _embed_single(self, text: str) -> Optional[List[float]]:
        """Fallback: embed one text via /api/embeddings."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._embed_url_single,
                    json={"model": self.model, "prompt": text},
                    timeout=120,
                )
                resp.raise_for_status()
                embedding = resp.json().get("embedding")
                if embedding:
                    return embedding
                return None
            except (requests.ConnectionError, requests.Timeout) as exc:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama connection error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            except requests.HTTPError as exc:
                logger.error("Ollama HTTP error: %s", exc)
                return None
        return None

    def embed_batch(self, texts: List[str], delay: float = 0) -> List[Optional[List[float]]]:
        """Alias for backward compatibility."""
        return self.embed_texts(texts)
