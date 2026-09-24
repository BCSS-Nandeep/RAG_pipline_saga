"""
embedder.py — STAGE 4

Local embeddings with nomic-ai/nomic-embed-text-v1.5, run in-process on this
machine through PyTorch/Transformers. No network service is involved: no
Ollama, no OpenAI, no Gemini, no remote embedding server.

Nomic v1.5 is an instruction-prefixed model. The prefix is part of the input,
not decoration — a document embedded as a query lands in a different region of
the space, so retrieval quietly degrades if the two sides disagree:

    documents : "search_document: <text>"
    queries   : "search_query: <text>"

``embed_documents`` and ``embed_query`` apply these for you. The lower-level
``embed_texts``/``embed_text`` default to the query prefix, because that is
what the RAG path calls at question time.

Configuration (.env):

    EMBED_MODEL        nomic-ai/nomic-embed-text-v1.5
    EMBED_DIM          768
    EMBED_BATCH_SIZE   texts per forward pass (tune to VRAM)
    EMBED_DEVICE       cuda | cpu | auto   (auto falls back to CPU)
    EMBED_MAX_LENGTH   token truncation length
    EMBED_NORMALIZE    L2-normalise outputs (default true)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

from dotenv import load_dotenv

# Every EMBED_* constant below is read at import time. api_server.py imports
# this module before it calls load_dotenv(), so without this call the .env
# overrides are silently dropped and the serving process embeds with the code
# defaults (notably EMBED_MAX_LENGTH=512) while regenerate_embeddings.py --
# which does load .env first -- embeds at 1024. That mismatch writes vectors
# into the corpus that no query can reproduce, so load the file here too.
load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5").strip()
EXPECTED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Weights live in the HuggingFace cache. Set EMBED_CACHE_DIR to place it on a
# specific disk in deployment; unset uses the default ~/.cache/huggingface.
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "").strip() or None
if EMBED_CACHE_DIR:
    # HF_HUB_CACHE, not HF_HOME: HF_HOME is the parent that holds a `hub/`
    # subdirectory, while cache_dir IS the hub directory. Setting HF_HOME to a
    # cache_dir value makes the loader resolve a snapshot path that is never
    # created, and it fails with WinError 3 instead of downloading.
    os.environ.setdefault("HF_HUB_CACHE", EMBED_CACHE_DIR)

# Pin both repos. nomic-embed-text-v1.5 needs trust_remote_code, which means
# transformers executes modeling code fetched from nomic-ai/nomic-bert-2048 at
# load time. Unpinned, a push to that repo silently changes what runs on this
# machine — and the loader has already been observed reporting "A new version
# of the following files was downloaded". Pinning makes deploys reproducible.
EMBED_REVISION = os.getenv("EMBED_REVISION", "e9b6763023c676ca8431644204f50c2b100d9aab").strip() or None
EMBED_CODE_REVISION = os.getenv("EMBED_CODE_REVISION", "7710840340a098cfb869c4f65e87cf2b1b70caca").strip() or None
# 0 / unset means "decide from the hardware actually present".
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "0"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto").strip().lower()
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "true").lower() in ("1", "true", "yes", "on")

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

DEFAULT_EMBED_MODEL = EMBED_MODEL      # kept for older imports


class EmbeddingConfigError(RuntimeError):
    """Model could not be loaded, or the configuration is unusable."""


class EmbeddingDimensionError(RuntimeError):
    """A vector's width does not match the configured corpus dimension."""


# --------------------------------------------------------------------------- #
# model cache
# --------------------------------------------------------------------------- #

# Enough to load with: config, tokenizer and weights. A full snapshot check
# would demand files this project never fetches (e.g. pytorch_model.bin when
# safetensors is present) and report a cached model as missing.
_REQUIRED_PATTERNS = ("config.json", "tokenizer_config.json")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def model_cache_path(model_name: str = EMBED_MODEL) -> Optional[str]:
    """Local snapshot directory holding usable weights, or None."""
    import glob
    from pathlib import Path

    # Resolution order mirrors huggingface_hub: an explicit cache_dir wins,
    # then HF_HUB_CACHE (the hub dir), then HF_HOME/hub, then the default.
    if EMBED_CACHE_DIR:
        hub = Path(EMBED_CACHE_DIR)
    elif os.getenv("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"])
    elif os.getenv("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache" / "huggingface" / "hub"

    repo_dir = hub / ("models--" + model_name.replace("/", "--"))
    if not repo_dir.is_dir():
        return None

    snapshots = repo_dir / "snapshots"
    if EMBED_REVISION and (snapshots / EMBED_REVISION).is_dir():
        candidates = [snapshots / EMBED_REVISION]
    elif snapshots.is_dir():
        candidates = [d for d in snapshots.iterdir() if d.is_dir()]
    else:
        return None

    for snap in candidates:
        names = {p.name for p in snap.iterdir()} if snap.is_dir() else set()
        has_meta = all(r in names for r in _REQUIRED_PATTERNS)
        has_weights = any(n.endswith(_WEIGHT_SUFFIXES) for n in names)
        if has_meta and has_weights:
            return str(snap)
    return None


def check_cache_path(model_name: str = EMBED_MODEL) -> None:
    """Refuse a cache directory that Windows cannot hold the snapshot in.

    HuggingFace nests weights under
    ``<cache>/models--<org>--<name>/snapshots/<40-char sha>/<filename>``, which
    adds roughly 120 characters. Past the 260-character MAX_PATH limit the
    download fails with a bare "WinError 3: cannot find the path specified"
    that looks like a missing model rather than a path-length problem.
    """
    if os.name != "nt" or not EMBED_CACHE_DIR:
        return
    nested = len(os.path.abspath(EMBED_CACHE_DIR)) + len(
        f"models--{model_name.replace('/', '--')}") + 11 + 40 + 30
    if nested > 260:
        raise EmbeddingConfigError(
            f"EMBED_CACHE_DIR is too deep for Windows: the model snapshot would "
            f"need ~{nested} characters and MAX_PATH is 260.\n"
            f"  current: {EMBED_CACHE_DIR}\n"
            f"  use a short path such as C:\\hf-cache, or enable long paths via "
            f"LongPathsEnabled in the registry."
        )


def is_model_cached(model_name: str = EMBED_MODEL) -> bool:
    """True if usable weights are already on this machine.

    Informational only — from_pretrained downloads on a miss either way.
    Knowing which happened is what makes a slow first start explainable
    instead of looking like a hang.
    """
    return model_cache_path(model_name) is not None


def warmup(fatal: bool = False) -> dict:
    """Download (if needed) and load the model into GPU memory.

    Call this at service start so the cost lands during boot rather than on
    whichever user happens to ask the first question — on a fresh deploy that
    is a ~520 MB download plus load, which would otherwise look like the API
    hanging.
    """
    import time as _time
    started = _time.time()
    cached_before = is_model_cached()
    try:
        emb = get_embedder()
        emb.load()
        info = {
            "ok": True,
            "model": emb.model_name,
            "device": emb.device,
            "batch_size": emb.batch_size,
            "downloaded": not cached_before,
            "cache_path": model_cache_path(),
            "seconds": round(_time.time() - started, 1),
            "error": None,
        }
        logger.info("Embedding model ready: %s on %s in %.1fs (%s)",
                    info["model"], info["device"], info["seconds"],
                    "downloaded" if info["downloaded"] else "from cache")
        return info
    except Exception as exc:
        logger.error("Embedding model warm-up FAILED: %s", exc)
        if fatal:
            raise
        return {"ok": False, "model": EMBED_MODEL, "device": None,
                "batch_size": None, "downloaded": False, "cache_path": None,
                "seconds": round(_time.time() - started, 1), "error": str(exc)}


# --------------------------------------------------------------------------- #
# hardware detection
# --------------------------------------------------------------------------- #

def detect_hardware() -> dict:
    """Report what this machine actually has, and a batch size that fits it.

    Nothing here is assumed from a spec sheet: GPU name, VRAM, CUDA support and
    system RAM are all read from the running system, because a batch size tuned
    for the wrong card either wastes throughput or dies with CUDA OOM.
    """
    info = {
        "torch": None, "torch_cuda_build": None, "cuda_available": False,
        "gpu_name": None, "vram_gb": None, "device": "cpu",
        "system_ram_gb": None, "cpu_count": os.cpu_count(),
        "suggested_batch_size": 8,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
            info["device"] = "cuda"
    except Exception as exc:
        info["error"] = str(exc)

    try:
        import ctypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            info["system_ram_gb"] = round(st.ullTotalPhys / 1e9, 1)
    except Exception:
        pass

    # nomic-embed-text-v1.5 is ~137M params; in fp16 the weights are ~0.3 GB and
    # activations dominate. These sizes leave room for the driver and other apps.
    if info["cuda_available"] and info["vram_gb"]:
        v = info["vram_gb"]
        if v >= 16:
            info["suggested_batch_size"] = 128
        elif v >= 10:
            info["suggested_batch_size"] = 64
        elif v >= 7:
            info["suggested_batch_size"] = 48
        elif v >= 5:
            info["suggested_batch_size"] = 32
        else:                                   # 4 GB class, e.g. RTX 500 Ada
            info["suggested_batch_size"] = 16
    else:
        info["suggested_batch_size"] = 8        # CPU: keep latency sane

    return info


def describe_hardware() -> str:
    hw = detect_hardware()
    if hw["cuda_available"]:
        where = f"{hw['gpu_name']} ({hw['vram_gb']} GB VRAM)"
    else:
        where = f"CPU ({hw['cpu_count']} cores)"
    return (f"torch {hw['torch']} cuda={hw['torch_cuda_build']} -> {where}, "
            f"RAM {hw['system_ram_gb']} GB, batch {hw['suggested_batch_size']}")


# --------------------------------------------------------------------------- #
# local model
# --------------------------------------------------------------------------- #

class NomicLocalEmbedder:
    """In-process nomic-embed-text-v1.5.

    The model is loaded lazily and only once per process — loading costs a few
    seconds and ~550 MB, so the API server must not pay it per request.
    """

    _lock = threading.Lock()

    def __init__(self,
                 model_name: str = EMBED_MODEL,
                 device: Optional[str] = None,
                 batch_size: int = 0,
                 max_length: int = EMBED_MAX_LENGTH,
                 expected_dim: int = EXPECTED_DIM,
                 normalize: bool = EMBED_NORMALIZE):
        self.model = model_name
        self.model_name = model_name
        resolved = batch_size or EMBED_BATCH_SIZE
        if not resolved:
            resolved = detect_hardware()["suggested_batch_size"]
        self.batch_size = max(1, resolved)
        self.max_length = max_length
        self.expected_dim = expected_dim
        self.normalize = normalize
        self._device_pref = (device or EMBED_DEVICE).lower()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self.device = None
        # Mirrors the old network clients so logging/health code keeps working.
        self.base_url = "local://torch"

    def __repr__(self) -> str:                             # pragma: no cover
        return f"<NomicLocalEmbedder {self.model_name} device={self.device or self._device_pref}>"

    # -- loading -------------------------------------------------------------

    def _resolve_device(self, torch) -> str:
        if self._device_pref == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if self._device_pref == "cuda":
            logger.warning("EMBED_DEVICE=cuda but CUDA is unavailable "
                           "(torch %s, cuda build %s) — falling back to CPU.",
                           torch.__version__, torch.version.cuda)
        return "cpu"

    def load(self):
        """Load tokenizer + model once. Safe to call repeatedly."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:                      # pragma: no cover
                raise EmbeddingConfigError(
                    f"torch/transformers are required for local embeddings: {exc}"
                ) from exc

            self._torch = torch
            self.device = self._resolve_device(torch)
            logger.info("Loading %s on %s ...", self.model_name, self.device)

            check_cache_path()
            cached = is_model_cached(self.model_name)
            logger.info("Weights %s in the local cache — %s.",
                        "found" if cached else "NOT found",
                        "loading from disk" if cached
                        else "downloading now (one-time, ~520 MB)")

            kwargs = {"trust_remote_code": True}
            if EMBED_REVISION:
                kwargs["revision"] = EMBED_REVISION
            if EMBED_CODE_REVISION:
                kwargs["code_revision"] = EMBED_CODE_REVISION
            if EMBED_CACHE_DIR:
                kwargs["cache_dir"] = EMBED_CACHE_DIR

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, **kwargs)
                model = AutoModel.from_pretrained(
                    self.model_name,
                    # fp16 halves VRAM on GPU; CPU keeps fp32 for stability.
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    **kwargs,
                )
            except Exception as exc:
                raise EmbeddingConfigError(
                    f"Could not load {self.model_name!r}: {exc}\n"
                    f"  cache_dir={EMBED_CACHE_DIR or '<default>'} "
                    f"revision={EMBED_REVISION}\n"
                    f"  If this machine is offline, pre-populate the cache or "
                    f"unset HF_HUB_OFFLINE."
                ) from exc

            model.eval()
            model.to(self.device)
            self._model = model
            if self.device == "cuda":
                props = torch.cuda.get_device_properties(0)
                logger.info("Embedding model ready on %s (%.1f GB VRAM).",
                            props.name, props.total_memory / 1e9)
            else:
                logger.info("Embedding model ready on CPU.")
            return self._model

    # -- pooling -------------------------------------------------------------

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """Mean pooling over real tokens — what Nomic v1.5 expects."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    # -- embedding -----------------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        """Run the model over already-prefixed texts."""
        if not texts:
            return []
        self.load()
        torch = self._torch
        out: List[Optional[List[float]]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            try:
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)

                with torch.inference_mode():
                    hidden = self._model(**enc).last_hidden_state
                    vecs = self._mean_pool(hidden, enc["attention_mask"])
                    if self.normalize:
                        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                    vecs = vecs.float().cpu()

                if not torch.isfinite(vecs).all():
                    raise ValueError("model produced NaN/Inf values")

                for row in vecs:
                    vec = row.tolist()
                    if len(vec) != self.expected_dim:
                        raise EmbeddingDimensionError(
                            f"{self.model_name} produced dimension {len(vec)}, "
                            f"expected {self.expected_dim}."
                        )
                    out.append(vec)

            except EmbeddingDimensionError:
                raise
            except Exception as exc:
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and self.batch_size > 1:
                    # Halve and retry this batch rather than losing it.
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning("CUDA OOM — reducing EMBED_BATCH_SIZE to %d "
                                   "and retrying this batch.", self.batch_size)
                    if self._torch is not None and self.device == "cuda":
                        self._torch.cuda.empty_cache()
                    out.extend(self._encode(batch))
                    continue
                logger.error("Embedding batch failed: %s", exc)
                out.extend([None] * len(batch))

        return out

    def embed_documents(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        """Embed corpus text with Nomic's document prefix."""
        return self._encode([DOCUMENT_PREFIX + (t or "") for t in texts])

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Embed a user question with Nomic's query prefix."""
        res = self._encode([QUERY_PREFIX + (text or "")])
        return res[0] if res else None

    # The RAG path embeds questions, so the bare helpers use the query prefix.
    def embed_text(self, text: str) -> Optional[List[float]]:
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        return self._encode([QUERY_PREFIX + (t or "") for t in texts])

    def embed_batch(self, texts: List[str], delay: float = 0) -> List[Optional[List[float]]]:
        """Alias retained for older call sites."""
        return self.embed_texts(texts)

    # -- health --------------------------------------------------------------

    def check_health(self) -> bool:
        try:
            return self.embed_query("health probe") is not None
        except Exception as exc:
            logger.error("Embedding health-check failed: %s", exc)
            return False

    def probe(self) -> dict:
        """Load the model and embed a probe, verifying width and finiteness."""
        info = {
            "provider": "local-torch",
            "endpoint": self.base_url,
            "model": self.model_name,
            "device": None,
            "healthy": False,
            "dimension": None,
            "expected_dimension": self.expected_dim,
            "dimension_ok": False,
            "error": None,
        }
        try:
            self.load()
            info["device"] = self.device
            vec = self.embed_query("dimension probe")
            if not vec:
                info["error"] = "model returned no embedding"
                return info
            info["dimension"] = len(vec)
            info["dimension_ok"] = len(vec) == self.expected_dim
            finite = all(isinstance(v, float) and v == v and abs(v) != float("inf")
                         for v in vec)
            if not finite:
                info["error"] = "embedding contains NaN or Inf"
                return info
            info["healthy"] = info["dimension_ok"]
            if not info["dimension_ok"]:
                info["error"] = (f"dimension {len(vec)} != expected "
                                 f"{self.expected_dim}")
        except Exception as exc:
            info["error"] = str(exc)
        return info


# --------------------------------------------------------------------------- #
# process-wide singleton
# --------------------------------------------------------------------------- #

_EMBEDDER: Optional[NomicLocalEmbedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(**_ignored) -> NomicLocalEmbedder:
    """Return the shared local embedder.

    One instance per process: the weights are loaded once and reused, so an
    API request never pays the load cost. ``**_ignored`` absorbs legacy
    keyword arguments from older call sites.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = NomicLocalEmbedder()
    return _EMBEDDER
