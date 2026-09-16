# SOC-EYE RAG Pipeline

Retrieval-augmented question answering over the SOC-EYE MongoDB corpus.

```
MongoDB  ──►  Nomic Embed Text v1.5  ──►  vector collections
(source)      (local, GPU, in-process)     (MongoDB)
                                                │
user question ──► search_query: ──► embedding ──┤
                                                ▼
                                       vector search + RRF
                                                │
                                                ▼
                                   Qwen3-14B (self-hosted vLLM)
                                                │
                                                ▼
                                        grounded answer
```

All inference is self-hosted: embeddings run locally on this machine's GPU,
generation runs on our own vLLM server. No Ollama, no OpenAI, no Gemini, no
external embedding API.

## Quick start

```powershell
cp .env.example .env      # then fill in LLM_BASE_URL / LLM_API_KEY
pip install -r requirements.txt
.\run.ps1                 # http://127.0.0.1:8099/ui/
```

`run.ps1` preflights MongoDB, the LLM endpoint and the embedding model before
starting, so a misconfiguration fails at launch rather than on the first query.

## Models

| Role | Model | Where it runs |
|---|---|---|
| Embeddings | `nomic-ai/nomic-embed-text-v1.5`, 768-d | Locally, in-process (PyTorch/CUDA) |
| Generation | `qwen3-14b` | Self-hosted vLLM, `LLM_BASE_URL` |

Weights download automatically to the HuggingFace cache on first start and are
reused afterwards. Both the model and its `trust_remote_code` module are pinned
(`EMBED_REVISION`, `EMBED_CODE_REVISION`) so deploys are reproducible.

`EMBED_BATCH_SIZE=0` auto-tunes the batch from the GPU and VRAM actually
detected at runtime, and falls back to CPU when CUDA is unavailable.

**Nomic requires instruction prefixes**, applied automatically:

- documents → `search_document: `
- queries → `search_query: `

Mixing them degrades retrieval silently, so use `embed_documents()` for corpus
text and `embed_query()` for questions — never `embed_text()` for documents.

## Background ingest scheduler

Keeps vector collections in step with MongoDB without manual commands.

| Setting | Value | Meaning |
|---|---|---|
| `INGEST_SCHEDULER_ENABLED` | `true` | scheduler runs |
| `INGEST_INTERVAL_HOURS` | `8` | **28,800 s** between cycle starts |
| first cycle | 30 s after startup | keeps the API responsive at boot |
| `INGEST_COLLECTIONS` | comma-separated | collections synced each cycle |

Each cycle walks the configured collections sequentially; a failure in one is
caught and logged so the rest still run.

### How changes are detected

Every stored chunk records a SHA-1 `content_hash` of the text it was built
from. Each cycle re-hashes the source document and compares:

| State | Action |
|---|---|
| `_id` not in the vector store | **new** → embed |
| hash matches | **unchanged** → skipped, model not called |
| hash differs | **changed** → old chunks deleted, re-embedded |
| no hash recorded (legacy vector) | **left alone** — see below |

Old chunks are deleted before re-embedding so an edit that yields fewer chunks
cannot leave orphaned vectors behind that still match queries.

There is no progress file: the state lives in the vectors themselves. An
interrupted or failed document simply has no recorded hash, so the next cycle
treats it as new. That is what makes resume and retry automatic.

Detection is poll-based, so a change is picked up within one interval — worst
case 8 hours. Near-real-time would need MongoDB change streams, which require a
replica set.

### Legacy vectors are not touched

Vectors written before hash tracking have no `content_hash`. A missing hash is
**not** evidence that the source changed, so the scheduler skips them and
reports a `docs_legacy` count rather than deleting and re-embedding them.
Without this, enabling the scheduler over an existing corpus would silently
trigger a full rebuild on its first cycle.

Migrating those is a deliberate action:

```powershell
python regenerate_embeddings.py --preflight     # validate, change nothing
python regenerate_embeddings.py --rebuild       # retires old vectors first
python regenerate_embeddings.py --verify
```

## Deploying on a server

The embedding model runs **in the API process**, so the host needs a GPU (or
accepts slow CPU embedding). Generation is remote — only network access to the
vLLM server is required.

### 1. Host requirements

| | |
|---|---|
| Python | 3.10+ (3.14 verified) |
| GPU | NVIDIA, 4 GB VRAM or more, recent driver. CPU works but is ~10x slower |
| Disk | ~3 GB for PyTorch + ~600 MB for the model, plus the MongoDB data |
| Network | MongoDB, and outbound to `LLM_BASE_URL` |

### 2. Install

```bash
git clone <repo> && cd sockeye_rag_pipeline
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate

# torch FIRST, from the index matching the host's CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"     # must print True

pip install -r requirements.txt
```

If that prints `False`, the CPU wheel got installed — uninstall torch and
reinstall from the CUDA index, otherwise embedding silently runs on CPU.

### 3. Configure

```bash
cp .env.example .env
```

Set `MONGODB_URI`, `DB_NAME`, `LLM_BASE_URL` and `LLM_API_KEY`. Everything else
has a working default. **`.env` is gitignored — never commit it.**

### 4. Verify before serving

```bash
python pipeline.py --check          # MongoDB + embedding model + LLM
python validate_gates.py            # CUDA, 768-d, prefixes, retrieval, RAG
```

Both must pass. The first run downloads the model (~520 MB) into the
HuggingFace cache; later starts load from disk.

### 5. Run

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8099
```

On Windows `run.ps1` does the same with a preflight first.

**systemd unit:**

```ini
[Unit]
Description=SOC-EYE RAG API
After=network-online.target

[Service]
User=soceye
WorkingDirectory=/opt/soceye-rag
EnvironmentFile=/opt/soceye-rag/.env
ExecStart=/opt/soceye-rag/.venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8099
Restart=always
RestartSec=10
# The model loads into GPU memory at startup; give it room before a restart.
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now soceye-rag
curl -s localhost:8099/api/rag/health | jq .
```

### 6. Confirm it is healthy

`/api/rag/health` should report all three green:

```json
{"healthy": true,
 "embedding": {"status": "HEALTHY", "device": "cuda", "dimension": 768},
 "llm": {"status": "HEALTHY", "model": "qwen3-14b"}}
```

### Operational notes

- **One process per GPU.** Each worker loads its own copy of the model, so
  `--workers N` multiplies VRAM use. Scale with a queue, not workers.
- **First request after boot** may wait on warm-up; the API answers other
  routes immediately and `/api/rag/health` shows `warmup`.
- **The scheduler starts automatically** 30 s after boot and runs every 8 h.
  Set `INGEST_SCHEDULER_ENABLED=false` to disable it on a read-only replica.
- **Existing pre-hash vectors are left alone** by the scheduler. Migrating them
  is a deliberate `regenerate_embeddings.py --rebuild`.

## Retrieval and ranking

Database retrieval and vector retrieval score on scales that cannot be
compared — a recency ordering versus cosine similarity. They are combined with
Reciprocal Rank Fusion using each list's *rank*:

```
RRF(d) = Σ 1 / (K + rank(d))        K = RRF_K, default 60
```

so a strong semantic hit always competes instead of being crowded out by high
database scores.

Context sent to the LLM is selected by relevance and trimmed to a token budget
(`RAG_MAX_CONTEXT_TOKENS`), typically 2–4 chunks for a narrow question and more
for a broad one — never a fixed number, and never the whole database.

## Scripts

| Command | Purpose |
|---|---|
| `python validate_gates.py` | pre-regeneration gates: CUDA, model load, 768-d, prefixes, retrieval, RAG |
| `python validate_rag_pipeline.py` | full A–I pipeline validation |
| `python test_sync_workflow.py` | sync tests on an isolated DB (never touches production) |
| `python regenerate_embeddings.py --preflight` | validate before any rebuild |
| `python pipeline.py --check` | MongoDB + embedding + LLM health |
| `python pipeline.py --stats` | collection and vector counts |

## API

| Endpoint | Purpose |
|---|---|
| `GET /ui/` | browser test console |
| `GET /docs` | OpenAPI reference |
| `GET /api/rag/health` | MongoDB, embedding (model/device/dimension), LLM |
| `POST /api/rag/query` | ask a question |
| `POST /api/rag/ingest` | sync one collection now |
| `GET /api/rag/scheduler/status` | last run, next run, current collection |
| `POST /api/rag/scheduler/run-now` | trigger a cycle immediately |

## Requirements

- MongoDB (local)
- Python 3.14 with CUDA-enabled PyTorch for GPU embedding
- Reachable vLLM server running Qwen3-14B

On Windows keep `EMBED_CACHE_DIR` short. HuggingFace nests weights under
`models--<org>--<name>/snapshots/<40-char sha>/`, and a deep path exceeds the
260-character `MAX_PATH` limit, which surfaces as a misleading
`WinError 3: cannot find the path specified` during download.
