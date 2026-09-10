# RAG Pipeline — Setup Guide

## Prerequisites

- Python 3.10+
- MongoDB (local or Atlas)
- Ollama installed locally
- ~8 GB RAM minimum (16 GB+ recommended for 14b model)

---

## 1. Install Ollama

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify installation
```bash
ollama --version
```

---

## 2. Start Ollama Server

```bash
ollama serve
```

By default this runs on `http://localhost:11434`. To bind to a custom address
(e.g. `http://172.16.212.229:11434`), set the environment variable:

```bash
OLLAMA_HOST=172.16.212.229:11434 ollama serve
```

---

## 3. Pull Required Models

### Embedding model (required)
```bash
ollama pull nomic-embed-text
```

### LLM — choose ONE based on your RAM

| Model | RAM Required | Command |
|-------|-------------|---------|
| qwen2.5:7b | ~8 GB | `ollama pull qwen2.5:7b` |
| qwen2.5:14b | ~16 GB | `ollama pull qwen2.5:14b` |

```bash
# 7B (default — works on most machines)
ollama pull qwen2.5:7b

# 14B (better quality, needs more RAM)
ollama pull qwen2.5:14b
```

### Verify models are available
```bash
ollama list
```

---

## 4. Install Python Dependencies

```bash
cd rag_pipeline
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

---

## 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
MONGODB_URI=mongodb://localhost:27017
DB_NAME=blura_saga
COLLECTION_NAME=posts
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
VECTOR_COLLECTION=vector_embeddings
```

If Ollama runs on a remote machine, update `OLLAMA_BASE_URL` accordingly:
```env
OLLAMA_BASE_URL=http://172.16.212.229:11434
```

---

## 6. Run the Pipeline

### Step 1 — Health check
```bash
python pipeline.py --check
```

This verifies:
- MongoDB is reachable and the source collection exists
- Ollama is running and both models (embedding + LLM) are pulled

### Step 2 — Ingest data
```bash
python pipeline.py --ingest
```

This streams all documents from your MongoDB collection, converts them to
text, chunks them, generates embeddings via Ollama, and stores everything
in the `vector_embeddings` collection.

**Resumable**: If the process is interrupted, re-run the same command.
Already-embedded documents are skipped automatically.

### Step 3 — Ask questions
```bash
python pipeline.py --query "What are the most recent incidents?"
```

### Step 4 — View stats
```bash
python pipeline.py --stats
```

---

## 7. Performance Tips

- **Ingestion speed** depends heavily on Ollama embedding throughput.
  On Apple Silicon with Metal, `nomic-embed-text` embeds ~50–100 chunks/sec.
- For large collections (100k+ docs), consider running ingestion overnight.
- The cosine search is brute-force numpy. For < 500k chunks it runs in
  under 1 second. For larger stores, enable Atlas Vector Search (see
  comments in `vector_store.py`).
- If you have a GPU, Ollama will automatically use it for faster inference.

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ConnectionError: Cannot reach Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull <model-name>` |
| `MongoDB connection refused` | Check `mongod` is running and URI is correct |
| `tiktoken` import error | `pip install tiktoken` (needs Rust toolchain on some systems) |
| Out of memory during ingestion | Reduce `BATCH_SIZE` in `.env` to 50 or lower |
| Slow embeddings | Normal for CPU; GPU significantly speeds this up |
