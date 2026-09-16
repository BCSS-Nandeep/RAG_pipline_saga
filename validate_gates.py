#!/usr/bin/env python3
"""
validate_gates.py — the checks that must pass BEFORE any regeneration runs.

    python validate_gates.py

    1  CUDA / PyTorch verification
    2  local Nomic model loads
    3  real 768-dim DOCUMENT embedding
    4  real 768-dim QUERY embedding
    5  prefix differentiation (search_document: vs search_query:)
    6  semantic retrieval using freshly-made vectors
    7  retrieval -> Qwen3-14B -> grounded answer

Gate 6 embeds a small in-memory pilot from a real source collection with the
new model and searches that, so both sides of the comparison come from the
same model. Nothing is written to MongoDB and no existing vector is touched —
this runs safely before the corpus is rebuilt.

Exit code 0 only when every gate passes.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(message)s")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
DB_NAME = os.getenv("DB_NAME", "test")
PILOT_COLLECTION = os.getenv("GATE_PILOT_COLLECTION", "alerts")
PILOT_DOCS = int(os.getenv("GATE_PILOT_DOCS", "150"))

PROBES = [
    ("accidents",  "Give me an accident post which are in alerts"),
    ("grievances", "Show me grievance posts"),
    ("political",  "Show me political posts"),
    ("traffic",    "Show me traffic related complaints"),
    ("crime",      "Show me crime and law and order incidents"),
]

_gates: List[tuple] = []


def gate(n: int, name: str, ok: Optional[bool], detail: str = "") -> bool:
    _gates.append((n, name, ok, detail))
    mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"  [{mark}] Gate {n}: {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    print("=" * 76)
    print("PRE-REGENERATION VALIDATION GATES")
    print("=" * 76)

    # ---- Gate 1: CUDA / PyTorch ------------------------------------------
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, cuda build {torch.version.cuda}"
        if cuda_ok:
            p = torch.cuda.get_device_properties(0)
            detail += f", {p.name} {p.total_memory/1e9:.2f} GB"
            # prove the GPU actually computes, not just that it is listed
            x = torch.randn(512, 512, device="cuda")
            float((x @ x).sum().item())
            detail += ", matmul OK"
        else:
            detail += ", CUDA UNAVAILABLE -> CPU"
        gate(1, "CUDA / PyTorch", True, detail)
        if not cuda_ok:
            print("         (CPU works but embedding will be markedly slower)")
    except Exception as exc:
        gate(1, "CUDA / PyTorch", False, str(exc)[:120])
        return summary()

    # ---- Gate 2: model loads ---------------------------------------------
    from embedder import (EXPECTED_DIM, DOCUMENT_PREFIX, QUERY_PREFIX,
                          describe_hardware, get_embedder)
    print(f"         hardware: {describe_hardware()}")
    try:
        emb = get_embedder()
        t0 = time.time()
        emb.load()
        gate(2, "local Nomic model loads", True,
             f"{emb.model_name} on {emb.device} in {time.time()-t0:.1f}s, "
             f"batch={emb.batch_size}")
    except Exception as exc:
        gate(2, "local Nomic model loads", False, str(exc)[:200])
        return summary()

    # ---- Gate 3: document embedding --------------------------------------
    doc_text = "A road accident was reported near Hyderabad on the ORR."
    try:
        dvec = emb.embed_documents([doc_text])[0]
        finite = dvec is not None and all(v == v and abs(v) != float("inf") for v in dvec)
        gate(3, "768-dim DOCUMENT embedding",
             bool(dvec) and len(dvec) == EXPECTED_DIM and finite,
             f"dim={len(dvec) if dvec else None}, finite={finite}, "
             f"prefix={DOCUMENT_PREFIX!r}")
    except Exception as exc:
        gate(3, "768-dim DOCUMENT embedding", False, str(exc)[:160])
        dvec = None

    # ---- Gate 4: query embedding -----------------------------------------
    try:
        qvec = emb.embed_query(doc_text)
        finite = qvec is not None and all(v == v and abs(v) != float("inf") for v in qvec)
        gate(4, "768-dim QUERY embedding",
             bool(qvec) and len(qvec) == EXPECTED_DIM and finite,
             f"dim={len(qvec) if qvec else None}, finite={finite}, "
             f"prefix={QUERY_PREFIX!r}")
    except Exception as exc:
        gate(4, "768-dim QUERY embedding", False, str(exc)[:160])
        qvec = None

    # ---- Gate 5: prefixes actually differ --------------------------------
    if dvec and qvec:
        import math
        dot = sum(a * b for a, b in zip(dvec, qvec))
        na = math.sqrt(sum(a * a for a in dvec))
        nb = math.sqrt(sum(b * b for b in qvec))
        cos = dot / (na * nb) if na and nb else 0.0
        # Same sentence, different instruction: related but not identical.
        gate(5, "prefix differentiation", dvec != qvec and cos < 0.999,
             f"cosine(document, query) = {cos:.6f} on identical text")
    else:
        gate(5, "prefix differentiation", None, "needs gates 3 and 4")

    # ---- Gate 6: semantic retrieval on fresh same-model vectors ----------
    import numpy as np
    from pymongo import MongoClient
    from processor import DocumentConverter

    retrieved_ctx: List[str] = []
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        db = client[DB_NAME]
        conv = DocumentConverter()
        texts, labels = [], []
        for doc in db[PILOT_COLLECTION].find({}).limit(PILOT_DOCS):
            t = conv.convert(doc)
            if t.strip():
                texts.append(t[:1500])
                labels.append(str(doc.get("_id")))
        client.close()

        if not texts:
            gate(6, "semantic retrieval", False,
                 f"no readable text in {PILOT_COLLECTION}")
        else:
            t0 = time.time()
            dvecs = emb.embed_documents(texts)
            build_s = time.time() - t0
            keep = [(v, l, t) for v, l, t in zip(dvecs, labels, texts) if v]
            M = np.array([v for v, _, _ in keep], dtype=np.float32)
            M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

            print(f"         pilot: {len(keep)} chunks from '{PILOT_COLLECTION}' "
                  f"embedded in {build_s:.1f}s "
                  f"({len(keep)/max(build_s,1e-6):.0f}/s)")

            per_query_top = {}
            for cat, q in PROBES:
                qv = np.array(emb.embed_query(q), dtype=np.float32)
                qv /= np.linalg.norm(qv)
                scores = M @ qv
                order = np.argsort(scores)[::-1][:3]
                per_query_top[cat] = [keep[i][1] for i in order]
                print(f"         {cat:<11} top={scores[order[0]]:.4f} "
                      f"ids={per_query_top[cat][:2]}")
                if cat == PROBES[0][0]:
                    retrieved_ctx = [keep[i][2][:700] for i in order]

            # different questions must not all land on the same documents
            sets = [set(v) for v in per_query_top.values()]
            identical = all(s == sets[0] for s in sets)
            gate(6, "semantic retrieval (fresh same-model vectors)", not identical,
                 f"{len(keep)} vectors searched, "
                 f"{'all queries returned the SAME docs' if identical else 'per-question document sets differ'}")
    except Exception as exc:
        gate(6, "semantic retrieval", False, str(exc)[:200])

    # ---- Gate 7: retrieval -> Qwen3-14B ----------------------------------
    try:
        import llm_client
        if not llm_client.check_health():
            gate(7, "retrieval -> Qwen3-14B RAG", False,
                 f"LLM unhealthy at {llm_client.LLM_BASE_URL}")
        elif not retrieved_ctx:
            gate(7, "retrieval -> Qwen3-14B RAG", None, "no retrieved context")
        else:
            ctx = "\n\n---\n\n".join(retrieved_ctx)
            prompt = (
                "Use ONLY the records below to answer. If they do not contain "
                "the answer, say so plainly.\n\n"
                f"=== RECORDS ===\n{ctx}\n=== END ===\n\n"
                f"Question: {PROBES[0][1]}\n\nAnswer:")
            meta: dict = {}
            t0 = time.time()
            answer = llm_client.generate(prompt, max_tokens=400,
                                         temperature=0.2, meta=meta)
            ok = (meta.get("status") == 200 and answer
                  and "LLM generation failed" not in answer)
            gate(7, "retrieval -> Qwen3-14B RAG", ok,
                 f"{meta.get('model')} status={meta.get('status')} "
                 f"{time.time()-t0:.0f}s, {len(answer)} chars")
            if ok:
                print("\n         --- grounded answer (first 400 chars) ---")
                for line in answer[:400].split("\n"):
                    print("         " + line)
    except Exception as exc:
        gate(7, "retrieval -> Qwen3-14B RAG", False, str(exc)[:200])

    return summary()


def summary() -> int:
    print()
    print("=" * 76)
    passed = sum(1 for *_, o, _ in [(g[0], g[1], g[2], g[3]) for g in _gates] if o is True)
    failed = sum(1 for g in _gates if g[2] is False)
    skipped = sum(1 for g in _gates if g[2] is None)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    if failed or skipped:
        print("\n  Not clear to regenerate yet:")
        for n, name, o, det in _gates:
            if o is not True:
                print(f"    Gate {n} ({name}): {det}")
    else:
        print("\n  ALL GATES PASSED — safe to run the full regeneration.")
    print("=" * 76)
    return 0 if (failed == 0 and skipped == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
