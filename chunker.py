"""
chunker.py — STAGE 3
  Token-aware text chunking with tiktoken (cl100k_base).
  Produces 300–800 token chunks with 50-token overlap.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import tiktoken

logger = logging.getLogger(__name__)

# Initialise encoder once (thread-safe, reusable)
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkMetadata:
    source_collection: str
    document_id: str
    chunk_index: int
    total_chunks: int


@dataclass
class TextChunk:
    text: str
    metadata: ChunkMetadata
    token_count: int = 0


class TokenAwareChunker:
    """Split a text block into chunks of *max_tokens* with *overlap* tokens."""

    def __init__(
        self,
        min_tokens: int = 300,
        max_tokens: int = 800,
        overlap: int = 50,
    ):
        if overlap >= max_tokens:
            raise ValueError("overlap must be less than max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap

    # -- public API ----------------------------------------------------------

    def chunk_document(
        self,
        text: str,
        source_collection: str,
        document_id: str,
    ) -> List[TextChunk]:
        """Return a list of ``TextChunk`` objects for a single document.

        Short documents (< max_tokens) are returned as a single chunk.
        """
        tokens = _ENCODER.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        # If the whole document fits in one chunk, return as-is
        if total_tokens <= self.max_tokens:
            meta = ChunkMetadata(
                source_collection=source_collection,
                document_id=document_id,
                chunk_index=0,
                total_chunks=1,
            )
            return [TextChunk(text=text, metadata=meta, token_count=total_tokens)]

        # Build raw slices
        raw_slices: List[tuple] = []  # (start, end)
        start = 0
        while start < total_tokens:
            end = min(start + self.max_tokens, total_tokens)
            raw_slices.append((start, end))
            if end == total_tokens:
                break
            start = end - self.overlap

        # Merge a trailing runt (< min_tokens) into the previous chunk
        if len(raw_slices) > 1:
            last_start, last_end = raw_slices[-1]
            if (last_end - last_start) < self.min_tokens:
                prev_start, _ = raw_slices[-2]
                raw_slices[-2] = (prev_start, last_end)
                raw_slices.pop()

        total_chunks = len(raw_slices)
        chunks: List[TextChunk] = []

        for idx, (s, e) in enumerate(raw_slices):
            chunk_tokens = tokens[s:e]
            chunk_text = _ENCODER.decode(chunk_tokens)
            meta = ChunkMetadata(
                source_collection=source_collection,
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks,
            )
            chunks.append(
                TextChunk(text=chunk_text, metadata=meta, token_count=len(chunk_tokens))
            )

        return chunks


def count_tokens(text: str) -> int:
    """Utility: count tokens in *text* using cl100k_base."""
    return len(_ENCODER.encode(text))
