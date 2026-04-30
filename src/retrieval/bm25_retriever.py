"""Sparse BM25 retriever: builds and queries a rank_bm25 index over all chunks."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

_log = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based sparse retriever backed by rank_bm25."""

    def __init__(self, bm25: BM25Okapi, chunks: list[dict[str, Any]]) -> None:
        self._bm25 = bm25
        self._chunks = chunks

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[dict[str, Any]], index_path: Path) -> "BM25Retriever":
        print(f"[BM25] Tokenizing {len(chunks)} chunks...")
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        print("[BM25] Building BM25Okapi index...")
        bm25 = BM25Okapi(tokenized)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "wb") as fh:
            pickle.dump({"bm25": bm25, "chunks": chunks}, fh)
        print(f"[BM25] Index saved → {index_path}")
        return cls(bm25, chunks)

    @classmethod
    def load(cls, index_path: Path) -> "BM25Retriever":
        print(f"[BM25] Loading existing index from {index_path}")
        with open(index_path, "rb") as fh:
            data = pickle.load(fh)
        print(f"[BM25] Loaded index with {len(data['chunks'])} chunks")
        return cls(data["bm25"], data["chunks"])

    @classmethod
    def load_or_build(cls, chunks: list[dict[str, Any]], index_path: Path) -> "BM25Retriever":
        if index_path.exists():
            return cls.load(index_path)
        return cls.build(chunks, index_path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def extend(self, new_chunks: list[dict[str, Any]], index_path: Path) -> None:
        """Append new chunks, rebuild BM25Okapi (not incrementally updatable), and persist."""
        all_chunks = self._chunks + new_chunks
        tokenized = [c["text"].lower().split() for c in all_chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunks = all_chunks
        with open(index_path, "wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, fh)
        _log.info("BM25: extended to %d chunks → %s", len(self._chunks), index_path)

    def remove_by_filename(self, filename: str, index_path: Path) -> int:
        """Remove all chunks for a given filename, rebuild index, and persist."""
        remaining = [c for c in self._chunks if c.get("filename") != filename]
        removed = len(self._chunks) - len(remaining)
        if removed == 0:
            return 0
        tokenized = [c["text"].lower().split() for c in remaining]
        self._bm25 = BM25Okapi(tokenized) if remaining else BM25Okapi([[]])
        self._chunks = remaining
        with open(index_path, "wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, fh)
        _log.info("BM25: removed %d chunks for '%s' → %d remaining", removed, filename, len(remaining))
        return removed

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k if top_k is not None else int(os.getenv("TOP_K_RETRIEVE", "20"))
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_indices]
