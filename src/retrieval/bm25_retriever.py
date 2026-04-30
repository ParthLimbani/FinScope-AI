"""Sparse BM25 retriever: builds and queries a rank_bm25 index over all chunks."""

import os
import pickle
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()


class BM25Retriever:
    """
    BM25-based sparse retriever backed by rank_bm25.

    The BM25 model and the original chunk list are serialized together into a
    single pickle file so the retriever can be restored without rebuilding.
    Tokenization is whitespace-based lowercase splitting — fast and sufficient
    for keyword-heavy regulatory text.
    """

    def __init__(self, bm25: BM25Okapi, chunks: list[dict[str, Any]]) -> None:
        self._bm25 = bm25
        self._chunks = chunks

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        index_path: Path,
    ) -> "BM25Retriever":
        """
        Tokenize chunk texts, build a BM25Okapi index, and persist to disk.

        Args:
            chunks: List of chunk dicts — each must contain a ``text`` key.
            index_path: Destination ``.pkl`` file path.

        Returns:
            Initialized :class:`BM25Retriever` instance.
        """
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
        """
        Load a previously built BM25 index from disk.

        Args:
            index_path: Path to the ``.pkl`` file created by :meth:`build`.

        Returns:
            Initialized :class:`BM25Retriever` instance.
        """
        print(f"[BM25] Loading existing index from {index_path}")
        with open(index_path, "rb") as fh:
            data = pickle.load(fh)
        print(f"[BM25] Loaded index with {len(data['chunks'])} chunks")
        return cls(data["bm25"], data["chunks"])

    @classmethod
    def load_or_build(
        cls,
        chunks: list[dict[str, Any]],
        index_path: Path,
    ) -> "BM25Retriever":
        """
        Load the index if it exists on disk, otherwise build and persist it.

        Args:
            chunks: Chunk list (only used when building).
            index_path: Path to the ``.pkl`` index file.

        Returns:
            Initialized :class:`BM25Retriever` instance.
        """
        if index_path.exists():
            return cls.load(index_path)
        return cls.build(chunks, index_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Return the top-K most relevant chunks for a natural-language query.

        Tokenizes the query identically to the corpus (lowercase split).
        Ties are broken by corpus insertion order.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return. Defaults to the TOP_K_RETRIEVE
                   environment variable, or ``20``.

        Returns:
            List of chunk dicts ordered by descending BM25 score.
        """
        k = top_k if top_k is not None else int(os.getenv("TOP_K_RETRIEVE", "20"))
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_indices]
