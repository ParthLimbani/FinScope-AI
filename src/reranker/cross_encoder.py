"""Cross-encoder reranker: re-scores (query, chunk) pairs for precision ranking."""

import os
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Second-stage reranker that scores (query, passage) pairs jointly.

    Unlike bi-encoders (FAISS), a cross-encoder processes the full query+passage
    concatenation, producing much more accurate relevance scores at the cost of
    higher latency.  Used to re-rank the top-20 hybrid retrieval candidates and
    return the top-5 before generation.

    Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — ~22 MB, runs on CPU in
    roughly 1 second for 20 candidate pairs.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder identifier.  Defaults to the
                        RERANKER_MODEL environment variable, or
                        ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        """
        name = model_name or os.getenv("RERANKER_MODEL", _DEFAULT_MODEL)
        print(f"[Reranker] Loading cross-encoder '{name}'...")
        self._model = CrossEncoder(name)
        self._model_name = name

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Score every (query, chunk_text) pair and return the top-K by ``ce_score``.

        Each output chunk is a shallow copy of the input dict with a new
        ``ce_score`` (float) key appended.  All original keys — including
        ``rrf_score`` and ``appeared_in`` — are preserved intact so the caller
        retains full provenance.

        Args:
            query: Natural-language query string.
            chunks: Candidate chunks from the hybrid retriever.  Each dict must
                    contain a ``text`` key.
            top_k: Maximum number of chunks to return (default 5).

        Returns:
            List of chunk dicts (at most *top_k* items) with ``ce_score``
            attached, sorted by ``ce_score`` descending.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        raw_scores = self._model.predict(pairs)

        scored: list[dict[str, Any]] = []
        for chunk, score in zip(chunks, raw_scores):
            result = dict(chunk)          # shallow copy — preserves rrf_score etc.
            result["ce_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda x: x["ce_score"], reverse=True)
        return scored[:top_k]
