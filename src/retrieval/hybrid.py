"""Hybrid retriever: fuses BM25 sparse and FAISS dense results via Reciprocal Rank Fusion."""

import os
from typing import Any, TypedDict

from dotenv import load_dotenv

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

load_dotenv()

# Fixed pool drawn from each retriever before fusion.
# The pipeline spec calls for BM25(top 20) + FAISS(top 20) → RRF → top 20.
_POOL_SIZE = 20

# RRF smoothing constant from the original Cormack & Clarke 2009 paper.
# k=60 is the de-facto default; larger k reduces the influence of top ranks.
_DEFAULT_RRF_K = 60


class HybridResult(TypedDict):
    """A single result returned by :class:`HybridRetriever`."""

    chunk_id: str
    text: str
    source: str       # arxiv | rbi | bis | sebi
    filename: str
    page_number: int
    chunk_index: int
    rrf_score: float  # sum of 1/(k+rank) contributions across retrieval lists
    appeared_in: list[str]  # subset of ["bm25", "vector"]


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 (sparse) and FAISS (dense) results using
    Reciprocal Rank Fusion (RRF).

    **RRF formula**::

        score(chunk) = Σ  1 / (k + rank_in_list)

    where k=60 is the standard smoothing constant and the sum runs over every
    ranked list the chunk appears in.  A chunk ranked 3rd in BM25 AND 5th in
    FAISS scores higher than a chunk ranked 1st in only one list, because it has
    contributions from both.

    Each retriever is always queried for a fixed pool of ``_POOL_SIZE=20``
    candidates.  After fusion the results are deduplicated by ``chunk_id``,
    sorted by RRF score descending, and trimmed to ``top_k``.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = _DEFAULT_RRF_K,
    ) -> None:
        """
        Args:
            bm25_retriever: Initialized :class:`BM25Retriever` instance.
            vector_retriever: Initialized :class:`VectorRetriever` instance.
            rrf_k: RRF smoothing constant (default 60).
        """
        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        explain: bool = False,
    ) -> list[HybridResult]:
        """
        Fuse BM25 and FAISS results for *query* using RRF and return the top-K.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return (default 20).
            explain: If ``True``, print a debug table showing per-chunk BM25
                     rank, vector rank, and final RRF score.

        Returns:
            List of :class:`HybridResult` dicts, at most *top_k* items, ordered
            by ``rrf_score`` descending.  No duplicate ``chunk_id`` values.
        """
        bm25_hits = self._bm25.retrieve(query, top_k=_POOL_SIZE)
        vector_hits = self._vector.retrieve(query, top_k=_POOL_SIZE)

        rrf_scores, chunk_store, bm25_rank_map, vector_rank_map = self._fuse(
            bm25_hits, vector_hits
        )

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results: list[HybridResult] = []
        for cid in ranked_ids[:top_k]:
            base = chunk_store[cid]
            appeared: list[str] = []
            if cid in bm25_rank_map:
                appeared.append("bm25")
            if cid in vector_rank_map:
                appeared.append("vector")
            results.append(
                HybridResult(
                    chunk_id=cid,
                    text=base["text"],
                    source=base.get("source", ""),
                    filename=base.get("filename", ""),
                    page_number=base.get("page_number", 0),
                    chunk_index=base.get("chunk_index", -1),
                    rrf_score=round(rrf_scores[cid], 6),
                    appeared_in=appeared,
                )
            )

        if explain:
            _print_explain_table(results, bm25_rank_map, vector_rank_map)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse(
        self,
        bm25_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
    ) -> tuple[
        dict[str, float],        # chunk_id → rrf_score
        dict[str, dict[str, Any]],  # chunk_id → first-seen chunk data
        dict[str, int],          # chunk_id → bm25 rank (1-indexed)
        dict[str, int],          # chunk_id → vector rank (1-indexed)
    ]:
        """
        Core RRF computation — intentionally kept small and readable.

        Rank is 1-indexed (position 0 in the list = rank 1).
        A chunk that appears in both lists receives two additive contributions.
        ``chunk_store`` retains the first-seen copy of each chunk so metadata
        is always available regardless of which list contributed it.
        """
        rrf_scores: dict[str, float] = {}
        chunk_store: dict[str, dict[str, Any]] = {}
        bm25_rank_map: dict[str, int] = {}
        vector_rank_map: dict[str, int] = {}

        for rank, chunk in enumerate(bm25_hits, start=1):
            cid: str = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
            chunk_store.setdefault(cid, chunk)
            bm25_rank_map[cid] = rank

        for rank, chunk in enumerate(vector_hits, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
            chunk_store.setdefault(cid, chunk)
            vector_rank_map[cid] = rank

        return rrf_scores, chunk_store, bm25_rank_map, vector_rank_map


# ------------------------------------------------------------------
# Debug / explain helper
# ------------------------------------------------------------------

def _print_explain_table(
    results: list[HybridResult],
    bm25_rank_map: dict[str, int],
    vector_rank_map: dict[str, int],
) -> None:
    """Print a formatted table of RRF fusion details for *results*."""
    col_widths = (4, 36, 10, 12, 10, 20)
    headers = ("#", "chunk_id", "bm25_rank", "vector_rank", "rrf_score", "appeared_in")
    sep = "-" * (sum(col_widths) + len(col_widths) * 2)

    def row(values: tuple[Any, ...]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(values, col_widths))

    print(sep)
    print(row(headers))
    print(sep)
    for pos, r in enumerate(results, start=1):
        cid = r["chunk_id"]
        print(
            row((
                pos,
                cid,
                bm25_rank_map.get(cid, "-"),
                vector_rank_map.get(cid, "-"),
                f"{r['rrf_score']:.6f}",
                ", ".join(r["appeared_in"]),
            ))
        )
    print(sep)


# ------------------------------------------------------------------
# __main__ — run a sample query against the pre-built indexes
# ------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    _index_dir = Path(os.getenv("INDEX_DIR", "indexes"))
    _embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    print("Loading BM25 index...")
    _bm25 = BM25Retriever.load(_index_dir / "bm25_index.pkl")

    print("Loading FAISS index...")
    _vector = VectorRetriever.load(_index_dir / "faiss_index", _embed_model)

    _retriever = HybridRetriever(_bm25, _vector)

    _query = "What is RBI's approach to digital currency?"
    print(f'\nQuery: "{_query}"')
    print("Top 10 results (explain=True):\n")

    _results = _retriever.retrieve(_query, top_k=10, explain=True)

    for _i, _r in enumerate(_results, 1):
        print(
            f"\n[{_i}] {_r['source'].upper()} — {_r['filename']} (page {_r['page_number']})"
        )
        print(f"     rrf_score={_r['rrf_score']:.6f}  appeared_in={_r['appeared_in']}")
        print(f"     {_r['text'][:200].strip()}...")
