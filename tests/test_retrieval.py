"""Tests for the hybrid RRF retriever."""

import pytest

from src.retrieval.hybrid import HybridRetriever, _DEFAULT_RRF_K


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, source: str = "test") -> dict:
    """Minimal chunk dict that satisfies HybridRetriever's expectations."""
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "source": source,
        "filename": "test.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }


class MockRetriever:
    """Drop-in replacement for BM25Retriever / VectorRetriever in tests."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:  # noqa: ARG002
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Test 1 — output is deduplicated by chunk_id
# ---------------------------------------------------------------------------

def test_no_duplicate_chunk_ids():
    """A chunk_id that appears in both retriever lists must appear only once."""
    shared = [make_chunk(f"shared_{i}") for i in range(5)]
    bm25_only = [make_chunk(f"bm25_{i}") for i in range(10)]
    vector_only = [make_chunk(f"vec_{i}") for i in range(10)]

    bm25 = MockRetriever(bm25_only + shared)     # 15 results, 5 overlap
    vector = MockRetriever(vector_only + shared)  # 15 results, 5 overlap

    retriever = HybridRetriever(bm25, vector)
    results = retriever.retrieve("any query", top_k=30)

    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids)), (
        "Duplicate chunk_ids found in RRF output"
    )


# ---------------------------------------------------------------------------
# Test 2 — chunk in both lists outscores any chunk in only one list
# ---------------------------------------------------------------------------

def test_chunk_in_both_lists_outscores_single_list():
    """
    RRF property: a chunk contributing from both lists must have a higher
    fused score than any chunk contributing from only one list, provided
    its individual ranks are at least as good.

    Setup
    -----
    BM25  : [A, B, C]          ranks 1, 2, 3
    Vector: [A, D, E]          ranks 1, 2, 3

    Expected RRF scores (k=60)
    --------------------------
    A:  1/(60+1) + 1/(60+1) = 2/61  ≈ 0.03279
    B:  1/(60+2)             = 1/62  ≈ 0.01613
    D:  1/(60+2)             = 1/62  ≈ 0.01613
    """
    chunk_a = make_chunk("A")
    bm25 = MockRetriever([chunk_a, make_chunk("B"), make_chunk("C")])
    vector = MockRetriever([chunk_a, make_chunk("D"), make_chunk("E")])

    retriever = HybridRetriever(bm25, vector)
    results = retriever.retrieve("any query", top_k=10)

    rrf_by_id = {r["chunk_id"]: r["rrf_score"] for r in results}

    assert rrf_by_id["A"] > rrf_by_id["B"], (
        f"Expected A ({rrf_by_id['A']:.6f}) > B ({rrf_by_id['B']:.6f})"
    )
    assert rrf_by_id["A"] > rrf_by_id["D"], (
        f"Expected A ({rrf_by_id['A']:.6f}) > D ({rrf_by_id['D']:.6f})"
    )

    # Bonus: appeared_in must reflect both retrievers for A
    appeared = next(r["appeared_in"] for r in results if r["chunk_id"] == "A")
    assert set(appeared) == {"bm25", "vector"}


# ---------------------------------------------------------------------------
# Test 3 — results are sorted descending by rrf_score
# ---------------------------------------------------------------------------

def test_results_sorted_descending_by_rrf_score():
    """RRF output must be in non-increasing rrf_score order."""
    # BM25 and vector share chunks 5-9 (overlap), rest are unique.
    # The overlapping chunks will score higher; regardless, the list must be sorted.
    bm25_chunks = [make_chunk(f"b{i}") for i in range(10)]
    vector_chunks = [make_chunk(f"b{i}" if i >= 5 else f"v{i}") for i in range(10)]

    retriever = HybridRetriever(MockRetriever(bm25_chunks), MockRetriever(vector_chunks))
    results = retriever.retrieve("any query", top_k=20)

    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        "RRF results are not sorted in descending order"
    )


# ---------------------------------------------------------------------------
# Test 4 — top_k caps the number of returned results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 5, 10, 15])
def test_top_k_caps_result_count(top_k: int):
    """retrieve() must return at most top_k results."""
    # Both retrievers return 20 distinct chunks each (40 unique total after fusion).
    bm25 = MockRetriever([make_chunk(f"bm_{i}") for i in range(20)])
    vector = MockRetriever([make_chunk(f"vec_{i}") for i in range(20)])

    retriever = HybridRetriever(bm25, vector)
    results = retriever.retrieve("any query", top_k=top_k)

    assert len(results) <= top_k, (
        f"Expected at most {top_k} results, got {len(results)}"
    )


# ---------------------------------------------------------------------------
# Additional edge-case: exact RRF score values
# ---------------------------------------------------------------------------

def test_rrf_score_values_match_formula():
    """
    Verify the numeric RRF score with k=60 for a known configuration.

    BM25  : [X]   → rank 1  → contribution = 1/(60+1) = 1/61
    Vector: [X]   → rank 1  → contribution = 1/(60+1) = 1/61
    Expected rrf_score(X) = 2/61 ≈ 0.032787
    """
    chunk_x = make_chunk("X")
    retriever = HybridRetriever(
        MockRetriever([chunk_x]),
        MockRetriever([chunk_x]),
        rrf_k=60,
    )
    results = retriever.retrieve("any query", top_k=5)

    assert len(results) == 1
    expected = round(2.0 / 61.0, 6)
    assert results[0]["rrf_score"] == expected, (
        f"Expected {expected}, got {results[0]['rrf_score']}"
    )
