"""Tests for CrossEncoderReranker (API-based, no local model)."""

import pytest
from unittest.mock import patch

from src.reranker.cross_encoder import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, rrf_score: float = 0.02) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"Sample regulatory text for chunk {chunk_id}.",
        "source": "rbi",
        "filename": "rbi_test.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "rrf_score": rrf_score,
        "appeared_in": ["bm25", "vector"],
    }


@pytest.fixture
def reranker():
    """CrossEncoderReranker with _score_api mocked — no HF API calls."""
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# Test 1 — output length is at most top_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_output_length_at_most_top_k(reranker, top_k: int):
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    scores = list(range(10, 0, -1))  # 10..1

    with patch.object(reranker, "_score_api", return_value=scores):
        result = reranker.rerank("query", chunks, top_k=top_k)

    assert len(result) <= top_k


# ---------------------------------------------------------------------------
# Test 2 — results are sorted descending by ce_score
# ---------------------------------------------------------------------------

def test_sorted_by_ce_score_descending(reranker):
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    scores = [0.3, 0.9, 0.1, 0.7, 0.5]

    with patch.object(reranker, "_score_api", return_value=scores):
        result = reranker.rerank("query", chunks, top_k=5)

    ce_scores = [r["ce_score"] for r in result]
    assert ce_scores == sorted(ce_scores, reverse=True)


# ---------------------------------------------------------------------------
# Test 3 — ce_score is attached to every result
# ---------------------------------------------------------------------------

def test_ce_score_attached_to_every_result(reranker):
    chunks = [make_chunk(f"c{i}") for i in range(4)]
    scores = [0.8, 0.6, 0.4, 0.2]

    with patch.object(reranker, "_score_api", return_value=scores):
        result = reranker.rerank("query", chunks, top_k=4)

    for r in result:
        assert "ce_score" in r
        assert isinstance(r["ce_score"], float)


# ---------------------------------------------------------------------------
# Test 4 — original rrf_score is preserved
# ---------------------------------------------------------------------------

def test_rrf_score_preserved_in_output(reranker):
    chunks = [make_chunk(f"c{i}", rrf_score=round(0.1 * (i + 1), 2)) for i in range(3)]
    scores = [0.9, 0.5, 0.7]

    with patch.object(reranker, "_score_api", return_value=scores):
        result = reranker.rerank("query", chunks, top_k=3)

    result_by_id = {r["chunk_id"]: r for r in result}
    assert result_by_id["c0"]["rrf_score"] == pytest.approx(0.1)
    assert result_by_id["c1"]["rrf_score"] == pytest.approx(0.2)
    assert result_by_id["c2"]["rrf_score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Edge case — empty input returns empty output without calling API
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(reranker):
    with patch.object(reranker, "_score_api") as mock_api:
        result = reranker.rerank("query", [], top_k=5)
    assert result == []
    mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# Edge case — API failure falls back gracefully (all ce_score=0, sorted by rrf)
# ---------------------------------------------------------------------------

def test_api_failure_falls_back_gracefully(reranker):
    chunks = [make_chunk(f"c{i}") for i in range(3)]

    with patch.object(reranker, "_score_api", return_value=[0.0, 0.0, 0.0]):
        result = reranker.rerank("query", chunks, top_k=3)

    assert len(result) == 3
    for r in result:
        assert r["ce_score"] == 0.0
