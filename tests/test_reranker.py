"""Tests for CrossEncoderReranker."""

import pytest
from unittest.mock import patch, MagicMock

from src.reranker.cross_encoder import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, rrf_score: float = 0.02) -> dict:
    """Minimal chunk dict that mirrors HybridResult fields."""
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
def reranker_and_mock():
    """CrossEncoderReranker with the underlying CrossEncoder model mocked out.

    Prevents HuggingFace model downloads during tests.  The fixture yields
    ``(reranker_instance, mock_model)`` so tests can control predict() output.
    """
    with patch("src.reranker.cross_encoder.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        ranker = CrossEncoderReranker("mock-model")
        yield ranker, mock_model


# ---------------------------------------------------------------------------
# Test 1 — output length is at most top_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_output_length_at_most_top_k(reranker_and_mock, top_k: int):
    """rerank() must return at most top_k chunks regardless of input size."""
    ranker, mock_model = reranker_and_mock
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    mock_model.predict.return_value = list(range(10, 0, -1))  # scores 10..1

    result = ranker.rerank("query", chunks, top_k=top_k)

    assert len(result) <= top_k, f"Expected ≤{top_k} results, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 2 — results are sorted descending by ce_score
# ---------------------------------------------------------------------------

def test_sorted_by_ce_score_descending(reranker_and_mock):
    """rerank() output must be ordered by ce_score from highest to lowest."""
    ranker, mock_model = reranker_and_mock
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    mock_model.predict.return_value = [0.3, 0.9, 0.1, 0.7, 0.5]

    result = ranker.rerank("query", chunks, top_k=5)
    scores = [r["ce_score"] for r in result]

    assert scores == sorted(scores, reverse=True), (
        f"Scores not in descending order: {scores}"
    )


# ---------------------------------------------------------------------------
# Test 3 — ce_score is attached to every result
# ---------------------------------------------------------------------------

def test_ce_score_attached_to_every_result(reranker_and_mock):
    """Every chunk in the output must have a ce_score float key."""
    ranker, mock_model = reranker_and_mock
    chunks = [make_chunk(f"c{i}") for i in range(4)]
    mock_model.predict.return_value = [0.8, 0.6, 0.4, 0.2]

    result = ranker.rerank("query", chunks, top_k=4)

    for r in result:
        assert "ce_score" in r, f"ce_score missing from chunk {r['chunk_id']}"
        assert isinstance(r["ce_score"], float), (
            f"ce_score must be float, got {type(r['ce_score'])}"
        )


# ---------------------------------------------------------------------------
# Test 4 — original rrf_score is preserved in output
# ---------------------------------------------------------------------------

def test_rrf_score_preserved_in_output(reranker_and_mock):
    """rerank() must not mutate or discard rrf_score from input chunks."""
    ranker, mock_model = reranker_and_mock
    # Assign distinct rrf_scores so we can verify each one individually.
    chunks = [make_chunk(f"c{i}", rrf_score=round(0.1 * (i + 1), 2)) for i in range(3)]
    # Return scores in reverse-rrf order to ensure reranking actually reorders.
    mock_model.predict.return_value = [0.9, 0.5, 0.7]

    result = ranker.rerank("query", chunks, top_k=3)
    result_by_id = {r["chunk_id"]: r for r in result}

    assert result_by_id["c0"]["rrf_score"] == pytest.approx(0.1)
    assert result_by_id["c1"]["rrf_score"] == pytest.approx(0.2)
    assert result_by_id["c2"]["rrf_score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Edge case — empty input returns empty output
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(reranker_and_mock):
    ranker, mock_model = reranker_and_mock
    assert ranker.rerank("query", [], top_k=5) == []
    mock_model.predict.assert_not_called()
