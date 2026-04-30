"""Tests for RAGPipeline — Groq API is always mocked; no real LLM calls."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# UUIDs used across tests — real one exists in retrieved set, fake one does not.
_REAL_ID = "550e8400-e29b-41d4-a716-446655440000"
_FAKE_ID = "ffffffff-ffff-4fff-afff-ffffffffffff"

_SAMPLE_CHUNKS = [
    {
        "chunk_id": _REAL_ID,
        "text": "The Reserve Bank of India has published comprehensive CBDC guidelines.",
        "source": "rbi",
        "filename": "rbi_cbdc_2023.pdf",
        "page_number": 5,
        "chunk_index": 10,
        "rrf_score": 0.032,
        "appeared_in": ["bm25", "vector"],
        "ce_score": 0.95,
    }
]


@pytest.fixture
def mock_pipeline():
    """
    RAGPipeline instance with every external dependency replaced by a mock.

    - BM25Retriever, VectorRetriever, HybridRetriever → MagicMock (no disk access)
    - CrossEncoderReranker → MagicMock (no model download)
    - GroqClient → MagicMock (no API key required; generate is set per-test)

    The fixture yields the pipeline with retriever and reranker pre-wired to
    return _SAMPLE_CHUNKS so tests only need to control the LLM response.
    """
    with (
        patch("src.pipeline.rag_pipeline.BM25Retriever"),
        patch("src.pipeline.rag_pipeline.VectorRetriever"),
        patch("src.pipeline.rag_pipeline.HybridRetriever"),
        patch("src.pipeline.rag_pipeline.CrossEncoderReranker"),
        patch("src.pipeline.rag_pipeline.GroqClient"),
    ):
        from src.pipeline.rag_pipeline import RAGPipeline

        pipe = RAGPipeline()

    # Wire controlled behaviour after construction (patches no longer needed).
    pipe._retriever = MagicMock()
    pipe._retriever.retrieve.return_value = _SAMPLE_CHUNKS

    pipe._reranker = MagicMock()
    pipe._reranker.rerank.return_value = _SAMPLE_CHUNKS

    pipe._llm = MagicMock()
    return pipe


# ---------------------------------------------------------------------------
# Test 1 — hallucinated citation triggers all_valid = False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_citation_detected(mock_pipeline):
    """
    citation_validation.all_valid must be False when the LLM cites a chunk_id
    that was not in the retrieved context.
    """
    fake_answer = (
        f"The RBI has outlined digital currency regulations [{_FAKE_ID}] "
        "building on BIS recommendations."
    )
    mock_pipeline._llm.generate = AsyncMock(return_value=fake_answer)

    result = await mock_pipeline.query("What is RBI's stance on CBDC?")

    assert result["citation_validation"]["all_valid"] is False
    assert _FAKE_ID in result["citation_validation"]["invalid"]


# ---------------------------------------------------------------------------
# Test 2 — response dict contains all required keys
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_returns_correct_structure(mock_pipeline):
    """query() must return a dict with all specified top-level and nested keys."""
    answer = (
        f"The RBI has comprehensive CBDC guidelines [{_REAL_ID}] that outline "
        "a phased rollout approach."
    )
    mock_pipeline._llm.generate = AsyncMock(return_value=answer)

    result = await mock_pipeline.query("What is RBI's CBDC policy?")

    # --- top-level keys ---
    for key in ("answer", "citations", "citation_validation", "metadata"):
        assert key in result, f"Missing top-level key: {key!r}"

    # --- citation entry keys ---
    assert len(result["citations"]) > 0, "citations list must not be empty"
    for key in ("chunk_id", "source", "filename", "page_number", "snippet",
                "ce_score", "rrf_score"):
        assert key in result["citations"][0], (
            f"Missing key in citations[0]: {key!r}"
        )

    # --- citation_validation keys ---
    for key in ("all_valid", "invalid", "uncited_chunks"):
        assert key in result["citation_validation"], (
            f"Missing key in citation_validation: {key!r}"
        )

    # --- metadata keys ---
    for key in ("retrieved", "reranked_to", "latency_ms"):
        assert key in result["metadata"], f"Missing key in metadata: {key!r}"


# ---------------------------------------------------------------------------
# Test 3 — valid citation is recognised correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_citation_recognised(mock_pipeline):
    """A chunk_id that IS in the retrieved set must appear in valid, not invalid."""
    answer = f"The RBI CBDC framework [{_REAL_ID}] is comprehensive."
    mock_pipeline._llm.generate = AsyncMock(return_value=answer)

    result = await mock_pipeline.query("CBDC?")

    cv = result["citation_validation"]
    assert cv["all_valid"] is True
    assert _REAL_ID not in cv["invalid"]
    assert _REAL_ID not in cv["uncited_chunks"]


# ---------------------------------------------------------------------------
# Test 4 — snippet is truncated to 200 characters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snippet_truncated_to_200_chars(mock_pipeline):
    """citations[*].snippet must be at most 200 characters long."""
    mock_pipeline._llm.generate = AsyncMock(return_value="Some answer.")

    result = await mock_pipeline.query("test")

    for citation in result["citations"]:
        assert len(citation["snippet"]) <= 200, (
            f"Snippet too long: {len(citation['snippet'])} chars"
        )
