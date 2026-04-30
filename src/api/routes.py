"""API route handlers for FinScope AI."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.generator.llm import LLMError

router = APIRouter()

# Populated by the lifespan context manager in main.py.
_state: dict[str, Any] = {"pipeline": None, "model": "unknown"}

# Absolute project root — used to jail the /ingest data_dir to safe paths.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_pipeline():
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised — indexes may still be loading or startup failed.",
        )
    return pipeline


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, pipeline=Depends(_get_pipeline)):
    """Run the full RAG pipeline and return a cited answer."""
    try:
        result = await pipeline.query(
            body.question,
            top_k_retrieve=20,
            top_k_rerank=body.top_k,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest):
    """Re-run the ingestion pipeline and rebuild both indexes from scratch."""
    # Resolve and jail the supplied path to prevent directory traversal.
    data_path = (_PROJECT_ROOT / body.data_dir).resolve()
    if not str(data_path).startswith(str(_PROJECT_ROOT)):
        raise HTTPException(status_code=400, detail="data_dir must be within the project directory.")

    def _run_ingestion() -> tuple[int, float]:
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_pdfs
        from src.ingestion.metadata import attach_metadata
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.vector_retriever import VectorRetriever

        t0 = time.perf_counter()
        pages = load_pdfs(data_path)
        raw_chunks = chunk_documents(pages)
        chunks = attach_metadata(raw_chunks)

        index_dir = Path(os.getenv("INDEX_DIR", "indexes"))
        embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        BM25Retriever.build(chunks, index_dir / "bm25_index.pkl")
        VectorRetriever.build(chunks, index_dir / "faiss_index", embed_model)

        return len(chunks), round((time.perf_counter() - t0) * 1000, 2)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        chunks_indexed, elapsed_ms = await loop.run_in_executor(pool, _run_ingestion)

    return IngestResponse(
        status="ok",
        chunks_indexed=chunks_indexed,
        index_build_time_ms=elapsed_ms,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return server liveness and pipeline readiness."""
    pipeline = _state.get("pipeline")
    return HealthResponse(
        status="healthy",
        indexes_loaded=pipeline is not None,
        model=_state.get("model", "unknown"),
    )
