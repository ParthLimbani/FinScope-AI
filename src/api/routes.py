"""API route handlers for FinScope AI."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.schemas import (
    DeleteResponse,
    FileItem,
    FilesResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from src.generator.llm import LLMError

_log = logging.getLogger(__name__)

router = APIRouter()

_state: dict[str, Any] = {"pipeline": None, "model": "unknown"}
_init_lock = asyncio.Lock()

# Absolute project root — used to jail the /ingest data_dir to safe paths.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_pipeline_sync():
    from src.pipeline.rag_pipeline import RAGPipeline
    return RAGPipeline()


async def _get_pipeline():
    """Lazily initialise the pipeline on first request, then cache it."""
    if _state["pipeline"] is None:
        async with _init_lock:
            if _state["pipeline"] is None:  # double-check after acquiring lock
                _log.info("First request — loading pipeline (this takes ~30s)...")
                loop = asyncio.get_event_loop()
                pipeline = await loop.run_in_executor(None, _load_pipeline_sync)
                _state["pipeline"] = pipeline
                _state["model"] = pipeline._llm.model
                _log.info("Pipeline ready.")
    return _state["pipeline"]


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, pipeline=Depends(_get_pipeline)):
    """Run the full RAG pipeline and return a cited answer."""
    try:
        result = await pipeline.query(
            body.question,
            top_k_retrieve=20,
            top_k_rerank=body.top_k,
            history=[h.model_dump() for h in body.history],
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest):
    """Re-run the ingestion pipeline and rebuild both indexes from scratch."""
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


@router.get("/files", response_model=FilesResponse)
async def list_files():
    """List all PDF files currently in the data directory."""
    data_dir = _PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    files: list[FileItem] = []
    if data_dir.exists():
        for pdf in sorted(data_dir.rglob("*.pdf")):
            relative = pdf.relative_to(data_dir)
            source = relative.parts[0].lower() if len(relative.parts) > 1 else "uploads"
            files.append(FileItem(
                filename=pdf.name,
                source=source,
                size_kb=round(pdf.stat().st_size / 1024, 1),
            ))
    return FilesResponse(files=files, total=len(files))


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), pipeline=Depends(_get_pipeline)):
    """Upload a PDF, chunk it, and add it incrementally to the live index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    upload_dir = _PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())
    _log.info("Uploaded PDF saved → %s", dest)

    def _ingest() -> int:
        import fitz
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.metadata import attach_metadata

        doc = fitz.open(str(dest))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text().strip()
            if len(text) >= 10:
                pages.append({
                    "text": text,
                    "source": "uploads",
                    "filename": file.filename,
                    "page_number": i + 1,
                })
        doc.close()

        raw_chunks = chunk_documents(pages)
        new_chunks = attach_metadata(raw_chunks)

        existing = len(pipeline._retriever._bm25._chunks)
        for i, c in enumerate(new_chunks):
            c["chunk_index"] = existing + i

        index_dir = _PROJECT_ROOT / Path(os.getenv("INDEX_DIR", "indexes"))
        pipeline._retriever._bm25.extend(new_chunks, index_dir / "bm25_index.pkl")
        pipeline._retriever._vector.extend(new_chunks, index_dir / "faiss_index")
        return len(new_chunks)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        n_chunks = await loop.run_in_executor(pool, _ingest)

    _log.info("Incremental index: added %d chunks from '%s'", n_chunks, file.filename)
    return UploadResponse(status="ok", filename=file.filename, chunks_added=n_chunks)


@router.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str, pipeline=Depends(_get_pipeline)):
    """Remove a PDF from disk and purge all its chunks from both live indexes."""
    data_dir = _PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    found = next((p for p in data_dir.rglob("*.pdf") if p.name == filename), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    found.unlink()
    _log.info("Deleted file → %s", found)

    def _remove() -> int:
        index_dir = _PROJECT_ROOT / Path(os.getenv("INDEX_DIR", "indexes"))
        pipeline._retriever._bm25.remove_by_filename(filename, index_dir / "bm25_index.pkl")
        return pipeline._retriever._vector.remove_by_filename(filename, index_dir / "faiss_index")

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        chunks_removed = await loop.run_in_executor(pool, _remove)

    _log.info("Removed %d chunks for '%s' from live indexes", chunks_removed, filename)
    return DeleteResponse(status="ok", filename=filename, chunks_removed=chunks_removed)


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return server liveness and pipeline readiness."""
    pipeline = _state.get("pipeline")
    return HealthResponse(
        status="healthy" if pipeline is not None else "loading",
        indexes_loaded=pipeline is not None,
        model=_state.get("model", "unknown"),
    )
