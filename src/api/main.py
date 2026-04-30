"""FastAPI application entry point for FinScope AI."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server starts immediately; pipeline loads lazily on first request."""
    _log.info("FinScope AI — server starting (pipeline loads on first request).")
    yield
    from src.api.routes import _state
    _log.info("FinScope AI — shutting down.")
    _state["pipeline"] = None


app = FastAPI(
    title="FinScope AI",
    description=(
        "Production RAG system over 40 fintech regulatory documents "
        "(Arxiv, RBI, BIS, SEBI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials=True is a CSRF risk
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _log.info(
        "%s %s → %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    _log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred.", "status": 500},
    )


from src.api.routes import router  # noqa: E402
app.include_router(router, prefix="/api/v1")

# Serve the frontend — must be mounted AFTER /api/v1 routes.
# html=True makes FastAPI return index.html for any unmatched path (SPA mode).
_static = Path(__file__).resolve().parent.parent / "static"
_static.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
