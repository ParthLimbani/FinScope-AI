"""End-to-end RAG pipeline: retrieve → rerank → generate → validate citations."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.generator.citation_validator import validate_citations
from src.generator.llm import GroqClient, LLMError
from src.generator.prompt import SYSTEM_PROMPT, build_prompt
from src.reranker.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever

load_dotenv()

_log = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full FinScope AI retrieval-augmented generation pipeline.

    Initialisation loads all heavy resources once (indexes, embedding model,
    cross-encoder, Groq client).  After that, :meth:`query` is a pure async
    call with no disk I/O.

    Pipeline stages
    ---------------
    1. **Hybrid retrieval** — BM25 + FAISS fused with RRF (top-20 candidates).
    2. **Cross-encoder reranking** — re-scores the 20 candidates, keeps top-5.
    3. **Prompt construction** — formats chunks into a numbered context block.
    4. **LLM generation** — sends context + question to Groq LLaMA 3.1 70B.
    5. **Citation validation** — checks every [chunk_id] in the answer against
       the retrieved set; flags hallucinated references.
    """

    def __init__(self) -> None:
        """
        Load indexes, models, and API clients.

        Reads paths and model names from environment variables (with sensible
        defaults).  See ``.env.example`` for the full list of variables.

        Raises:
            FileNotFoundError: If the BM25 or FAISS index files do not exist.
            ValueError: If ``GROQ_API_KEY`` is not set.
        """
        index_dir = Path(os.getenv("INDEX_DIR", "indexes"))
        embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

        _log.info("Loading BM25 index from %s ...", index_dir / "bm25_index.pkl")
        bm25 = BM25Retriever.load(index_dir / "bm25_index.pkl")

        _log.info("Loading FAISS index from %s (embed=%s) ...", index_dir / "faiss_index", embed_model)
        vector = VectorRetriever.load(index_dir / "faiss_index", embed_model)

        self._retriever = HybridRetriever(bm25, vector)
        self._reranker = CrossEncoderReranker()
        self._llm = GroqClient()

        _log.info("Pipeline ready — model=%s  reranker=%s", self._llm.model, self._reranker.model)

    async def query(
        self,
        question: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline for a natural-language question.

        Args:
            question: User's question in plain English.
            top_k_retrieve: Number of candidates from hybrid retrieval (default 20).
            top_k_rerank: Number of chunks kept after cross-encoder reranking
                          (default 5).  These are passed to the LLM.

        Returns:
            Dict with the following structure::

                {
                    "answer": str,
                    "citations": [
                        {
                            "chunk_id": str,
                            "source": str,        # arxiv | rbi | bis | sebi
                            "filename": str,
                            "page_number": int,
                            "snippet": str,       # first 200 chars of chunk text
                            "ce_score": float,
                            "rrf_score": float,
                        },
                        ...
                    ],
                    "citation_validation": {
                        "all_valid": bool,
                        "invalid": list[str],        # hallucinated chunk_ids
                        "uncited_chunks": list[str], # retrieved but never cited
                    },
                    "metadata": {
                        "retrieved": int,
                        "reranked_to": int,
                        "latency_ms": float,
                    },
                }

        Raises:
            LLMError: If the Groq API call fails (rate limit, connection error, etc.).
        """
        t0 = time.perf_counter()
        q_preview = question[:80] + ("…" if len(question) > 80 else "")
        _log.info('Query: "%s"', q_preview)

        # Stage 1 — Hybrid retrieval (BM25 + FAISS → RRF)
        t1 = time.perf_counter()
        hybrid_results = self._retriever.retrieve(question, top_k=top_k_retrieve)
        ms1 = round((time.perf_counter() - t1) * 1000, 1)
        _log.info(
            "[1/5] Hybrid retrieval — %d candidates in %.1f ms",
            len(hybrid_results), ms1,
        )

        # Stage 2 — Cross-encoder reranking
        t2 = time.perf_counter()
        reranked = self._reranker.rerank(question, hybrid_results, top_k=top_k_rerank)
        ms2 = round((time.perf_counter() - t2) * 1000, 1)
        top_ce = reranked[0]["ce_score"] if reranked else float("nan")
        sources = ", ".join(dict.fromkeys(c.get("source", "?") for c in reranked))
        _log.info(
            "[2/5] Reranking — kept %d/%d  top_ce=%.3f  sources=[%s]  %.1f ms",
            len(reranked), len(hybrid_results), top_ce, sources, ms2,
        )

        # Stage 3 — Build LLM prompt
        t3 = time.perf_counter()
        prompt = build_prompt(question, reranked)
        ms3 = round((time.perf_counter() - t3) * 1000, 1)
        _log.info(
            "[3/5] Prompt built — %d chunks, %d chars  %.1f ms",
            len(reranked), len(prompt), ms3,
        )

        # Stage 4 — LLM generation (may raise LLMError)
        t4 = time.perf_counter()
        answer = await self._llm.generate(prompt, SYSTEM_PROMPT, history=history or [])
        ms4 = round((time.perf_counter() - t4) * 1000, 1)
        _log.info(
            "[4/5] LLM generation — model=%s  answer=%d chars  %.1f ms",
            self._llm.model, len(answer), ms4,
        )

        # Stage 5 — Citation validation
        t5 = time.perf_counter()
        validation = validate_citations(answer, reranked)
        ms5 = round((time.perf_counter() - t5) * 1000, 1)
        cited_count = len(reranked) - len(validation.get("uncited_chunks", []))
        _log.info(
            "[5/5] Citation validation — all_valid=%s  cited=%d/%d  invalid=%d  %.1f ms",
            validation["all_valid"], cited_count, len(reranked),
            len(validation.get("invalid", [])), ms5,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _log.info("Query complete — total %.1f ms", latency_ms)

        citations: list[dict[str, Any]] = [
            {
                "chunk_id": c["chunk_id"],
                "source": c.get("source", ""),
                "filename": c.get("filename", ""),
                "page_number": c.get("page_number", 0),
                "snippet": c["text"][:200],
                "ce_score": c.get("ce_score", 0.0),
                "rrf_score": c.get("rrf_score", 0.0),
            }
            for c in reranked
        ]

        return {
            "answer": answer,
            "citations": citations,
            "citation_validation": {
                "all_valid": validation["all_valid"],
                "invalid": validation["invalid"],
                "uncited_chunks": validation["uncited_chunks"],
            },
            "metadata": {
                "retrieved": len(hybrid_results),
                "reranked_to": len(reranked),
                "latency_ms": latency_ms,
            },
            # Full chunk texts passed to the LLM — used by the RAGAS eval pipeline.
            "contexts": [c["text"] for c in reranked],
        }


# ---------------------------------------------------------------------------
# __main__ — run a sample query and pretty-print the result
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    async def _main() -> None:
        print("Initializing RAG pipeline (loading indexes and models)...\n")
        pipeline = RAGPipeline()

        question = "What is RBI's approach to digital currency and CBDC?"
        print(f'Query: "{question}"\n')

        try:
            result = await pipeline.query(question, top_k_retrieve=20, top_k_rerank=5)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except LLMError as exc:
            print(f"\n[LLM ERROR] {exc}")
        except Exception as exc:
            print(f"\n[ERROR] {type(exc).__name__}: {exc}")

    asyncio.run(_main())
