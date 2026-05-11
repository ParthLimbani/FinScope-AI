"""
RAGAS evaluation pipeline for FinScope AI.

Run:
    python evaluation/eval_runner.py

Requires GROQ_API_KEY in the environment.
Exits with code 1 if faithfulness < FAITHFULNESS_THRESHOLD (.env default 0.80).
"""

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.80"))
_EVAL_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# LLM helper — LangchainLLMWrapper(ChatGroq) is the documented RAGAS 0.4.x approach
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Export it or add it to your .env file before running evals."
        )
    chat_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    return LangchainLLMWrapper(chat_llm)




# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

async def _run_evaluation() -> None:
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import ContextRecall, Faithfulness

    from src.pipeline.rag_pipeline import RAGPipeline

    # ── 1. Load Q&A dataset ───────────────────────────────────────────────
    # EVAL_QUESTIONS_FILE overrides the default (used by CI to load eval_questions.json).
    _qfile_env = os.getenv("EVAL_QUESTIONS_FILE")
    qa_path = (_PROJECT_ROOT / _qfile_env) if _qfile_env else (_EVAL_DIR / "golden_qa.json")
    with open(qa_path, encoding="utf-8") as fh:
        qa_pairs: list[dict] = json.load(fh)

    print(f"[Eval] Loaded {len(qa_pairs)} Q&A pairs from {qa_path}")

    # ── 2. Initialise pipeline ────────────────────────────────────────────
    print("[Eval] Initialising RAG pipeline...")
    pipeline = RAGPipeline()

    # ── 3. Run all queries ─────────────────────────────────────────────────
    # Token budget: top_k_rerank=3 and max_chars_per_chunk=800 cut prompt tokens
    # ~50 % vs defaults (5 chunks × full text).  This keeps the daily TPD limit
    # from being exhausted on a single CI run while preserving retrieval quality.
    print(f"[Eval] Running {len(qa_pairs)} pipeline queries...")
    t0 = time.perf_counter()
    pipeline_results: list[dict] = []
    for i, qa in enumerate(qa_pairs, 1):
        print(f"  [{i:02d}/{len(qa_pairs)}] {qa['question'][:65]}...")
        result = await pipeline.query(
            qa["question"],
            top_k_retrieve=10,
            top_k_rerank=3,
            max_chars_per_chunk=800,
        )
        pipeline_results.append(result)
    query_elapsed = round((time.perf_counter() - t0) * 1000)
    print(f"[Eval] Queries complete in {query_elapsed} ms")

    # ── 4. Build RAGAS dataset ─────────────────────────────────────────────
    # AnswerRelevancy requires RAGAS-native embeddings (not supported with our HF adapter).
    # CI gate only checks faithfulness; ContextRecall is LLM-only.
    print("[Eval] Configuring RAGAS metrics with Groq LLM...")
    ragas_llm = _build_ragas_llm()

    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    samples = [
        SingleTurnSample(
            user_input=qa["question"],
            response=result["answer"],
            # Prefer full chunk texts; fall back to 200-char snippets.
            retrieved_contexts=result.get(
                "contexts",
                [c["snippet"] for c in result["citations"]],
            ),
            reference=qa["ground_truth"],
        )
        for qa, result in zip(qa_pairs, pipeline_results)
    ]
    dataset = EvaluationDataset(samples=samples)

    # ── 5. Run RAGAS evaluation ────────────────────────────────────────────
    print("[Eval] Running RAGAS evaluation (may take several minutes)...")
    eval_result = ragas_evaluate(dataset, metrics=metrics, show_progress=True)

    # ── 6. Extract scores ──────────────────────────────────────────────────
    score_dicts: list[dict] = eval_result.scores

    def _safe_mean(key: str) -> float:
        vals = [s[key] for s in score_dicts if s.get(key) is not None]
        return statistics.mean(vals) if vals else 0.0

    mean_faith = _safe_mean("faithfulness")
    mean_rec   = _safe_mean("context_recall")

    per_question = [
        {
            "question":       qa["question"],
            "source":         qa.get("source", ""),
            "difficulty":     qa.get("difficulty", ""),
            "faithfulness":   scores.get("faithfulness"),
            "context_recall": scores.get("context_recall"),
        }
        for qa, scores in zip(qa_pairs, score_dicts)
    ]

    summary = {
        "faithfulness":     round(mean_faith, 4),
        "answer_relevancy": None,
        "context_recall":   round(mean_rec,   4),
        "per_question":     per_question,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    # ── 7. Save results ────────────────────────────────────────────────────
    results_path = _EVAL_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[Eval] Results saved → {results_path}")

    # ── 8. Print summary table ─────────────────────────────────────────────
    _pass = lambda v, t: "PASS ✓" if v >= t else "FAIL ✗"
    print()
    print("=" * 60)
    print("  FinScope AI — RAGAS Evaluation Summary")
    print(f"  CI gate: faithfulness ≥ {FAITHFULNESS_THRESHOLD}")
    print("=" * 60)
    print(f"  {'Metric':<24} {'Score':>8}   Status")
    print("  " + "-" * 56)
    print(f"  {'Faithfulness':<24} {mean_faith:>8.4f}   {_pass(mean_faith, FAITHFULNESS_THRESHOLD)}")
    print(f"  {'Context Recall':<24} {mean_rec:>8.4f}   {_pass(mean_rec, 0.70)}")
    print("=" * 60)
    print()

    # ── 9. CI gate ─────────────────────────────────────────────────────────
    if mean_faith < FAITHFULNESS_THRESHOLD:
        print(
            f"[GATE FAIL] faithfulness {mean_faith:.4f} < {FAITHFULNESS_THRESHOLD} — "
            "PR blocked."
        )
        sys.exit(1)

    print("[GATE PASS] All thresholds met.")


if __name__ == "__main__":
    asyncio.run(_run_evaluation())
