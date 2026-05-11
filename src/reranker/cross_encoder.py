"""Cross-encoder reranker via HuggingFace Inference API — no local model."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _parse_scores(raw: Any, n: int) -> list[float]:
    """
    Normalise the three response shapes the HF Inference API can return for
    a cross-encoder / text-classification model.

    Shapes observed:
      A) [[{"label":"LABEL_0","score":0.12},{"label":"LABEL_1","score":0.88}], ...]
         — one list of label dicts per pair; take the highest score per row.
      B) [{"label":"LABEL_1","score":0.88}, ...]
         — single dict per pair (binary classification, one result per pair).
      C) [0.88, 0.12, ...]
         — flat list of floats.
    """
    if not raw:
        return [0.0] * n

    # Shape C — flat floats
    if isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]

    # Shape B — single dict per pair
    if isinstance(raw[0], dict):
        return [float(x.get("score", 0.0)) for x in raw]

    # Shape A — list of label dicts per pair; take max score per row
    scores = []
    for row in raw:
        if isinstance(row, list):
            scores.append(max(float(d.get("score", 0.0)) for d in row))
        elif isinstance(row, dict):
            scores.append(float(row.get("score", 0.0)))
        else:
            scores.append(float(row))
    return scores


class CrossEncoderReranker:
    """
    Second-stage reranker that scores (query, passage) pairs via the
    HuggingFace Inference API.  No local model is loaded.

    Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (default).
    API fallback: if the HF call fails, chunks are returned with ce_score=0
    sorted by their existing rrf_score so the pipeline never hard-crashes.
    """

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or os.getenv("RERANKER_MODEL", _DEFAULT_MODEL)
        if "/" not in name:
            name = f"cross-encoder/{name}"
        self._model_name = name
        self.model = name           # exposed for rag_pipeline logging
        _log.info("[Reranker] API mode — model=%s", self._model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score every (query, chunk_text) pair and return the top-K by ce_score."""
        if not chunks:
            return []

        scores = self._score_api(query, [c["text"] for c in chunks])

        scored: list[dict[str, Any]] = []
        for chunk, score in zip(chunks, scores):
            result = dict(chunk)
            result["ce_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda x: x["ce_score"], reverse=True)
        return scored[:top_k]

    def _score_api(self, query: str, passages: list[str]) -> list[float]:
        """Call HF Inference API and return one relevance score per passage."""
        import requests

        hf_token = os.getenv("HF_TOKEN")
        pairs = [[query, p] for p in passages]
        url = f"https://api-inference.huggingface.co/models/{self._model_name}"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        try:
            resp = requests.post(url, headers=headers, json={"inputs": pairs}, timeout=30)
            resp.raise_for_status()
            return _parse_scores(resp.json(), len(passages))
        except Exception as exc:
            _log.warning(
                "[Reranker] HF API error (%s) — falling back to rrf_score ordering", exc
            )
            return [0.0] * len(passages)
