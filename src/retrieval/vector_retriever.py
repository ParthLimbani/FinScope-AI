"""Dense FAISS retriever: embeds via HF Inference API, indexes with FAISS."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)

_FAISS_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"


class VectorRetriever:
    """
    Dense semantic retriever backed by a FAISS IndexFlatIP index.

    Query-time embeddings are always fetched from the HuggingFace Inference
    API (no local model loaded).  Index build uses a local SentenceTransformer
    — that is an offline, one-off operation and is not part of the server path.

    Embeddings are L2-normalised so inner-product search equals cosine similarity.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict[str, Any]],
        model_name: str,
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        index_dir: Path,
        model_name: str,
    ) -> "VectorRetriever":
        """Offline index build — uses local SentenceTransformer (not the server path)."""
        from sentence_transformers import SentenceTransformer

        _log.info("Loading embedding model '%s' for index build...", model_name)
        model = SentenceTransformer(model_name)

        texts = [chunk["text"] for chunk in chunks]
        _log.info("Encoding %d chunks...", len(texts))
        embeddings: np.ndarray = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        _log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / _FAISS_FILE))
        with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False)

        _log.info("Index saved → %s", index_dir)
        return cls(index, chunks, model_name)

    @classmethod
    def load(cls, index_dir: Path, model_name: str) -> "VectorRetriever":
        """Load a persisted FAISS index.  No embedding model is loaded into memory."""
        _log.info("Loading FAISS index from %s", index_dir)
        index = faiss.read_index(str(index_dir / _FAISS_FILE))
        with open(index_dir / _CHUNKS_FILE, encoding="utf-8") as fh:
            chunks: list[dict[str, Any]] = json.load(fh)
        _log.info("FAISS: %d vectors, %d chunks", index.ntotal, len(chunks))
        return cls(index, chunks, model_name)

    @classmethod
    def load_or_build(
        cls,
        chunks: list[dict[str, Any]],
        index_dir: Path,
        model_name: str,
    ) -> "VectorRetriever":
        index_ready = (index_dir / _FAISS_FILE).exists() and (index_dir / _CHUNKS_FILE).exists()
        if index_ready:
            return cls.load(index_dir, model_name)
        return cls.build(chunks, index_dir, model_name)

    # ------------------------------------------------------------------
    # Embedding helpers (HF Inference API only)
    # ------------------------------------------------------------------

    def _embed_batch_api(self, texts: list[str]) -> np.ndarray:
        """Embed multiple strings in one HF Inference API call."""
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HF_TOKEN")
        hf_model = (
            self._model_name if "/" in self._model_name
            else f"sentence-transformers/{self._model_name}"
        )
        client = InferenceClient(token=hf_token)
        result = client.feature_extraction(texts, model=hf_model)
        arr = np.array(result, dtype=np.float32)
        # Some models return [batch, seq, dim] — pool to [batch, dim]
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        return arr.reshape(len(texts), -1)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def extend(self, new_chunks: list[dict[str, Any]], index_dir: Path) -> None:
        """Embed new chunks via API, add to FAISS index, and persist."""
        texts = [c["text"] for c in new_chunks]
        embeddings = self._embed_batch_api(texts)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._chunks.extend(new_chunks)

        faiss.write_index(self._index, str(index_dir / _FAISS_FILE))
        with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._chunks, fh, ensure_ascii=False)
        _log.info("FAISS: extended to %d vectors → %s", self._index.ntotal, index_dir)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-K semantically similar chunks for a query."""
        k = top_k if top_k is not None else int(os.getenv("TOP_K_RETRIEVE", "20"))

        query_emb = self._embed_batch_api([query])
        faiss.normalize_L2(query_emb)

        scores, indices = self._index.search(query_emb, k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
