
"""Dense FAISS retriever: embeds chunks with sentence-transformers, indexes with FAISS."""

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_FAISS_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"


class VectorRetriever:
    """
    Dense semantic retriever backed by a FAISS IndexFlatIP index.

    Embeddings are L2-normalised before insertion so inner-product search is
    equivalent to cosine similarity — the standard metric for sentence-transformers
    models.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict[str, Any]],
        model: SentenceTransformer,
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._model = model

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
        """
        Encode all chunk texts, build a FAISS IndexFlatIP index, and persist.

        Embeddings are float32 and L2-normalised before adding to the index.
        Both the FAISS index and the chunk metadata are written to index_dir.

        Args:
            chunks: List of chunk dicts — each must contain a ``text`` key.
            index_dir: Directory where ``index.faiss`` and ``chunks.json`` are written.
            model_name: sentence-transformers model identifier (e.g. ``all-MiniLM-L6-v2``).

        Returns:
            Initialized :class:`VectorRetriever` instance.
        """
        print(f"[FAISS] Loading embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)

        texts = [chunk["text"] for chunk in chunks]
        print(f"[FAISS] Encoding {len(texts)} chunks (this may take a moment)...")
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
        print(f"[FAISS] Index built: {index.ntotal} vectors, dim={dim}")

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / _FAISS_FILE))
        with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False)

        print(f"[FAISS] Index and chunks saved → {index_dir}")
        return cls(index, chunks, model)

    @classmethod
    def load(cls, index_dir: Path, model_name: str) -> "VectorRetriever":
        """
        Load a previously built FAISS index and its associated chunk metadata.

        Args:
            index_dir: Directory containing ``index.faiss`` and ``chunks.json``.
            model_name: sentence-transformers model identifier.

        Returns:
            Initialized :class:`VectorRetriever` instance.
        """
        print(f"[FAISS] Loading existing index from {index_dir}")
        index = faiss.read_index(str(index_dir / _FAISS_FILE))
        with open(index_dir / _CHUNKS_FILE, encoding="utf-8") as fh:
            chunks: list[dict[str, Any]] = json.load(fh)

        print(f"[FAISS] Loading embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)
        print(f"[FAISS] Loaded {index.ntotal} vectors, {len(chunks)} chunks")
        return cls(index, chunks, model)

    @classmethod
    def load_or_build(
        cls,
        chunks: list[dict[str, Any]],
        index_dir: Path,
        model_name: str,
    ) -> "VectorRetriever":
        """
        Load if a valid index exists on disk, otherwise build and persist it.

        Validity check: both ``index.faiss`` and ``chunks.json`` must be present.

        Args:
            chunks: Chunk list (only used when building).
            index_dir: Directory to load from or build into.
            model_name: sentence-transformers model identifier.

        Returns:
            Initialized :class:`VectorRetriever` instance.
        """
        index_ready = (index_dir / _FAISS_FILE).exists() and (index_dir / _CHUNKS_FILE).exists()
        if index_ready:
            return cls.load(index_dir, model_name)
        return cls.build(chunks, index_dir, model_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Return the top-K most semantically similar chunks for a query.

        The query is encoded and L2-normalised identically to the indexed
        embeddings so the inner-product scores reflect cosine similarity.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return. Defaults to the TOP_K_RETRIEVE
                   environment variable, or ``20``.

        Returns:
            List of chunk dicts (with an added ``score`` key) ordered by
            descending cosine similarity.
        """
        k = top_k if top_k is not None else int(os.getenv("TOP_K_RETRIEVE", "20"))
        query_emb: np.ndarray = self._model.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_emb)

        scores, indices = self._index.search(query_emb, k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when the index has fewer vectors than k
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
