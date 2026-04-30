"""Text chunker: splits page-level documents into fixed-size overlapping chunks."""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split each document's text into overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter with character-level sizing.
    All metadata keys from the source document (source, filename, page_number)
    are forwarded to every chunk produced from it.

    Args:
        documents: List of page-level dicts, each containing at least a ``text`` key.
        chunk_size: Maximum characters per chunk. Defaults to the CHUNK_SIZE
                    environment variable, or ``512``.
        chunk_overlap: Overlap between consecutive chunks in characters. Defaults to
                       the CHUNK_OVERLAP environment variable, or ``64``.

    Returns:
        List of chunk dicts with the same metadata keys as the input documents.
    """
    size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", "512"))
    overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "64"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=False,
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            chunks.append(
                {
                    "text": piece,
                    "source": doc.get("source", "unknown"),
                    "filename": doc.get("filename", ""),
                    "page_number": doc.get("page_number", 0),
                }
            )

    print(
        f"[Chunker] {len(documents)} pages → {len(chunks)} chunks "
        f"(chunk_size={size}, overlap={overlap})"
    )
    return chunks
