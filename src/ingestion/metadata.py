"""Metadata tagger: attaches chunk IDs and sequential indexes to every chunk."""

import uuid
from typing import Any, TypedDict


class Chunk(TypedDict):
    """A fully-tagged text chunk ready for ingestion into retrieval indexes."""

    text: str
    source: str       # data subfolder name (arxiv | rbi | bis | sebi)
    filename: str     # original PDF filename
    page_number: int  # 1-indexed page the chunk came from
    chunk_id: str     # UUID4 string — globally unique across all chunks
    chunk_index: int  # sequential position across all chunks (0-based)


def attach_metadata(raw_chunks: list[dict[str, Any]]) -> list[Chunk]:
    """
    Assign a unique chunk_id (UUID4) and a sequential chunk_index to every chunk.

    Args:
        raw_chunks: List of dicts containing at least ``text``, ``source``,
                    ``filename``, and ``page_number`` keys.

    Returns:
        List of fully-tagged :class:`Chunk` TypedDicts.
    """
    tagged: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        tagged.append(
            Chunk(
                text=raw["text"],
                source=raw.get("source", "unknown"),
                filename=raw.get("filename", ""),
                page_number=raw.get("page_number", 0),
                chunk_id=str(uuid.uuid4()),
                chunk_index=idx,
            )
        )
    return tagged
