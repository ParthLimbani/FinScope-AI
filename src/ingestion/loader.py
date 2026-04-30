"""PDF loader: extracts text page-by-page from all PDFs under the data directory."""

import os
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()


class PageDocument(TypedDict):
    """A single page extracted from a PDF."""

    text: str
    source: str       # data subfolder name, normalized to lowercase
    filename: str     # PDF filename (no path)
    page_number: int  # 1-indexed


def load_pdfs(data_dir: str | None = None) -> list[PageDocument]:
    """
    Recursively load all PDFs from subfolders of data_dir.

    Each page becomes a separate PageDocument. Pages with fewer than 10
    characters are skipped (blank or image-only pages).

    Args:
        data_dir: Root directory containing PDF subfolders. Defaults to the
                  DATA_DIR environment variable, or ``data/``.

    Returns:
        List of PageDocument dicts, one per non-empty page.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no PDF files are found.
    """
    root = Path(data_dir or os.getenv("DATA_DIR", "data"))
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root.resolve()}")

    pdf_paths = sorted(root.rglob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found under {root.resolve()}")

    print(f"[Loader] Found {len(pdf_paths)} PDF files in {root.resolve()}")
    documents: list[PageDocument] = []

    for pdf_path in pdf_paths:
        # Source is the immediate subdirectory name under data_dir, lowercased.
        relative = pdf_path.relative_to(root)
        source = relative.parts[0].lower() if len(relative.parts) > 1 else "unknown"
        filename = pdf_path.name

        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            extracted = 0
            for page_idx in range(page_count):
                text = doc[page_idx].get_text().strip()
                if len(text) < 10:
                    continue  # skip blank / image-only pages
                documents.append(
                    PageDocument(
                        text=text,
                        source=source,
                        filename=filename,
                        page_number=page_idx + 1,
                    )
                )
                extracted += 1
            doc.close()
            print(f"  [Loader] {filename}: {extracted}/{page_count} pages extracted")
        except Exception as exc:  # noqa: BLE001
            print(f"  [Loader] WARNING — skipping {filename}: {exc}")

    print(f"[Loader] Total pages extracted: {len(documents)}")
    return documents


if __name__ == "__main__":
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.metadata import attach_metadata
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.vector_retriever import VectorRetriever

    _index_dir = Path(os.getenv("INDEX_DIR", "indexes"))
    _index_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 1: Loading PDFs ===")
    pages = load_pdfs()

    print("\n=== Phase 2: Chunking ===")
    raw_chunks = chunk_documents(pages)

    print("\n=== Phase 3: Attaching Metadata ===")
    chunks = attach_metadata(raw_chunks)
    print(f"[Metadata] Total chunks ready for indexing: {len(chunks)}")

    print("\n=== Phase 4: Building BM25 Index ===")
    BM25Retriever.load_or_build(chunks, _index_dir / "bm25_index.pkl")

    print("\n=== Phase 5: Building FAISS Index ===")
    VectorRetriever.load_or_build(
        chunks,
        _index_dir / "faiss_index",
        os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
    )

    print("\n=== Ingestion complete ===")
