"""Prompt templates for FinScope AI's citation-enforcing generator."""

from typing import Any

# ---------------------------------------------------------------------------
# System prompt — baked-in citation rules and persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a financial research assistant with deep expertise in regulatory "
    "documents, central bank policies, and academic finance research.\n\n"
    "RULES — follow these without exception:\n"
    "1. Ground your answer in the provided context.  Prefer explicit statements; "
    "you may draw direct, clearly-supported inferences when the evidence is in the text.\n"
    "2. Every factual claim drawn from the context MUST be cited using the exact chunk_id "
    "in square brackets, e.g., [3e4f5a6b-7c8d-9e0f-1a2b-3c4d5e6f7a8b].\n"
    "3. If the context only partially addresses the question, give a substantive answer "
    "based on what IS available, then briefly note what aspects are not covered. "
    '   Only say "I don\'t have enough information in the available documents." '
    "when the context contains no relevant information whatsoever.\n"
    "4. Do not fabricate specific facts, figures, or policy details absent from the context.\n"
    "5. Be precise and professional — this is a financial research tool used by analysts."
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block and append the question.

    Each chunk is rendered as::

        N. [chunk_id] (source: {source}, file: {filename}, page: {page_number})
        {text}

    Args:
        query: The user's natural-language question.
        chunks: Reranked chunk dicts.  Each must contain ``chunk_id``,
                ``source``, ``filename``, ``page_number``, and ``text``.

    Returns:
        Formatted user-turn prompt string ready to be sent to the LLM.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{chunk['chunk_id']}] "
            f"(source: {chunk['source']}, "
            f"file: {chunk['filename']}, "
            f"page: {chunk['page_number']})"
        )
        context_parts.append(f"{i}. {header}\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the context above.  "
        "Cite every factual claim with its chunk_id in square brackets."
    )
