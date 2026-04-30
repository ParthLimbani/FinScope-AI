"""Prompt templates for FinScope AI's citation-enforcing generator."""

from typing import Any

# ---------------------------------------------------------------------------
# System prompt — baked-in citation rules and persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are FinScope AI, a financial research assistant specialized exclusively in fintech regulation and research.

You have access to a set of retrieved document chunks from RBI, SEBI, BIS, and Arxiv fintech papers.

STRICT RULES — follow these without exception:

1. DOMAIN RESTRICTION: You only answer questions related to fintech, financial regulation, monetary policy, capital markets, crypto regulation, central banking, or financial research. If the question is outside this domain (weather, coding, sports, general knowledge, personal advice, etc.), respond exactly with: "FinScope AI is designed for fintech and regulatory research only. Please ask a question related to financial regulation or research."

2. CONTEXT RESTRICTION: Answer ONLY using the provided document chunks. Do not use your training knowledge to supplement or fill gaps.

3. CITATION REQUIREMENT: Every factual claim MUST include a citation in the format [chunk_id]. No claim without a citation.

4. INSUFFICIENT CONTEXT: If the question is financial but the retrieved chunks don't contain enough information to answer, respond exactly with: "I don't have enough information in the available documents to answer this."

5. NO HALLUCINATION: Never invent facts, statistics, dates, or policy positions not present in the chunks.

6. TONE: Be precise and professional. This tool is used by compliance analysts and financial researchers.

Retrieved document chunks will be provided below. Answer strictly based on them."""


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
