"""Pydantic v2 request/response models for the FinScope AI API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of reranked chunks passed to the LLM")


class CitationItem(BaseModel):
    chunk_id: str
    source: str
    filename: str
    page_number: int
    snippet: str
    ce_score: float
    rrf_score: float


class CitationValidation(BaseModel):
    all_valid: bool
    invalid: list[str]
    uncited_chunks: list[str]


class QueryMetadata(BaseModel):
    retrieved: int
    reranked_to: int
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    citation_validation: CitationValidation
    metadata: QueryMetadata
    contexts: list[str] = Field(default_factory=list, description="Full chunk texts passed to the LLM (for evaluation)")


class IngestRequest(BaseModel):
    data_dir: str = Field("data/", description="Path to PDF source directory (relative to project root)")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    index_build_time_ms: float


class HealthResponse(BaseModel):
    status: str
    indexes_loaded: bool
    model: str
