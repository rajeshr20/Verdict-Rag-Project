from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon successfully uploading and ingesting a document."""
    doc_id: str
    filename: str
    chunk_count: int
    message: str = "Document successfully ingested."


class DocumentInfo(BaseModel):
    """Metadata summary of an ingested document."""
    doc_id: str
    filename: str
    doc_title: str
    version: str
    effective_date: str
    chunk_count: int


class AskRequest(BaseModel):
    """Payload for POST /ask."""
    question: str = Field(
        ...,
        min_length=1,
        description="The question to be answered from retrieved documents.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional override for number of chunks to retrieve.",
    )


class ChunkEvidence(BaseModel):
    """Evidence excerpt directly tied to a verified claim."""
    document: str
    section: str
    version: str
    excerpt: str
    chunk_id: str
    verified: bool = True


class ConflictEvidence(BaseModel):
    """Report of a detected contradiction across documents."""
    topic: str
    description: str
    newer_document: Optional[str] = None
    newer_chunk_id: Optional[str] = None
    resolution_note: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)


class Diagnostics(BaseModel):
    """Pipeline performance and verification telemetry."""
    retrieved: int
    claims_total: int
    claims_verified: int
    claims_dropped: int
    gated: bool
    latency_ms: int


class AskResponse(BaseModel):
    """Final answer and verification response from POST /ask."""
    question: str
    answerable: bool
    answer: str
    confidence: float
    confidence_label: str
    evidence: list[ChunkEvidence] = Field(default_factory=list)
    conflicts: list[ConflictEvidence] = Field(default_factory=list)
    diagnostics: Diagnostics


class HealthResponse(BaseModel):
    """System status and active providers."""
    status: str = "healthy"
    version: str = "1.0.0"
    llm_provider: str
    embedding_provider: str
    document_count: int
    chunk_count: int
