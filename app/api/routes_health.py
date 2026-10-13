from fastapi import APIRouter, Request
from app.models.api_models import HealthResponse


router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Check system health, active providers, and document counts",
)
def get_health(request: Request) -> HealthResponse:
    """Return runtime system health and operational metrics."""
    settings = request.app.state.settings
    vector_store = request.app.state.vector_store

    summaries = vector_store.get_document_summary()
    total_chunks = vector_store.count()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=settings.LLM_PROVIDER,
        embedding_provider=settings.EMBEDDING_MODEL,
        document_count=len(summaries),
        chunk_count=total_chunks,
    )
