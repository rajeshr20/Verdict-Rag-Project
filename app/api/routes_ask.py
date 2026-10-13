import logging
from fastapi import APIRouter, HTTPException, Request, status
from app.core.generation.llm import LLMError
from app.models.api_models import AskRequest, AskResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["ask"])


@router.post(
    "",
    response_model=AskResponse,
    summary="Ask a question against ingested documents with claim verification and conflict detection",
)
def ask_question(request: Request, payload: AskRequest) -> AskResponse:
    """Submit a question to the claim-verified RAG pipeline.

    - Retrieves hybrid candidates via Dense + BM25 RRF.
    - Gates out queries with insufficient retrieval scores.
    - Defends against prompt injection via XML passive data boundaries.
    - Verifies LLM claims deterministically against verbatim excerpts.
    - Strips ungrounded claims and surfaces document conflicts.
    """
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty or whitespace only.",
        )

    pipeline = request.app.state.pipeline

    try:
        response = pipeline.ask(
            question=clean_question,
            top_k=payload.top_k,
        )
        return response
    except LLMError as e:
        logger.error("LLM generation error during /ask: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation failed or timed out: {str(e)}",
        )
    except Exception as e:
        logger.error("Unexpected error during pipeline execution: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the question.",
        )
