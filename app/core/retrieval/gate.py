from typing import Tuple
from app.core.retrieval.hybrid import RetrievedChunk


def evaluate_retrieval_gate(
    retrieved_chunks: list[RetrievedChunk],
    min_score: float = 0.01,
) -> Tuple[bool, str]:
    """Determine whether to abstain before LLM generation based on retrieval quality.

    If the collection is empty, no relevant chunks are found, or the top candidate's
    RRF score is below `min_score`, the gate triggers an early abstention.

    Args:
        retrieved_chunks: List of retrieved chunks ordered by descending RRF score.
        min_score: Minimum RRF score threshold required to proceed.

    Returns:
        A tuple of (is_gated: bool, reason: str).
        If is_gated is True, the pipeline must return answerable=False immediately.
    """
    if not retrieved_chunks:
        return True, "No documents were retrieved."

    top_chunk = retrieved_chunks[0]
    top_score = top_chunk.rrf_score

    if top_score < min_score:
        return True, f"Top retrieval RRF score ({top_score:.4f}) is below threshold ({min_score:.4f})."

    return False, ""
