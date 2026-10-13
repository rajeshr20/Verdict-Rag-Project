import time
from typing import Optional
from app.core.generation.llm import LLM
from app.core.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.core.retrieval.gate import evaluate_retrieval_gate
from app.core.retrieval.hybrid import HybridRetriever, RetrievedChunk
from app.core.verification.conflicts import resolve_conflicts
from app.core.verification.verifier import verify_claims
from app.models.api_models import (
    AskResponse,
    ChunkEvidence,
    ConflictEvidence,
    Diagnostics,
)


class RAGPipeline:
    """Master pipeline orchestrating retrieval, gating, LLM generation, deterministic claim verification, and conflict detection."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        default_top_k: int = 6,
        min_score: float = 0.01,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.default_top_k = default_top_k
        self.min_score = min_score

    def ask(self, question: str, top_k: Optional[int] = None) -> AskResponse:
        """Execute the end-to-end claim-verified RAG workflow.

        Steps:
        1. Retrieve top-K chunks with hybrid dense + BM25 + RRF.
        2. Gate: abstain immediately if top retrieval score < min_score.
        3. Build XML-delimited prompt protecting against prompt injection.
        4. Call LLM for structured output (JSON).
        5. Verify quotes deterministically against source chunk text.
        6. Rebuild final answer strictly from verified claims.
        7. Analyze conflicts and attach chronological metadata.
        8. Compute confidence and compile diagnostics.
        """
        start_time = time.perf_counter()
        k = top_k if (top_k is not None and top_k > 0) else self.default_top_k

        # 1. Retrieval
        retrieved_chunks = self.retriever.retrieve(question, top_k=k)
        retrieved_count = len(retrieved_chunks)

        # 2. Gate check (abstain before LLM)
        is_gated, gate_reason = evaluate_retrieval_gate(retrieved_chunks, min_score=self.min_score)
        if is_gated:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return AskResponse(
                question=question,
                answerable=False,
                answer="The provided documents do not contain this information.",
                confidence=0.0,
                confidence_label="low",
                evidence=[],
                conflicts=[],
                diagnostics=Diagnostics(
                    retrieved=retrieved_count,
                    claims_total=0,
                    claims_verified=0,
                    claims_dropped=0,
                    gated=True,
                    latency_ms=elapsed_ms,
                ),
            )

        # Build mapping from prompt IDs ('c1', 'c2', ...) to RetrievedChunk and raw texts
        prompt_id_to_chunk: dict[str, RetrievedChunk] = {}
        prompt_id_to_text: dict[str, str] = {}
        for i, item in enumerate(retrieved_chunks, start=1):
            pid = f"c{i}"
            prompt_id_to_chunk[pid] = item
            prompt_id_to_text[pid] = item.chunk.text
            # Also allow fallback to original chunk_id if LLM references it directly
            prompt_id_to_chunk[item.chunk.chunk_id] = item
            prompt_id_to_text[item.chunk.chunk_id] = item.chunk.text

        # 3. Build Prompt with XML chunk delimiters
        user_prompt = build_user_prompt(question, retrieved_chunks)

        # 4. Generate structured response via LLM
        llm_response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # If LLM explicitly reported not answerable or emitted no claims
        if not llm_response.answerable or not llm_response.claims:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return AskResponse(
                question=question,
                answerable=False,
                answer="The provided documents do not contain this information.",
                confidence=0.0,
                confidence_label="low",
                evidence=[],
                conflicts=[],
                diagnostics=Diagnostics(
                    retrieved=retrieved_count,
                    claims_total=len(llm_response.claims),
                    claims_verified=0,
                    claims_dropped=len(llm_response.claims),
                    gated=False,
                    latency_ms=elapsed_ms,
                ),
            )

        # 5. Deterministic Quote Verification
        v_result = verify_claims(llm_response.claims, prompt_id_to_text)

        # 6. If no verified claims survived -> abstain
        if not v_result.is_answerable or v_result.claims_verified == 0:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return AskResponse(
                question=question,
                answerable=False,
                answer="The provided documents do not contain this information.",
                confidence=0.0,
                confidence_label="low",
                evidence=[],
                conflicts=[],
                diagnostics=Diagnostics(
                    retrieved=retrieved_count,
                    claims_total=v_result.claims_total,
                    claims_verified=0,
                    claims_dropped=v_result.claims_dropped,
                    gated=False,
                    latency_ms=elapsed_ms,
                ),
            )

        # 7. Process Conflicts
        resolved_conflicts = resolve_conflicts(llm_response.conflicts, prompt_id_to_chunk)
        conflict_evidences: list[ConflictEvidence] = [
            ConflictEvidence(
                topic=rc.topic,
                description=rc.description,
                newer_document=rc.newer_document,
                newer_chunk_id=rc.newer_chunk_id,
                resolution_note=rc.resolution_note,
                sources=[
                    {
                        "chunk_id": s.chunk_id,
                        "document": s.document,
                        "version": s.version,
                        "effective_date": s.effective_date,
                        "section": s.section,
                        "excerpt": s.excerpt,
                    }
                    for s in rc.sources
                ],
            )
            for rc in resolved_conflicts
        ]

        # 8. Compile Evidence from verified claims
        evidence_list: list[ChunkEvidence] = []
        seen_evidence: set[tuple[str, str]] = set()

        for claim in v_result.verified_claims:
            chunk_item = prompt_id_to_chunk.get(claim.chunk_id)
            if not chunk_item:
                continue

            chunk = chunk_item.chunk
            doc_name = str(chunk.metadata.get("filename") or chunk.doc_id)
            pair_key = (doc_name, claim.quote)

            if pair_key not in seen_evidence:
                seen_evidence.add(pair_key)
                evidence_list.append(
                    ChunkEvidence(
                        document=doc_name,
                        section=chunk.section,
                        version=str(chunk.metadata.get("version", "1.0")),
                        excerpt=claim.quote,
                        chunk_id=claim.chunk_id,
                        verified=True,
                    )
                )

        # 9. Compute Confidence & Confidence Label
        # Blend claim verification ratio (70%) and top retrieval strength (30%)
        verif_ratio = v_result.claims_verified / max(1, v_result.claims_total)
        top_rrf = retrieved_chunks[0].rrf_score if retrieved_chunks else 0.0
        retrieval_factor = min(1.0, top_rrf / 0.025)

        raw_conf = (0.7 * verif_ratio) + (0.3 * retrieval_factor)
        confidence = round(float(max(0.0, min(1.0, raw_conf))), 2)

        if confidence >= 0.75:
            conf_label = "high"
        elif confidence >= 0.45:
            conf_label = "medium"
        else:
            conf_label = "low"

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return AskResponse(
            question=question,
            answerable=True,
            answer=v_result.rebuilt_answer,
            confidence=confidence,
            confidence_label=conf_label,
            evidence=evidence_list,
            conflicts=conflict_evidences,
            diagnostics=Diagnostics(
                retrieved=retrieved_count,
                claims_total=v_result.claims_total,
                claims_verified=v_result.claims_verified,
                claims_dropped=v_result.claims_dropped,
                gated=False,
                latency_ms=elapsed_ms,
            ),
        )
