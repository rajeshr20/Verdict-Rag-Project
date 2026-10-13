from dataclasses import dataclass
from typing import Optional
from app.core.ingestion.chunker import DocumentChunk
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.embedder import Embedder
from app.core.retrieval.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with hybrid scoring diagnostics."""
    chunk: DocumentChunk
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None


class HybridRetriever:
    """Retrieves documents by fusing dense vector embeddings and BM25 keyword rankings via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        """Perform hybrid search over indexed chunks.

        Args:
            query: User's question or search query.
            top_k: Number of highest-ranked chunks to return.

        Returns:
            List of RetrievedChunk objects ordered by descending RRF score.
        """
        if self.vector_store.count() == 0:
            return []

        search_k = max(top_k * 3, 10)

        # 1. Dense search
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.query_dense(query_vec, top_k=search_k)

        # 2. BM25 keyword search
        bm25_results = self.bm25_index.query(query, top_k=search_k)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF(d) = sum(1 / (k + rank_i))
        scores: dict[str, float] = {}
        dense_meta: dict[str, tuple[int, float]] = {}
        bm25_meta: dict[str, tuple[int, float]] = {}

        for rank, (cid, score) in enumerate(dense_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            dense_meta[cid] = (rank, score)

        for rank, (cid, score) in enumerate(bm25_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            bm25_meta[cid] = (rank, score)

        # Rank all candidates by fusion score
        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        top_cids = sorted_cids[:top_k]

        retrieved: list[RetrievedChunk] = []
        for cid in top_cids:
            chunk = self.vector_store.get_chunk(cid)
            if chunk is None:
                continue

            d_info = dense_meta.get(cid)
            b_info = bm25_meta.get(cid)

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    rrf_score=scores[cid],
                    dense_rank=d_info[0] if d_info else None,
                    dense_score=d_info[1] if d_info else None,
                    bm25_rank=b_info[0] if b_info else None,
                    bm25_score=b_info[1] if b_info else None,
                )
            )

        return retrieved
