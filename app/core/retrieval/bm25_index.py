import re
from typing import Optional
from rank_bm25 import BM25Okapi
from app.core.ingestion.chunker import DocumentChunk


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into alphanumeric words and preserved numbers for BM25.

    Converts to lowercase and splits on whitespace/punctuation while
    preserving numbers and terms like 'rs' or '3,500' -> ['rs', '3500', '3', '500'].
    """
    cleaned = text.lower()
    # Match words, alphanumeric sequences, and currency symbols
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    return tokens


class BM25Index:
    """Keyword search index using BM25Okapi over document chunks.

    Rebuilt in-memory whenever documents are ingested or deleted.
    """

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.chunks_map: dict[str, DocumentChunk] = {}
        self.bm25: Optional[BM25Okapi] = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index over a collection of chunks."""
        self.chunk_ids = []
        self.chunks_map = {}

        if not chunks:
            self.bm25 = None
            return

        corpus: list[list[str]] = []
        for chunk in chunks:
            self.chunk_ids.append(chunk.chunk_id)
            self.chunks_map[chunk.chunk_id] = chunk
            tokens = tokenize_text(chunk.text)
            corpus.append(tokens)

        self.bm25 = BM25Okapi(corpus)

    def query(self, query_text: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Query the BM25 index with a query string.

        Returns:
            List of (chunk_id, bm25_score) sorted descending by score.
        """
        if not self.bm25 or not self.chunk_ids:
            return []

        tokens = tokenize_text(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs: list[tuple[str, float]] = [
            (self.chunk_ids[i], float(scores[i]))
            for i in range(len(self.chunk_ids))
            if scores[i] > 0.0
        ]

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def count(self) -> int:
        return len(self.chunk_ids)
