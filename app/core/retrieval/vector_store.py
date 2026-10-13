from pathlib import Path
from typing import Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.ingestion.chunker import DocumentChunk


class VectorStore:
    """Wrapper around persistent ChromaDB for chunk storage and dense vector search."""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "verdict_chunks") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Upsert document chunks and their pre-computed embeddings into Chroma."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "doc_title": str(c.metadata.get("doc_title", "")),
                "version": str(c.metadata.get("version", "1.0")),
                "effective_date": str(c.metadata.get("effective_date", "1970-01-01")),
                "filename": str(c.metadata.get("filename", "")),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a given doc_id. Returns number of chunks removed."""
        results = self.collection.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def query_dense(self, query_embedding: list[float], top_k: int = 6) -> list[tuple[str, float]]:
        """Query Chroma for the top_k closest chunks.

        Returns:
            List of (chunk_id, similarity_score) sorted by descending similarity.
            Similarity score is in range [0.0, 1.0].
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["distances"],
        )

        output: list[tuple[str, float]] = []
        if res["ids"] and len(res["ids"][0]) > 0:
            ids = res["ids"][0]
            distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

            for cid, dist in zip(ids, distances):
                # For cosine distance in Chroma: similarity = 1 - distance
                sim = max(0.0, min(1.0, 1.0 - dist))
                output.append((cid, float(sim)))

        return output

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its chunk_id."""
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None

        text = res["documents"][0]
        meta = res["metadatas"][0]
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            text=text,
            section=meta.get("section", "General"),
            start_char=int(meta.get("start_char", 0)),
            end_char=int(meta.get("end_char", len(text))),
            metadata=meta,
        )

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all stored chunks across all documents."""
        res = self.collection.get(include=["documents", "metadatas"])
        chunks: list[DocumentChunk] = []
        if not res["ids"]:
            return chunks

        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=doc,
                    section=meta.get("section", "General"),
                    start_char=int(meta.get("start_char", 0)),
                    end_char=int(meta.get("end_char", len(doc))),
                    metadata=meta,
                )
            )
        return chunks

    def count(self) -> int:
        """Total number of chunks stored."""
        return self.collection.count()

    def get_document_summary(self) -> list[dict[str, Any]]:
        """Group chunks by doc_id to list stored documents with chunk count and metadata."""
        all_chunks = self.get_all_chunks()
        docs: dict[str, dict[str, Any]] = {}
        for c in all_chunks:
            did = c.doc_id
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "filename": c.metadata.get("filename", did),
                    "doc_title": c.metadata.get("doc_title", did),
                    "version": c.metadata.get("version", "1.0"),
                    "effective_date": c.metadata.get("effective_date", "1970-01-01"),
                    "chunk_count": 0,
                }
            docs[did]["chunk_count"] += 1
        return list(docs.values())
