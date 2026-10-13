"""Ingest all sample corpus files into the local or remote VERDICT service."""
import sys
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.core.ingestion.chunker import chunk_document
from app.core.ingestion.loaders import load_from_path
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.embedder import get_embedder
from app.core.retrieval.vector_store import VectorStore


def ingest_via_api(base_url: str = "http://localhost:8000") -> bool:
    corpus_dir = REPO_ROOT / "data" / "sample_docs"
    files = sorted(corpus_dir.glob("*.*"))
    print(f"Attempting API upload to {base_url} for {len(files)} files...")

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            resp = client.get("/health")
            if resp.status_code != 200:
                return False

            for f in files:
                if f.suffix.lower() not in (".txt", ".md", ".pdf"):
                    continue
                with open(f, "rb") as fp:
                    content_type = "application/pdf" if f.suffix == ".pdf" else "text/markdown"
                    upload_resp = client.post(
                        "/documents",
                        files={"file": (f.name, fp, content_type)},
                    )
                    if upload_resp.status_code in (200, 201):
                        data = upload_resp.json()
                        print(f" -> Ingested {f.name}: {data['chunk_count']} chunks")
                    else:
                        print(f" -> Failed {f.name}: {upload_resp.status_code} {upload_resp.text}")
        print("API ingestion completed successfully.")
        return True
    except Exception as e:
        print(f"Could not reach running server ({e}). Falling back to local store.")
        return False


def ingest_locally() -> None:
    settings = get_settings()
    corpus_dir = REPO_ROOT / "data" / "sample_docs"
    files = sorted(corpus_dir.glob("*.*"))

    print(f"Ingesting {len(files)} files locally into {settings.CHROMA_DIR}...")
    vector_store = VectorStore(persist_dir=settings.CHROMA_DIR)
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    bm25_index = BM25Index()

    total_chunks = 0
    for f in files:
        if f.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        doc = load_from_path(f)
        chunks = chunk_document(doc, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        if not chunks:
            continue
        for c in chunks:
            c.metadata["filename"] = f.name

        vector_store.delete_by_doc_id(doc.doc_id)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        print(f" -> Ingested {f.name}: {len(chunks)} chunks")

    all_chunks = vector_store.get_all_chunks()
    bm25_index.build(all_chunks)
    print(f"Local ingestion complete. Total indexed chunks: {total_chunks}")


if __name__ == "__main__":
    if not ingest_via_api():
        ingest_locally()
