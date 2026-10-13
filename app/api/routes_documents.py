from pathlib import Path
from typing import Any
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from app.core.ingestion.chunker import chunk_document
from app.core.ingestion.loaders import DocumentLoadError, load_from_bytes
from app.models.api_models import DocumentInfo, DocumentUploadResponse


router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document (PDF, TXT, MD)",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="Document file to upload (.pdf, .txt, .md)"),
) -> DocumentUploadResponse:
    """Upload a document, extract text and metadata, generate heading-aware chunks, and index into vector + BM25 stores.

    If a document with the same filename already exists, it is replaced cleanly.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. Allowed: .pdf, .txt, .md",
        )

    content = await file.read()
    if not content or len(content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        loaded_doc = load_from_bytes(file.filename, content)
    except DocumentLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process document: {str(e)}",
        )

    settings = request.app.state.settings
    vector_store = request.app.state.vector_store
    embedder = request.app.state.embedder
    bm25_index = request.app.state.bm25_index

    # Chunk the document
    chunks = chunk_document(
        loaded_doc,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{file.filename}' yielded no readable text chunks.",
        )

    # Attach filename into chunk metadata
    for c in chunks:
        c.metadata["filename"] = file.filename

    # Clean replacement: delete old chunks for this doc_id
    vector_store.delete_by_doc_id(loaded_doc.doc_id)

    # Embed and store
    embeddings = embedder.embed_documents([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    # Rebuild BM25 index over all active chunks in vector store
    all_chunks = vector_store.get_all_chunks()
    bm25_index.build(all_chunks)

    return DocumentUploadResponse(
        doc_id=loaded_doc.doc_id,
        filename=file.filename,
        chunk_count=len(chunks),
        message=f"Document '{file.filename}' successfully ingested and indexed.",
    )


@router.get(
    "",
    response_model=list[DocumentInfo],
    summary="List all ingested documents and their chunk counts",
)
def list_documents(request: Request) -> list[DocumentInfo]:
    """Retrieve catalog of all ingested documents."""
    vector_store = request.app.state.vector_store
    summaries = vector_store.get_document_summary()
    return [
        DocumentInfo(
            doc_id=item["doc_id"],
            filename=item["filename"],
            doc_title=item["doc_title"],
            version=item["version"],
            effective_date=item["effective_date"],
            chunk_count=item["chunk_count"],
        )
        for item in summaries
    ]


@router.delete(
    "/{doc_id}",
    summary="Delete a document and remove its chunks from search indexes",
)
def delete_document(request: Request, doc_id: str) -> dict[str, Any]:
    """Remove a document by its doc_id from both vector and keyword stores."""
    vector_store = request.app.state.vector_store
    bm25_index = request.app.state.bm25_index

    removed_count = vector_store.delete_by_doc_id(doc_id)
    if removed_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with doc_id '{doc_id}' not found.",
        )

    # Rebuild BM25 index with surviving chunks
    all_chunks = vector_store.get_all_chunks()
    bm25_index.build(all_chunks)

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_removed": removed_count,
        "message": f"Successfully deleted document '{doc_id}'.",
    }
