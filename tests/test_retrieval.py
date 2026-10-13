import pytest
from app.core.ingestion.chunker import DocumentChunk
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.embedder import FakeEmbedder
from app.core.retrieval.gate import evaluate_retrieval_gate
from app.core.retrieval.hybrid import HybridRetriever, RetrievedChunk
from app.core.retrieval.vector_store import VectorStore


def test_fake_embedder_deterministic_and_normalized():
    embedder = FakeEmbedder(dimension=64)
    v1 = embedder.embed_query("Meridian Labs remote work policy")
    v2 = embedder.embed_query("Meridian Labs remote work policy")
    assert v1 == v2
    assert len(v1) == 64

    # Norm must be ~1.0
    norm = sum(x * x for x in v1) ** 0.5
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Query with overlapping words should have higher dot product than disjoint words
    v_related = embedder.embed_query("remote work guidelines")
    v_unrelated = embedder.embed_query("quantum gravitational singularities")

    sim_related = sum(a * b for a, b in zip(v1, v_related))
    sim_unrelated = sum(a * b for a, b in zip(v1, v_unrelated))
    assert sim_related > sim_unrelated


def test_bm25_catches_exact_terms():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Meal allowance is capped at Rs 2,000 per person per day.",
            section="Expenses",
            start_char=0,
            end_char=50,
            metadata={},
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            text="Meal expense limit has been raised to Rs 3,500 per person.",
            section="Expenses",
            start_char=0,
            end_char=52,
            metadata={},
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            text="Security policy requires MFA on all employee devices.",
            section="Security",
            start_char=0,
            end_char=50,
            metadata={},
        ),
    ]

    index = BM25Index()
    index.build(chunks)

    results = index.query("Rs 3,500", top_k=3)
    assert len(results) > 0
    top_chunk_id, top_score = results[0]
    assert top_chunk_id == "c2"
    assert top_score > 0.0


def test_hybrid_retriever_and_rrf(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_hybrid")
    embedder = FakeEmbedder(dimension=64)
    bm25 = BM25Index()

    chunks = [
        DocumentChunk(
            chunk_id="c_leave",
            doc_id="d_handbook",
            text="Employees receive 20 days of annual paid leave.",
            section="Leave",
            start_char=0,
            end_char=45,
            metadata={"doc_title": "Handbook"},
        ),
        DocumentChunk(
            chunk_id="c_remote",
            doc_id="d_handbook",
            text="Remote work is permitted 2 days per week for engineering.",
            section="Remote Work",
            start_char=46,
            end_char=100,
            metadata={"doc_title": "Handbook"},
        ),
        DocumentChunk(
            chunk_id="c_product",
            doc_id="d_product",
            text="Lumen Gateway tier pricing starts at $499 monthly.",
            section="Pricing",
            start_char=0,
            end_char=48,
            metadata={"doc_title": "Product FAQ"},
        ),
    ]

    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    bm25.build(chunks)

    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, embedder=embedder, rrf_k=60)
    results = retriever.retrieve("How many days of paid leave?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c_leave"
    assert results[0].rrf_score > 0.0


def test_gate_triggers_on_irrelevant_or_empty():
    # Empty chunks -> gated
    is_gated, reason = evaluate_retrieval_gate([], min_score=0.01)
    assert is_gated is True
    assert "No documents were retrieved" in reason

    # Low score chunk -> gated
    dummy_chunk = DocumentChunk(
        chunk_id="c_low",
        doc_id="d_low",
        text="Arbitrary text",
        section="General",
        start_char=0,
        end_char=14,
    )
    low_result = RetrievedChunk(chunk=dummy_chunk, rrf_score=0.005)
    is_gated_low, reason_low = evaluate_retrieval_gate([low_result], min_score=0.01)
    assert is_gated_low is True
    assert "below threshold" in reason_low

    # Sufficient score chunk -> not gated
    high_result = RetrievedChunk(chunk=dummy_chunk, rrf_score=0.025)
    is_gated_high, reason_high = evaluate_retrieval_gate([high_result], min_score=0.01)
    assert is_gated_high is False
    assert reason_high == ""


def test_embedder_factory():
    from app.core.retrieval.embedder import get_embedder, FakeEmbedder
    fe = get_embedder("fake")
    assert isinstance(fe, FakeEmbedder)
    assert fe.dimension == 128
    vecs = fe.embed_documents(["doc one", "doc two"])
    assert len(vecs) == 2
