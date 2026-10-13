import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.core.generation.llm import FakeLLM, LLMError
from app.core.generation.schema import Claim, LLMResponse
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        LLM_PROVIDER="fake",
        EMBEDDING_MODEL="fake",
        CHROMA_DIR=str(tmp_path / "chroma_api"),
        RETRIEVAL_MIN_SCORE=0.005,
    )
    test_app = create_app(custom_settings=settings)
    with TestClient(test_app) as c:
        yield c, test_app


def test_health_endpoint(client):
    c, app = client
    response = c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["llm_provider"] == "fake"
    assert data["embedding_provider"] == "fake"


def test_upload_valid_document(client):
    c, app = client
    content = b"""---
title: Vacation Rules
version: "2025"
effective_date: "2025-01-01"
---
# Vacation Policy
Full-time employees receive 25 days paid vacation annually.
"""
    files = {"file": ("vacation.md", content, "text/markdown")}
    resp = c.post("/documents", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "vacation.md"
    assert data["chunk_count"] >= 1
    assert "doc_id" in data


def test_upload_unsupported_file_format(client):
    c, app = client
    files = {"file": ("binary.bin", b"\x00\x01\x02\x03", "application/octet-stream")}
    resp = c.post("/documents", files=files)
    assert resp.status_code == 415
    assert "Unsupported file format" in resp.json()["error"]


def test_upload_empty_file(client):
    c, app = client
    files = {"file": ("empty.txt", b"", "text/plain")}
    resp = c.post("/documents", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["error"]


def test_duplicate_upload_replaces_old_version(client):
    c, app = client
    v1_content = b"# Expense\nLimit is Rs 1,000."
    v2_content = b"# Expense\nLimit has been updated to Rs 2,500."

    # First upload
    resp1 = c.post("/documents", files={"file": ("expenses.md", v1_content, "text/markdown")})
    assert resp1.status_code == 201
    doc_id = resp1.json()["doc_id"]

    # Check list
    list1 = c.get("/documents").json()
    assert len(list1) == 1
    assert list1[0]["doc_id"] == doc_id
    assert list1[0]["chunk_count"] == 1

    # Re-upload with same filename
    resp2 = c.post("/documents", files={"file": ("expenses.md", v2_content, "text/markdown")})
    assert resp2.status_code == 201

    # Chunk count should remain 1, not duplicate to 2
    list2 = c.get("/documents").json()
    assert len(list2) == 1
    assert list2[0]["chunk_count"] == 1


def test_delete_document_and_not_found(client):
    c, app = client
    content = b"# Security\nBadge required at all times."
    resp_upload = c.post("/documents", files={"file": ("badge.md", content, "text/markdown")})
    doc_id = resp_upload.json()["doc_id"]

    # Delete existing
    del_resp = c.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["chunks_removed"] >= 1

    # Delete again -> 404
    del_resp_404 = c.delete(f"/documents/{doc_id}")
    assert del_resp_404.status_code == 404


def test_ask_happy_path(client):
    c, app = client
    # Ingest document
    content = b"""---
title: Handbook
version: "2025"
---
# Leave
Employees receive 20 days paid leave per calendar year.
"""
    c.post("/documents", files={"file": ("handbook.md", content, "text/markdown")})

    # Configure FakeLLM response
    fake_llm: FakeLLM = app.state.llm
    fake_llm.set_response(
        LLMResponse(
            answerable=True,
            answer="Employees receive 20 days paid leave.",
            claims=[
                Claim(
                    text="Employees receive 20 days paid leave per calendar year.",
                    chunk_id="c1",
                    quote="receive 20 days paid leave per calendar year",
                )
            ],
            conflicts=[],
        )
    )

    ask_resp = c.post("/ask", json={"question": "How many days paid leave?"})
    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert data["answerable"] is True
    assert "20 days" in data["answer"]
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["verified"] is True
    assert data["diagnostics"]["claims_verified"] == 1


def test_ask_unanswerable_question(client):
    c, app = client
    content = b"# Tools\nWe use Slack and Jira."
    c.post("/documents", files={"file": ("tools.md", content, "text/markdown")})

    fake_llm: FakeLLM = app.state.llm
    fake_llm.set_response(
        LLMResponse(
            answerable=False,
            answer="The provided documents do not contain this information.",
            claims=[],
            conflicts=[],
        )
    )

    ask_resp = c.post("/ask", json={"question": "What is the capital of France?"})
    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert data["answerable"] is False
    assert "do not contain this information" in data["answer"]


def test_ask_empty_question_returns_422(client):
    c, app = client
    resp = c.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_ask_llm_failure_returns_503(client):
    c, app = client
    content = b"# Policy\nSome text."
    c.post("/documents", files={"file": ("policy.md", content, "text/markdown")})

    fake_llm: FakeLLM = app.state.llm
    fake_llm.set_error(LLMError("Upstream model rate limit exceeded"))

    resp = c.post("/ask", json={"question": "What is the policy?"})
    assert resp.status_code == 503
    assert "LLM generation failed" in resp.json()["error"]
