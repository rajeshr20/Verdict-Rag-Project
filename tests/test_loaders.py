import io
import pytest
from reportlab.pdfgen import canvas
from app.core.ingestion.loaders import (
    load_from_bytes,
    load_from_path,
    DocumentLoadError,
    generate_doc_id,
)
from app.core.ingestion.metadata import parse_frontmatter


def create_minimal_pdf(text: str = "Meridian Labs Security Policy 2025") -> bytes:
    """Helper to generate a valid PDF in-memory using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, text)
    c.save()
    return buf.getvalue()


def test_generate_doc_id():
    id1 = generate_doc_id("handbook_2023.md")
    id2 = generate_doc_id("handbook_2023.md")
    assert id1 == id2
    assert id1 == "doc_handbook_2023_md"


def test_parse_frontmatter():
    text_with_fm = """---
title: "Employee Handbook"
version: "2025"
effective_date: "2025-01-01"
author: "HR Team"
---
# Welcome
This is the handbook text.
"""
    body, meta = parse_frontmatter(text_with_fm)
    assert meta["doc_title"] == "Employee Handbook"
    assert meta["version"] == "2025"
    assert meta["effective_date"] == "2025-01-01"
    assert meta["author"] == "HR Team"
    assert "# Welcome" in body
    assert "---" not in body

    # Without frontmatter
    plain_text = "# Plain Note\nNo frontmatter here."
    body2, meta2 = parse_frontmatter(plain_text, default_title="Fallback")
    assert body2 == plain_text
    assert meta2["doc_title"] == "Fallback"
    assert meta2["version"] == "1.0"


def test_load_txt_file():
    content = b"Simple plain text file contents."
    doc = load_from_bytes("notes.txt", content)
    assert doc.doc_id == "doc_notes_txt"
    assert doc.filename == "notes.txt"
    assert doc.raw_text == "Simple plain text file contents."
    assert doc.metadata["version"] == "1.0"


def test_load_md_file():
    content = b"""---
title: Meridian Policy
version: "2.1"
effective_date: "2025-02-15"
---
## Remote Work
Employees may work remotely 2 days per week.
"""
    doc = load_from_bytes("policy.md", content)
    assert doc.doc_id == "doc_policy_md"
    assert doc.metadata["doc_title"] == "Meridian Policy"
    assert doc.metadata["version"] == "2.1"
    assert doc.metadata["effective_date"] == "2025-02-15"
    assert "Employees may work remotely" in doc.raw_text


def test_load_pdf_file():
    pdf_bytes = create_minimal_pdf("Confidential Meridian Documentation.")
    doc = load_from_bytes("document.pdf", pdf_bytes)
    assert doc.doc_id == "doc_document_pdf"
    assert "Confidential Meridian Documentation." in doc.raw_text
    assert doc.metadata["page_count"] == 1


def test_corrupted_pdf_raises_clean_error():
    corrupted_bytes = b"%PDF-1.4 completely invalid stream content garbage"
    with pytest.raises(DocumentLoadError) as exc_info:
        load_from_bytes("corrupted.pdf", corrupted_bytes)
    assert "Corrupted or invalid PDF" in str(exc_info.value)


def test_empty_file_raises_error():
    with pytest.raises(DocumentLoadError) as exc_info:
        load_from_bytes("empty.txt", b"")
    assert "is empty" in str(exc_info.value)

    with pytest.raises(DocumentLoadError) as exc_info:
        load_from_bytes("whitespace.md", b"   \n  \t  ")
    assert "is empty" in str(exc_info.value)


def test_load_from_path(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello from file on disk.")
    doc = load_from_path(f)
    assert doc.raw_text == "Hello from file on disk."

    with pytest.raises(DocumentLoadError):
        load_from_path(tmp_path / "nonexistent.txt")


def test_unsupported_extension_raises_error():
    with pytest.raises(DocumentLoadError) as exc_info:
        load_from_bytes("archive.zip", b"PK000fakezipdata")
    assert "Unsupported document format" in str(exc_info.value)
