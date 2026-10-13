from app.core.ingestion.chunker import chunk_document
from app.core.ingestion.loaders import LoadedDocument


def test_chunk_empty_document():
    doc = LoadedDocument(doc_id="doc_empty", filename="empty.md", raw_text="", metadata={})
    chunks = chunk_document(doc)
    assert len(chunks) == 0

    doc_ws = LoadedDocument(doc_id="doc_ws", filename="ws.md", raw_text="   \n   ", metadata={})
    chunks_ws = chunk_document(doc_ws)
    assert len(chunks_ws) == 0


def test_chunk_very_short_document():
    text = "Short policy notice: all badges must be returned upon exit."
    doc = LoadedDocument(
        doc_id="doc_short",
        filename="short.md",
        raw_text=text,
        metadata={"version": "1.0"},
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc_short#c0"
    assert chunk.text == text
    assert chunk.start_char == 0
    assert chunk.end_char == len(text)
    assert chunk.section == "General"
    assert chunk.metadata["version"] == "1.0"


def test_chunk_heading_preservation_and_offsets():
    text = (
        "# Leave Policy\n"
        "Employees receive 20 days paid leave per calendar year.\n\n"
        "# Remote Work\n"
        "Employees are permitted to work remotely up to 2 days per week.\n\n"
        "# Expense Limit\n"
        "Daily meal allowance is capped at Rs 3,500 per person."
    )
    doc = LoadedDocument(
        doc_id="doc_handbook",
        filename="handbook.md",
        raw_text=text,
        metadata={"version": "2025"},
    )
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30)
    assert len(chunks) >= 3

    # Verify that every chunk's text precisely matches the slice of the original text
    sections_found = set()
    for chunk in chunks:
        assert doc.raw_text[chunk.start_char:chunk.end_char] == chunk.text
        sections_found.add(chunk.section)

    assert "Leave Policy" in sections_found
    assert "Remote Work" in sections_found
    assert "Expense Limit" in sections_found


def test_chunk_long_section_subsplitting_and_overlap():
    long_paragraph = (
        "## Engineering Standards\n"
        + "Sentence number one describes code quality rules. " * 30
    )
    doc = LoadedDocument(
        doc_id="doc_standards",
        filename="standards.md",
        raw_text=long_paragraph,
        metadata={},
    )
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1

    # Check offsets and overlap
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert doc.raw_text[c1.start_char:c1.end_char] == c1.text
        assert doc.raw_text[c2.start_char:c2.end_char] == c2.text
        # Ensure there is forward progress
        assert c2.start_char < c1.end_char
        assert c2.end_char > c1.end_char
