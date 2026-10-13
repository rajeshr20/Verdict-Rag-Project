import re
from dataclasses import dataclass, field
from typing import Any
from app.core.ingestion.loaders import LoadedDocument


@dataclass
class DocumentChunk:
    """Represents a chunk extracted from a loaded document."""
    chunk_id: str
    doc_id: str
    text: str
    section: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


HEADING_PATTERN = re.compile(r"^(#{1,6}\s+.*|[A-Z][A-Za-z0-9\s\-]{2,50}:)$", re.MULTILINE)


def _clean_section_heading(raw_heading: str) -> str:
    """Strip markdown hashes and trailing colons from section heading."""
    heading = re.sub(r"^#{1,6}\s+", "", raw_heading.strip())
    heading = heading.rstrip(":")
    return heading.strip() or "General"


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """Split a LoadedDocument into heading-aware, overlapping chunks with character offsets.

    Guarantees:
    - Every chunk keeps its section title.
    - start_char and end_char correspond exactly to doc.raw_text[start_char:end_char].
    - Handles very short documents by returning a single chunk.
    - Handles empty text by returning an empty list.

    Args:
        doc: The LoadedDocument to chunk.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlap length in characters between adjacent chunks.

    Returns:
        A list of DocumentChunk objects.
    """
    raw_text = doc.raw_text
    if not raw_text or not raw_text.strip():
        return []

    # Find heading boundaries
    heading_matches = list(HEADING_PATTERN.finditer(raw_text))

    # If no headings exist and text is shorter than chunk_size, return single chunk
    if not heading_matches and len(raw_text) <= chunk_size:
        return [
            DocumentChunk(
                chunk_id=f"{doc.doc_id}#c0",
                doc_id=doc.doc_id,
                text=raw_text,
                section="General",
                start_char=0,
                end_char=len(raw_text),
                metadata=dict(doc.metadata),
            )
        ]

    sections: list[tuple[str, int, int]] = []
    if not heading_matches:
        sections.append(("General", 0, len(raw_text)))
    else:
        # If text precedes the first heading
        if heading_matches[0].start() > 0:
            pre_text = raw_text[: heading_matches[0].start()].strip()
            if pre_text:
                sections.append(("General", 0, heading_matches[0].start()))

        for i, match in enumerate(heading_matches):
            section_title = _clean_section_heading(match.group(0))
            sec_start = match.start()
            sec_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(raw_text)
            sections.append((section_title, sec_start, sec_end))

    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for section_title, sec_start, sec_end in sections:
        sec_text = raw_text[sec_start:sec_end]
        sec_len = len(sec_text)

        if sec_len == 0:
            continue

        if sec_len <= chunk_size:
            chunk = DocumentChunk(
                chunk_id=f"{doc.doc_id}#c{chunk_idx}",
                doc_id=doc.doc_id,
                text=sec_text,
                section=section_title,
                start_char=sec_start,
                end_char=sec_end,
                metadata=dict(doc.metadata),
            )
            chunks.append(chunk)
            chunk_idx += 1
        else:
            # Sub-split long section with overlap
            pos = 0
            while pos < sec_len:
                target_end = min(pos + chunk_size, sec_len)

                # If not at the very end, try breaking at a newline or sentence/space boundary
                actual_end = target_end
                if target_end < sec_len:
                    # Look backwards for newline or period or space within last 120 chars
                    search_start = max(pos + chunk_size - 120, pos + 50)
                    segment = sec_text[search_start:target_end]
                    break_point = -1

                    for sep in ["\n\n", "\n", ". ", " "]:
                        idx = segment.rfind(sep)
                        if idx != -1:
                            break_point = search_start + idx + len(sep)
                            break

                    if break_point != -1 and break_point > pos:
                        actual_end = break_point

                chunk_str = sec_text[pos:actual_end]
                global_start = sec_start + pos
                global_end = sec_start + actual_end

                if chunk_str.strip():
                    chunk = DocumentChunk(
                        chunk_id=f"{doc.doc_id}#c{chunk_idx}",
                        doc_id=doc.doc_id,
                        text=chunk_str,
                        section=section_title,
                        start_char=global_start,
                        end_char=global_end,
                        metadata=dict(doc.metadata),
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                if actual_end >= sec_len:
                    break

                # Advance with overlap
                pos = max(pos + 1, actual_end - chunk_overlap)

    return chunks
