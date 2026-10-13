from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Optional
from app.core.generation.schema import Conflict
from app.core.retrieval.hybrid import RetrievedChunk


@dataclass
class ConflictingSource:
    """Metadata and excerpt for one side of a document conflict."""
    chunk_id: str
    document: str
    version: str
    effective_date: str
    section: str
    excerpt: str


@dataclass
class ConflictReport:
    """Enriched conflict report attaching document provenance and chronological resolution."""
    topic: str
    description: str
    sources: list[ConflictingSource] = field(default_factory=list)
    newer_chunk_id: Optional[str] = None
    newer_document: Optional[str] = None
    resolution_note: str = ""


def _parse_date_safe(date_str: str) -> Optional[date]:
    """Attempt to parse date from string (supports YYYY-MM-DD or YYYY)."""
    s = str(date_str).strip()
    if not s:
        return None
    try:
        if len(s) == 4 and s.isdigit():
            return date(int(s), 1, 1)
        return date.fromisoformat(s)
    except Exception:
        return None


def resolve_conflicts(
    raw_conflicts: list[Conflict],
    chunk_item_map: Mapping[str, RetrievedChunk],
) -> list[ConflictReport]:
    """Enrich LLM-reported conflicts with document metadata and identify the newer policy.

    Args:
        raw_conflicts: List of Conflict objects emitted by LLM.
        chunk_item_map: Map from prompt ID (e.g. 'c1') to the RetrievedChunk object.

    Returns:
        List of ConflictReport objects with explicit newer-document markings.
    """
    reports: list[ConflictReport] = []

    for conflict in raw_conflicts:
        sources: list[ConflictingSource] = []

        for cid in conflict.chunk_ids:
            if cid in chunk_item_map:
                item = chunk_item_map[cid]
                chunk = item.chunk
                sources.append(
                    ConflictingSource(
                        chunk_id=cid,
                        document=str(chunk.metadata.get("filename") or chunk.doc_id),
                        version=str(chunk.metadata.get("version", "1.0")),
                        effective_date=str(chunk.metadata.get("effective_date", "1970-01-01")),
                        section=chunk.section,
                        excerpt=chunk.text[:200].strip() + ("..." if len(chunk.text) > 200 else ""),
                    )
                )

        if not sources:
            continue

        # Determine newest source by effective_date or version
        best_source: Optional[ConflictingSource] = None
        best_date: Optional[date] = None

        for src in sources:
            parsed = _parse_date_safe(src.effective_date)
            if parsed is not None:
                if best_date is None or parsed > best_date:
                    best_date = parsed
                    best_source = src
            elif best_source is None:
                best_source = src

        # Build resolution note
        newer_cid: Optional[str] = None
        newer_doc: Optional[str] = None
        note = ""

        if best_source and len(sources) > 1:
            newer_cid = best_source.chunk_id
            newer_doc = best_source.document
            other_docs = [s.document for s in sources if s.chunk_id != newer_cid]
            note = (
                f"Document '{best_source.document}' (effective: {best_source.effective_date}, "
                f"v{best_source.version}) is newer and supersedes {', '.join(other_docs)}."
            )
        elif len(sources) > 1:
            note = "Conflicting documents share identical or undetermined effective dates; manual policy clarification is recommended."

        reports.append(
            ConflictReport(
                topic=conflict.topic,
                description=conflict.description,
                sources=sources,
                newer_chunk_id=newer_cid,
                newer_document=newer_doc,
                resolution_note=note,
            )
        )

    return reports
