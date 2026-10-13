import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union
import pypdf
from app.core.ingestion.metadata import parse_frontmatter


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded, parsed, or is invalid."""
    pass


@dataclass
class LoadedDocument:
    """Represents a loaded document before chunking."""
    doc_id: str
    filename: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_doc_id(filename: str) -> str:
    """Generate a clean, deterministic doc_id from filename.

    Re-uploading the same filename will result in the same doc_id,
    allowing clean replacement in vector and BM25 stores.
    """
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename.strip())
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return f"doc_{clean_name}"


def load_from_bytes(filename: str, content: bytes) -> LoadedDocument:
    """Load and parse document from raw bytes based on extension.

    Supports .txt, .md, .pdf.

    Args:
        filename: Name of the file with extension.
        content: Raw file bytes.

    Returns:
        LoadedDocument containing cleaned text and parsed metadata.

    Raises:
        DocumentLoadError: If file is empty, corrupted, or unsupported.
    """
    if not content or len(content.strip()) == 0:
        raise DocumentLoadError(f"File '{filename}' is empty.")

    ext = Path(filename).suffix.lower()
    doc_id = generate_doc_id(filename)

    if ext == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise DocumentLoadError(f"Could not decode text file '{filename}': {e}") from e

        if not text.strip():
            raise DocumentLoadError(f"File '{filename}' contains no readable text.")

        metadata = {
            "doc_title": Path(filename).stem,
            "version": "1.0",
            "effective_date": "1970-01-01",
        }
        return LoadedDocument(doc_id=doc_id, filename=filename, raw_text=text, metadata=metadata)

    elif ext == ".md":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise DocumentLoadError(f"Could not decode markdown file '{filename}': {e}") from e

        if not text.strip():
            raise DocumentLoadError(f"File '{filename}' contains no readable text.")

        body_text, metadata = parse_frontmatter(text, default_title=Path(filename).stem)
        return LoadedDocument(doc_id=doc_id, filename=filename, raw_text=body_text, metadata=metadata)

    elif ext == ".pdf":
        try:
            stream = io.BytesIO(content)
            reader = pypdf.PdfReader(stream)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    raise DocumentLoadError(f"Encrypted PDF '{filename}' cannot be opened: {e}") from e

            pages_text: list[str] = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages_text.append(extracted)

            combined_text = "\n\n".join(pages_text).strip()
            if not combined_text:
                raise DocumentLoadError(f"PDF file '{filename}' contains no extractable text.")

            # Extract any PDF metadata
            pdf_info = reader.metadata or {}
            title = pdf_info.title if getattr(pdf_info, "title", None) else Path(filename).stem
            metadata = {
                "doc_title": str(title) if title else Path(filename).stem,
                "version": "1.0",
                "effective_date": "1970-01-01",
                "page_count": len(reader.pages),
            }
            return LoadedDocument(doc_id=doc_id, filename=filename, raw_text=combined_text, metadata=metadata)
        except DocumentLoadError:
            raise
        except Exception as e:
            raise DocumentLoadError(f"Corrupted or invalid PDF '{filename}': {e}") from e

    else:
        raise DocumentLoadError(f"Unsupported document format '{ext}'. Supported: .pdf, .txt, .md")


def load_from_path(file_path: Union[str, Path]) -> LoadedDocument:
    """Load a document directly from a filesystem path."""
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadError(f"File not found: {path}")
    try:
        data = path.read_bytes()
    except Exception as e:
        raise DocumentLoadError(f"Failed to read file {path}: {e}") from e
    return load_from_bytes(path.name, data)
