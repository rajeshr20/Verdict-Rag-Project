import re
from typing import Any, Tuple
import yaml


FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(raw_text: str, default_title: str = "Untitled") -> Tuple[str, dict[str, Any]]:
    """Parse optional YAML front-matter from raw text.

    Supports front-matter bounded by '---'.
    Extracts doc_title (or title), version, effective_date, and other custom keys.

    Args:
        raw_text: Full document text.
        default_title: Fallback document title if not present in front-matter.

    Returns:
        A tuple of (body_text, metadata_dict).
    """
    match = FRONTMATTER_PATTERN.match(raw_text)
    metadata: dict[str, Any] = {
        "doc_title": default_title,
        "version": "1.0",
        "effective_date": "1970-01-01",
    }

    if not match:
        return raw_text, metadata

    yaml_block = match.group(1)
    body_text = raw_text[match.end():]

    try:
        parsed = yaml.safe_load(yaml_block)
        if isinstance(parsed, dict):
            if "title" in parsed:
                metadata["doc_title"] = str(parsed["title"])
            if "doc_title" in parsed:
                metadata["doc_title"] = str(parsed["doc_title"])
            if "version" in parsed:
                metadata["version"] = str(parsed["version"])
            if "effective_date" in parsed:
                metadata["effective_date"] = str(parsed["effective_date"])
            for k, v in parsed.items():
                if k not in metadata:
                    metadata[k] = v
    except Exception:
        # If YAML parsing fails, return raw text unchanged with default metadata
        return raw_text, metadata

    return body_text, metadata
