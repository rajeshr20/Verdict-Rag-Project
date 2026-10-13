import re
from dataclasses import dataclass, field
from typing import Mapping
from app.core.generation.schema import Claim


UNICODE_QUOTE_MAP = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "\u00a0": " ",  # Non-breaking space
    "\u2002": " ",  # En space
    "\u2003": " ",  # Em space
    "\u2009": " ",  # Thin space
    "\u202f": " ",  # Narrow no-break space
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
    "\u2212": "-",  # Minus
})


def normalize_text_for_verification(text: str) -> str:
    """Normalize text for robust substring quote verification.

    - Replaces unicode quotes, dashes, and special spaces with ASCII equivalents.
    - Collapses consecutive whitespace (spaces, tabs, newlines) into a single space.
    - Converts to lowercase.
    """
    if not text:
        return ""
    # Map unicode quotes/dashes/spaces
    t = text.translate(UNICODE_QUOTE_MAP)
    # Collapse all whitespace sequences to a single space
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


@dataclass
class VerificationResult:
    """Outcome of deterministic claim verification against source chunks."""
    verified_claims: list[Claim] = field(default_factory=list)
    dropped_claims: list[tuple[Claim, str]] = field(default_factory=list)
    claims_total: int = 0
    claims_verified: int = 0
    claims_dropped: int = 0
    is_answerable: bool = False
    rebuilt_answer: str = ""


def verify_claims(
    claims: list[Claim],
    chunk_map: Mapping[str, str],
) -> VerificationResult:
    """Deterministically verify that every claim's quote exists in its referenced chunk.

    Claims with fabricated quotes, empty quotes, or quotes not present in the
    referenced chunk are dropped. Reconstructs the answer text strictly from
    the surviving verified claims.

    Args:
        claims: Raw list of claims returned by the LLM.
        chunk_map: Dictionary mapping prompt chunk IDs (e.g. 'c1') to their raw text.

    Returns:
        VerificationResult detailing verified claims, dropped claims, counts, and rebuilt answer.
    """
    total = len(claims)
    verified: list[Claim] = []
    dropped: list[tuple[Claim, str]] = []

    for claim in claims:
        cid = claim.chunk_id.strip()
        quote = claim.quote.strip()

        if not cid or cid not in chunk_map:
            dropped.append((claim, f"Chunk ID '{cid}' does not exist in retrieved context."))
            continue

        if not quote:
            dropped.append((claim, "Empty quote provided."))
            continue

        chunk_text = chunk_map[cid]
        norm_quote = normalize_text_for_verification(quote)
        norm_chunk = normalize_text_for_verification(chunk_text)

        if not norm_quote:
            dropped.append((claim, "Normalized quote is empty."))
            continue

        if norm_quote in norm_chunk:
            verified.append(claim)
        else:
            dropped.append((claim, "Quote does not exist in referenced chunk text."))

    n_verified = len(verified)
    n_dropped = len(dropped)

    if n_verified == 0:
        return VerificationResult(
            verified_claims=[],
            dropped_claims=dropped,
            claims_total=total,
            claims_verified=0,
            claims_dropped=n_dropped,
            is_answerable=False,
            rebuilt_answer="The provided documents do not contain this information.",
        )

    # Reconstruct final answer purely from verified claims
    rebuilt_sentences: list[str] = []
    for c in verified:
        stmt = c.text.strip()
        if not stmt.endswith((".", "!", "?")):
            stmt += "."
        rebuilt_sentences.append(stmt)

    rebuilt_answer = " ".join(rebuilt_sentences)

    return VerificationResult(
        verified_claims=verified,
        dropped_claims=dropped,
        claims_total=total,
        claims_verified=n_verified,
        claims_dropped=n_dropped,
        is_answerable=True,
        rebuilt_answer=rebuilt_answer,
    )
