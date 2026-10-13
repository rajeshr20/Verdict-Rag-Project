from app.core.generation.schema import Claim
from app.core.verification.verifier import normalize_text_for_verification, verify_claims


def test_normalize_text_for_verification():
    raw = "  “Smart Quotes” and ‘Apostrophes’\u00a0with \n\t multiple   spaces—dash.  "
    normalized = normalize_text_for_verification(raw)
    assert normalized == '"smart quotes" and \'apostrophes\' with multiple spaces-dash.'


def test_exact_quote_passes():
    chunk_text = "Employees are entitled to 20 days paid leave per calendar year."
    claim = Claim(
        text="Annual leave is 20 days.",
        chunk_id="c1",
        quote="20 days paid leave per calendar year",
    )
    result = verify_claims([claim], {"c1": chunk_text})

    assert result.is_answerable is True
    assert result.claims_total == 1
    assert result.claims_verified == 1
    assert result.claims_dropped == 0
    assert len(result.verified_claims) == 1
    assert result.rebuilt_answer == "Annual leave is 20 days."


def test_whitespace_and_case_variants_pass():
    chunk_text = "Remote work is permitted\n  2 days   per week for full-time employees."
    claim = Claim(
        text="Employees can work remotely 2 days weekly.",
        chunk_id="c1",
        # Case variation + single space vs newline/multiple spaces
        quote="REMOTE WORK IS PERMITTED 2 DAYS PER WEEK",
    )
    result = verify_claims([claim], {"c1": chunk_text})

    assert result.is_answerable is True
    assert result.claims_verified == 1
    assert result.claims_dropped == 0


def test_unicode_quotes_and_nbsp_pass():
    chunk_text = 'Policy allows “reasonable” expenses up to \u00a0Rs 3,500.'
    claim = Claim(
        text="Expenses up to Rs 3500 allowed.",
        chunk_id="c1",
        # standard quotes and normal space
        quote='"reasonable" expenses up to  Rs 3,500',
    )
    result = verify_claims([claim], {"c1": chunk_text})

    assert result.is_answerable is True
    assert result.claims_verified == 1


def test_fabricated_quote_fails_and_drops():
    chunk_text = "Meal allowance is strictly capped at Rs 2,000 per person."
    claim = Claim(
        text="Employees receive unlimited dinner credits.",
        chunk_id="c1",
        quote="unlimited dinner credits without any limit",
    )
    result = verify_claims([claim], {"c1": chunk_text})

    assert result.is_answerable is False
    assert result.claims_total == 1
    assert result.claims_verified == 0
    assert result.claims_dropped == 1
    assert result.rebuilt_answer == "The provided documents do not contain this information."


def test_quote_from_wrong_chunk_fails():
    chunks = {
        "c1": "Probation period is 6 months from start date.",
        "c2": "Working hours are 9 AM to 6 PM Monday to Friday.",
    }
    # Claim points to c2, but quote is from c1
    claim = Claim(
        text="Probation is 6 months.",
        chunk_id="c2",
        quote="Probation period is 6 months",
    )
    result = verify_claims([claim], chunks)

    assert result.is_answerable is False
    assert result.claims_verified == 0
    assert result.claims_dropped == 1


def test_empty_or_missing_chunk_drops():
    chunks = {"c1": "Some text."}
    claim1 = Claim(text="Something", chunk_id="c99", quote="Some text")
    claim2 = Claim(text="Something else", chunk_id="c1", quote="")

    result = verify_claims([claim1, claim2], chunks)
    assert result.is_answerable is False
    assert result.claims_total == 2
    assert result.claims_dropped == 2


def test_partial_verification_rebuilds_answer_only_from_verified():
    chunks = {"c1": "Server uptime SLA is 99.95% on Enterprise tiers."}
    valid_claim = Claim(
        text="Enterprise tiers guarantee 99.95% uptime.",
        chunk_id="c1",
        quote="uptime SLA is 99.95%",
    )
    hallucinated_claim = Claim(
        text="Support response time is 10 seconds.",
        chunk_id="c1",
        quote="response time is 10 seconds everywhere",
    )

    result = verify_claims([valid_claim, hallucinated_claim], chunks)
    assert result.is_answerable is True
    assert result.claims_total == 2
    assert result.claims_verified == 1
    assert result.claims_dropped == 1
    assert "99.95%" in result.rebuilt_answer
    assert "10 seconds" not in result.rebuilt_answer
