from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single factual claim extracted from the answer with its supporting quote and chunk ID."""
    text: str = Field(
        ...,
        description="The statement or fact asserted in the answer.",
    )
    chunk_id: str = Field(
        ...,
        description="The ID of the chunk that provides direct evidence for this claim (e.g., 'c1').",
    )
    quote: str = Field(
        ...,
        description="A verbatim excerpt copied exactly from the referenced chunk text supporting the claim.",
    )


class Conflict(BaseModel):
    """A contradiction or policy discrepancy observed across two or more chunks."""
    topic: str = Field(
        ...,
        description="The specific subject matter of the contradiction (e.g., 'Meal Expense Allowance').",
    )
    chunk_ids: list[str] = Field(
        ...,
        description="List of chunk IDs that contain contradictory statements.",
    )
    description: str = Field(
        ...,
        description="Objective explanation of how the documents contradict each other.",
    )


class LLMResponse(BaseModel):
    """Structured response schema required from the LLM."""
    answerable: bool = Field(
        ...,
        description="True if the provided chunks contain sufficient evidence to answer the question, False otherwise.",
    )
    answer: str = Field(
        ...,
        description="Synthesized answer composed strictly of verifiable facts from the chunks.",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="List of all factual claims in the answer, each mapped to a chunk ID and verbatim quote.",
    )
    conflicts: list[Conflict] = Field(
        default_factory=list,
        description="List of any detected policy or factual contradictions between retrieved chunks.",
    )
