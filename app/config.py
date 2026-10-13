from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    LLM_PROVIDER: Literal["groq", "fake"] = Field(
        default="groq",
        description="LLM Provider: 'groq' or 'fake'",
    )
    GROQ_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Groq (set in .env or environment)",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model name (e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant)",
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "Settings":
        """Fail fast at startup if Groq is enabled without an API key."""
        if self.LLM_PROVIDER == "groq":
            key = self.GROQ_API_KEY.get_secret_value() if isinstance(self.GROQ_API_KEY, SecretStr) else str(self.GROQ_API_KEY)
            if not key or not key.strip():
                raise ValueError(
                    "GROQ_API_KEY is required when LLM_PROVIDER='groq'. "
                    "Please set GROQ_API_KEY in your environment or .env file."
                )
        return self

    # Embeddings Settings
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model name for sentence-transformers, or 'fake' for hash-based vectors",
    )

    # Chunking Settings
    CHUNK_SIZE: int = Field(
        default=600,
        description="Target chunk size in characters",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        description="Overlap between adjacent chunks in characters",
    )

    # Retrieval Settings
    TOP_K: int = Field(
        default=6,
        description="Number of chunks to retrieve for Q&A",
    )
    RETRIEVAL_MIN_SCORE: float = Field(
        default=0.01,
        description="Minimum hybrid RRF score required to proceed to LLM generation (abstain threshold)",
    )

    # Persistence
    CHROMA_DIR: str = Field(
        default="./chroma_db",
        description="Directory path for persistent Chroma vector storage",
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
