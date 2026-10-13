from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfigError(Exception):
    """Raised when configuration required specifically for LLM generation is missing."""
    pass


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

    def validate_for_llm(self) -> None:
        """Validate configuration required specifically when an LLM call is needed.

        Raises:
            LLMConfigError: If credentials or required parameters for the active provider are missing.
        """
        if self.LLM_PROVIDER == "groq":
            key = (
                self.GROQ_API_KEY.get_secret_value()
                if isinstance(self.GROQ_API_KEY, SecretStr)
                else str(self.GROQ_API_KEY)
            ).strip()
            if not key:
                raise LLMConfigError(
                    "GROQ_API_KEY is missing. "
                    "Please configure GROQ_API_KEY in your environment, Streamlit secrets, or .env file."
                )

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
