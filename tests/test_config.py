import pytest
from pydantic import SecretStr
from app.config import LLMConfigError, Settings


def test_settings_loads_with_no_env_vars(monkeypatch):
    """Settings should instantiate safely with default values without raising any exception."""
    # Clear any environment variables that might be set
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    # Instantiate Settings without loading any .env file
    settings = Settings(_env_file=None)

    assert settings.LLM_PROVIDER == "groq"
    assert settings.GROQ_MODEL == "llama-3.3-70b-versatile"
    assert settings.GROQ_API_KEY.get_secret_value() == ""
    assert settings.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert settings.TOP_K == 6
    assert settings.CHUNK_SIZE == 600
    assert settings.CHUNK_OVERLAP == 100
    assert settings.RETRIEVAL_MIN_SCORE == 0.01
    assert settings.CHROMA_DIR == "./chroma_db"
    assert settings.LOG_LEVEL == "INFO"


def test_validate_for_llm_raises_for_groq_with_empty_key():
    """validate_for_llm() must raise LLMConfigError when provider is groq and key is missing."""
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY=SecretStr(""),
        _env_file=None,
    )
    with pytest.raises(LLMConfigError) as exc_info:
        settings.validate_for_llm()

    assert "GROQ_API_KEY is missing" in str(exc_info.value)


def test_validate_for_llm_succeeds_for_groq_with_valid_key():
    """validate_for_llm() must succeed without error when provider is groq and key is provided."""
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY=SecretStr("gsk_mock_valid_key_123"),
        _env_file=None,
    )
    # Should not raise
    settings.validate_for_llm()


def test_validate_for_llm_succeeds_for_fake_provider_with_no_key():
    """validate_for_llm() must succeed without error when provider is fake, even if key is empty."""
    settings = Settings(
        LLM_PROVIDER="fake",
        GROQ_API_KEY=SecretStr(""),
        _env_file=None,
    )
    # Should not raise
    settings.validate_for_llm()
