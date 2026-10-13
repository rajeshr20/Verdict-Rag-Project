import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from groq import AuthenticationError, NotFoundError, RateLimitError
import httpx
from app.config import Settings
from app.core.generation.llm import GroqLLM, LLMError
from app.core.generation.schema import LLMResponse
from app.main import create_app


def test_vector_store_empty_gates_immediately_without_calling_llm(tmp_path):
    settings = Settings(
        LLM_PROVIDER="fake",
        EMBEDDING_MODEL="fake",
        CHROMA_DIR=str(tmp_path / "empty_chroma"),
        RETRIEVAL_MIN_SCORE=0.01,
    )
    test_app = create_app(custom_settings=settings)
    with TestClient(test_app) as client:
        fake_llm = test_app.state.llm
        initial_calls = fake_llm.call_count

        response = client.post("/ask", json={"question": "What is the policy?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answerable"] is False
        assert data["diagnostics"]["gated"] is True
        assert data["diagnostics"]["retrieved"] == 0
        # Ensure LLM was never called
        assert fake_llm.call_count == initial_calls


def test_llm_timeout_handled_gracefully(tmp_path):
    settings = Settings(
        LLM_PROVIDER="fake",
        EMBEDDING_MODEL="fake",
        CHROMA_DIR=str(tmp_path / "timeout_chroma"),
        RETRIEVAL_MIN_SCORE=0.001,
    )
    test_app = create_app(custom_settings=settings)
    with TestClient(test_app) as client:
        # Ingest one document so gate passes
        client.post(
            "/documents",
            files={"file": ("doc.txt", b"Document content about security.", "text/plain")},
        )

        fake_llm = test_app.state.llm
        fake_llm.set_error(LLMError("Read timed out after 30.0s"))

        response = client.post("/ask", json={"question": "What is the security rule?"})
        assert response.status_code == 503
        body = response.json()
        assert "LLM generation failed or timed out" in body["error"]
        assert "timed out" in body["error"]
        # Ensure clean JSON, no traceback leaked
        assert "Traceback" not in str(body)


def test_groq_valid_json_path():
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "answerable": True,
            "answer": "This is a valid answer.",
            "claims": [{"text": "Claim 1", "chunk_id": "c1", "quote": "Quote 1"}],
            "conflicts": [],
        })
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        llm = GroqLLM(api_key="test-mock-key", model_name="llama-3.3-70b-versatile")
        res = llm.generate("system", "user")

        assert res.answerable is True
        assert res.answer == "This is a valid answer."
        assert len(res.claims) == 1
        assert res.claims[0].quote == "Quote 1"
        assert mock_client.chat.completions.create.call_count == 1


def test_groq_retry_recovers_from_first_malformed_json():
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        c1 = MagicMock()
        c1.choices = [MagicMock()]
        c1.choices[0].message.content = "NOT VALID JSON {{"

        c2 = MagicMock()
        c2.choices = [MagicMock()]
        c2.choices[0].message.content = json.dumps({
            "answerable": True,
            "answer": "Recovered synthesized answer.",
            "claims": [],
            "conflicts": [],
        })

        mock_client.chat.completions.create.side_effect = [c1, c2]

        llm = GroqLLM(api_key="test-mock-key", model_name="llama-3.3-70b-versatile")
        res = llm.generate("system", "user")

        assert res.answerable is True
        assert res.answer == "Recovered synthesized answer."
        assert mock_client.chat.completions.create.call_count == 2


def test_groq_fails_after_two_malformed_json_attempts():
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].message.content = "corrupted {{"
        mock_client.chat.completions.create.return_value = c

        llm = GroqLLM(api_key="test-mock-key", model_name="llama-3.3-70b-versatile")
        with pytest.raises(LLMError) as exc_info:
            llm.generate("system", "user")

        assert "Groq generation failed after retry" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 2


def test_groq_rate_limit_retry_then_success():
    with patch("groq.Groq") as mock_groq_cls, patch("time.sleep") as mock_sleep:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        dummy_req = httpx.Request("POST", "https://api.groq.com")
        dummy_resp = httpx.Response(429, request=dummy_req)
        rate_err = RateLimitError("Rate limit reached", response=dummy_resp, body=None)

        success_comp = MagicMock()
        success_comp.choices = [MagicMock()]
        success_comp.choices[0].message.content = json.dumps({
            "answerable": True,
            "answer": "Answer after rate limit retry.",
            "claims": [],
            "conflicts": [],
        })

        mock_client.chat.completions.create.side_effect = [rate_err, success_comp]

        llm = GroqLLM(api_key="test-mock-key", model_name="llama-3.3-70b-versatile")
        res = llm.generate("system", "user")

        assert res.answerable is True
        assert res.answer == "Answer after rate limit retry."
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called()


def test_groq_authentication_error_does_not_leak_key():
    secret_key = "gsk_SUPER_SECRET_KEY_12345"
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        dummy_req = httpx.Request("POST", "https://api.groq.com")
        dummy_resp = httpx.Response(401, request=dummy_req)
        auth_err = AuthenticationError("Invalid API Key provided", response=dummy_resp, body=None)

        mock_client.chat.completions.create.side_effect = auth_err

        llm = GroqLLM(api_key=secret_key, model_name="llama-3.3-70b-versatile")
        with pytest.raises(LLMError) as exc_info:
            llm.generate("system", "user")

        err_msg = str(exc_info.value)
        assert "invalid api key" in err_msg.lower()
        assert secret_key not in err_msg


def test_groq_model_not_found_error_has_clear_message():
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        dummy_req = httpx.Request("POST", "https://api.groq.com")
        dummy_resp = httpx.Response(404, request=dummy_req)
        nf_err = NotFoundError("The model `non-existent-model` does not exist", response=dummy_resp, body=None)

        mock_client.chat.completions.create.side_effect = nf_err

        llm = GroqLLM(api_key="test-mock-key", model_name="non-existent-model")
        with pytest.raises(LLMError) as exc_info:
            llm.generate("system", "user")

        err_msg = str(exc_info.value)
        assert "not found" in err_msg.lower()
        assert "GROQ_MODEL" in err_msg
