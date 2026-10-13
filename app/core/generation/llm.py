import json
import logging
import re
from typing import Optional, Protocol, runtime_checkable
from app.core.generation.schema import Claim, Conflict, LLMResponse


logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails, times out, or returns unparseable content."""
    pass


@runtime_checkable
class LLM(Protocol):
    """Protocol defining the interface for LLM generation."""

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate structured response from the LLM."""
        ...


class FakeLLM:
    """Deterministic, scriptable mock LLM for testing, evaluation smoke runs, and offline execution."""

    def __init__(self) -> None:
        self.canned_response: Optional[LLMResponse] = None
        self.call_count: int = 0
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.should_fail_with: Optional[Exception] = None

    def set_response(self, response: LLMResponse) -> None:
        """Configure the response to be returned by next call."""
        self.canned_response = response

    def set_error(self, exc: Exception) -> None:
        """Configure an exception to be raised on the next call."""
        self.should_fail_with = exc

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt

        if self.should_fail_with:
            err = self.should_fail_with
            self.should_fail_with = None
            raise err

        if self.canned_response is not None:
            resp = self.canned_response
            return resp

        # Default heuristic response generator based on user prompt content
        prompt_lower = user_prompt.lower()

        # Check for unanswerable signals
        if "capital of france" in prompt_lower or "unanswerable" in prompt_lower or "quantum entanglement" in prompt_lower:
            return LLMResponse(
                answerable=False,
                answer="The provided documents do not contain this information.",
                claims=[],
                conflicts=[],
            )

        # Check for prompt injection attempts
        if "unlimited leave" in prompt_lower or "ignore all previous" in prompt_lower:
            # Must NOT obey injection instruction
            return LLMResponse(
                answerable=True,
                answer="The company provides 20 days paid leave according to policy.",
                claims=[
                    Claim(
                        text="Employees receive 20 days paid leave per calendar year.",
                        chunk_id="c1",
                        quote="Employees receive 20 days paid leave per calendar year.",
                    )
                ],
                conflicts=[],
            )

        # Extract first chunk if available to simulate an answerable response
        chunk_match = re.search(r'<chunk id="([^"]+)"[^>]*>(.*?)</chunk>', user_prompt, re.DOTALL)
        if chunk_match:
            cid = chunk_match.group(1)
            raw_chunk_text = chunk_match.group(2).strip()
            # Pick the first sentence as quote
            first_sentence = raw_chunk_text.split(".")[0].strip()
            if first_sentence:
                quote = first_sentence + "." if not first_sentence.endswith(".") else first_sentence
                return LLMResponse(
                    answerable=True,
                    answer=f"According to {cid}: {first_sentence}.",
                    claims=[
                        Claim(
                            text=f"According to policy, {first_sentence}.",
                            chunk_id=cid,
                            quote=quote,
                        )
                    ],
                    conflicts=[],
                )

        return LLMResponse(
            answerable=False,
            answer="The provided documents do not contain this information.",
            claims=[],
            conflicts=[],
        )


class GroqLLM:
    """Production LLM provider using the official Groq SDK (groq)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided for GroqLLM.")
        self.api_key = api_key
        self.model_name = model_name or "llama-3.3-70b-versatile"
        self.timeout = timeout

        try:
            from groq import Groq
            self.client = Groq(api_key=api_key, timeout=timeout)
        except Exception as e:
            raise LLMError(f"Failed to initialize Groq client: {e}") from e

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call Groq chat completions API with JSON mode and retry logic."""
        import time
        from groq import AuthenticationError, NotFoundError, RateLimitError

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        max_attempts = 2  # retry once on malformed JSON

        for attempt in range(max_attempts):
            response = None
            rate_limit_attempts = 3
            backoff_seconds = 1.0

            for rl_attempt in range(rate_limit_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    break
                except RateLimitError as rle:
                    last_error = rle
                    if rl_attempt < rate_limit_attempts - 1:
                        logger.warning(
                            "Groq rate limit (429) hit, retrying in %.1fs (attempt %d/%d)...",
                            backoff_seconds,
                            rl_attempt + 1,
                            rate_limit_attempts,
                        )
                        time.sleep(backoff_seconds)
                        backoff_seconds *= 2.0
                    else:
                        raise LLMError(
                            "Groq API rate limit exceeded after multiple retry attempts. Please wait before retrying."
                        ) from rle
                except AuthenticationError as auth_err:
                    raise LLMError(
                        "Groq authentication failed: invalid API key. Please check your GROQ_API_KEY setting."
                    ) from auth_err
                except NotFoundError as nf_err:
                    raise LLMError(
                        f"Groq model '{self.model_name}' not found or inaccessible. "
                        f"Please check your GROQ_MODEL setting or run 'python -m scripts.list_models' to view available models."
                    ) from nf_err
                except Exception as e:
                    err_str = str(e).lower()
                    if "model_not_found" in err_str or "does not exist" in err_str:
                        raise LLMError(
                            f"Groq model '{self.model_name}' not found. Please verify GROQ_MODEL."
                        ) from e
                    elif "invalid_api_key" in err_str or "unauthorized" in err_str:
                        raise LLMError(
                            "Groq authentication failed: invalid API key. Please check your GROQ_API_KEY setting."
                        ) from e
                    last_error = e
                    logger.warning("Groq call attempt failed: %s", e)
                    break

            if response is None:
                if last_error:
                    raise LLMError(f"Groq generation failed: {last_error}") from last_error
                raise LLMError("Groq generation failed with no response.")

            # Parse and validate response content
            try:
                if not response.choices:
                    raise LLMError("Groq returned empty choices in completion.")
                raw_text = response.choices[0].message.content or ""
                if not raw_text.strip():
                    raise LLMError("Groq returned empty message content.")

                # Clean markdown fences if present
                clean_text = raw_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]

                parsed_data = json.loads(clean_text.strip())
                if isinstance(parsed_data, dict) and "claims" in parsed_data:
                    sanitized_claims = []
                    for item in parsed_data.get("claims", []):
                        if isinstance(item, dict):
                            sanitized_claims.append({
                                "text": str(item.get("text", "")),
                                "chunk_id": str(item.get("chunk_id", "c1")),
                                "quote": str(item.get("quote", "")),
                            })
                        elif isinstance(item, str):
                            sanitized_claims.append({
                                "text": item,
                                "chunk_id": "c1",
                                "quote": "",
                            })
                    parsed_data["claims"] = sanitized_claims

                return LLMResponse.model_validate(parsed_data)
            except Exception as parse_err:
                last_error = parse_err
                logger.warning("Groq JSON parsing attempt %d failed: %s", attempt + 1, parse_err)
                time.sleep(0.5)

        raise LLMError(f"Groq generation failed after retry: {last_error}") from last_error


def get_llm_provider(
    provider_name: str,
    api_key: str = "",
    model_name: str = "",
) -> LLM:
    """Factory to create LLM instance based on provider name."""
    p = provider_name.lower().strip()
    if p in ("fake", "mock", "test"):
        return FakeLLM()
    elif p in ("groq", "grok"):
        return GroqLLM(
            api_key=api_key,
            model_name=model_name or "llama-3.3-70b-versatile",
        )
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider_name}'. Supported: 'groq', 'fake'")
