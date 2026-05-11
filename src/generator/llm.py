"""Async Groq API client for LLM text generation."""

import asyncio
import logging
import os

import groq as _groq
from dotenv import load_dotenv
from groq import AsyncGroq

_log = logging.getLogger(__name__)

load_dotenv()

_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMError(RuntimeError):
    """Raised when the Groq API call fails or returns an unusable response.

    Wraps SDK-specific exceptions so callers never see raw Groq tracebacks.
    """


class GroqClient:
    """
    Async LLM client backed by the Groq API (LLaMA 3.1 70B).

    Instantiating this class does not make any network calls — the first
    call happens when :meth:`generate` is awaited.

    Error handling converts all SDK-level exceptions into :class:`LLMError`
    with a human-readable description of what went wrong.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Args:
            api_key: Groq API key.  Defaults to the ``GROQ_API_KEY``
                     environment variable.
            model: Groq model identifier.  Defaults to the ``LLM_MODEL``
                   environment variable, or ``llama-3.1-70b-versatile``.

        Raises:
            ValueError: If no API key is provided and ``GROQ_API_KEY`` is unset.
        """
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set.  "
                "Export it or add it to your .env file."
            )
        self._client = AsyncGroq(api_key=key)
        self.model: str = model or os.getenv("LLM_MODEL", _DEFAULT_MODEL)

    async def generate(self, prompt: str, system: str, history: list[dict] | None = None) -> str:
        """
        Send a chat completion request and return the model's text response.

        Args:
            prompt: User-turn message (context block + question).
            system: System-turn message (citation rules, persona).
            history: Prior conversation turns injected between system and user messages.

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMError: On rate limits, connection failures, HTTP errors, or an
                      empty response — always with a human-readable description.
        """
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        _MAX_RETRIES = 3
        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1500,
                )
                break
            except _groq.RateLimitError as exc:
                msg = str(exc)
                # Daily token limit — no point retrying within the same CI run
                if "tokens per day" in msg or "TPD" in msg:
                    raise LLMError(
                        f"Groq daily token limit (TPD) exhausted — run again tomorrow: {exc}"
                    ) from exc
                # Per-minute rate limit — back off and retry
                if attempt < _MAX_RETRIES - 1:
                    wait = 30 * (attempt + 1)
                    _log.warning(
                        "[LLM] TPM rate limit hit — retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise LLMError(
                    f"Groq rate limit exceeded after {_MAX_RETRIES} attempts: {exc}"
                ) from exc
            except _groq.APIConnectionError as exc:
                raise LLMError(f"Could not connect to Groq API: {exc}") from exc
            except _groq.APIStatusError as exc:
                raise LLMError(
                    f"Groq API returned HTTP {exc.status_code}: {exc.message}"
                ) from exc
            except Exception as exc:
                raise LLMError(
                    f"Unexpected error from Groq ({type(exc).__name__}): {exc}"
                ) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Groq returned an empty response.")
        return content
