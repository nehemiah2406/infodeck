"""LLM providers (Claude API, Gemini API, Ollama) behind one small interface.

Every provider exposes stream(system, user) -> iterator of text pieces.
Errors are converted to LLMError with a friendly message that never contains
the API key. If no provider is usable, resolve_llm() returns None and the app
runs in "sources mode" (shows matching passages instead of AI answers).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """A friendly, user-facing error message (safe to show in the UI)."""


class LLM(Protocol):
    label: str

    def stream(self, system: str, user: str) -> Iterator[str]: ...


@dataclass
class LLMStatus:
    llm: LLM | None
    mode: str  # "claude" | "gemini" | "ollama" | "sources"
    label: str  # short text for the sidebar status line
    reason: str = ""  # why AI answers are off (sources mode only)


# --------------------------------------------------------------------------- #
# Claude (Anthropic API)
# --------------------------------------------------------------------------- #
class ClaudeLLM:
    def __init__(self, api_key: str, model: str, effort: str = "low", client=None):
        import anthropic

        self._anthropic = anthropic
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=60.0)
        self.model = model
        self.effort = effort
        self.label = f"Claude ({model})"

    def stream(self, system: str, user: str) -> Iterator[str]:
        a = self._anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            # Server-side fallback: if the model's safety filter declines, Anthropic
            # re-runs the request on a recommended fallback model automatically.
            with self._client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                raise LLMError("The AI declined to answer this question. Please rephrase it.")
        except LLMError:
            raise
        except a.AuthenticationError:
            raise LLMError("The Claude API key in .env is not valid. Check ANTHROPIC_API_KEY.") from None
        except a.PermissionDeniedError:
            raise LLMError("The Claude API key does not have permission to use this model.") from None
        except a.NotFoundError:
            raise LLMError(f"Claude model '{self.model}' was not found. Check CLAUDE_MODEL in .env.") from None
        except a.RateLimitError:
            raise LLMError("Too many requests to Claude right now. Please wait a minute and try again.") from None
        except a.BadRequestError as e:
            log.warning("Claude bad request (status %s)", e.status_code)
            raise LLMError("Claude could not process this request. Please try a shorter question.") from None
        except a.APIStatusError as e:
            log.warning("Claude API error (status %s)", e.status_code)
            raise LLMError("The Claude service is having problems. Please try again shortly.") from None
        except a.APIConnectionError:
            raise LLMError("Could not reach the Claude API. Check the internet connection.") from None


# --------------------------------------------------------------------------- #
# Gemini (Google AI API)
# --------------------------------------------------------------------------- #
class GeminiLLM:
    def __init__(self, api_key: str, model: str, client=None):
        from google import genai
        from google.genai import errors, types

        self._errors = errors
        self._types = types
        self._client = client or genai.Client(api_key=api_key)
        self.model = model
        self.label = f"Gemini ({model})"

    def stream(self, system: str, user: str) -> Iterator[str]:
        e_ = self._errors
        config = self._types.GenerateContentConfig(
            system_instruction=system, temperature=0.1, max_output_tokens=2048
        )
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=user, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except e_.ClientError as e:
            log.warning("Gemini client error (status %s)", e.code)
            if e.code in (401, 403) or (e.code == 400 and "API key" in (e.message or "")):
                raise LLMError("The Gemini API key in .env is not valid. Check GEMINI_API_KEY.") from None
            if e.code == 404:
                raise LLMError(f"Gemini model '{self.model}' was not found. Check GEMINI_MODEL in .env.") from None
            if e.code == 429:
                raise LLMError("Too many requests to Gemini right now. Please wait a minute and try again.") from None
            raise LLMError("Gemini could not process this request. Please try a shorter question.") from None
        except e_.APIError as e:
            log.warning("Gemini API error (status %s)", e.code)
            raise LLMError("The Gemini service is having problems. Please try again shortly.") from None
        except Exception as e:  # httpx connection / timeout errors
            log.warning("Gemini error: %s", type(e).__name__)
            raise LLMError("Could not reach the Gemini API. Check the internet connection.") from None


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #
class OllamaLLM:
    def __init__(self, host: str, model: str, client=None):
        import ollama

        self._client = client or ollama.Client(host=host, timeout=180)
        self.model = model
        self.label = f"Ollama ({model})"

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            for part in self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                stream=True,
                options={"temperature": 0.1},
            ):
                text = part["message"]["content"]
                if text:
                    yield text
        except Exception as e:  # ollama raises ResponseError / httpx errors
            log.warning("Ollama error: %s", type(e).__name__)
            raise LLMError("The local Ollama model is not responding. Is Ollama running?") from None


def _ollama_models(host: str) -> list[str]:
    import ollama

    resp = ollama.Client(host=host, timeout=2).list()
    return [m.model for m in resp.models]


# --------------------------------------------------------------------------- #
# Pick the provider from settings
# --------------------------------------------------------------------------- #
def resolve_llm(settings: Settings) -> LLMStatus:
    provider = settings.llm_provider
    if provider == "anthropic":
        if not settings.has_api_key:
            return LLMStatus(None, "sources", "Sources mode", "No Claude API key in .env yet.")
        try:
            llm = ClaudeLLM(settings.anthropic_api_key, settings.claude_model, settings.claude_effort)
        except Exception:
            log.exception("Could not create Claude client")
            return LLMStatus(None, "sources", "Sources mode", "The Claude client could not be started.")
        return LLMStatus(llm, "claude", llm.label)

    if provider == "gemini":
        if not settings.gemini_api_key.strip():
            return LLMStatus(None, "sources", "Sources mode", "No Gemini API key in .env yet.")
        try:
            llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model)
        except Exception:
            log.exception("Could not create Gemini client")
            return LLMStatus(None, "sources", "Sources mode", "The Gemini client could not be started.")
        return LLMStatus(llm, "gemini", llm.label)

    if provider == "ollama":
        if not settings.ollama_model:
            return LLMStatus(None, "sources", "Sources mode", "OLLAMA_MODEL is not set in .env.")
        try:
            models = _ollama_models(settings.ollama_host)
        except Exception:
            return LLMStatus(None, "sources", "Sources mode", "Ollama is not running on this computer.")
        wanted = settings.ollama_model
        if not any(m == wanted or m.split(":")[0] == wanted for m in models):
            return LLMStatus(None, "sources", "Sources mode",
                             f"Ollama model '{wanted}' is not downloaded (run: ollama pull {wanted}).")
        llm = OllamaLLM(settings.ollama_host, wanted)
        return LLMStatus(llm, "ollama", llm.label)

    return LLMStatus(None, "sources", "Sources mode",
                     f"Unknown LLM_PROVIDER '{provider}' in .env (use anthropic, gemini or ollama).")
