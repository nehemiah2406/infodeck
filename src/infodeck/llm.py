"""LLM providers (Claude API, Gemini API, Groq API, Ollama) behind one small interface.

Every provider exposes stream(system, user) -> iterator of text pieces.
Gemini and Groq are used through BackupChain: Gemini first, Groq when Gemini is busy or fails,
both within one time budget per question; the whole answer is returned at once so it can be
checked before it is shown.
Errors are converted to LLMError with a friendly message that never contains
the API key. If no provider is usable, resolve_llm() returns None and the app
runs in "sources mode" (shows matching passages instead of AI answers).
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)

# The Gemini SDK retries only server errors. 429 is handled in GeminiLLM.stream(): a used-up daily
# quota must fail at once (retrying it only made answers take 6-28 s), a short burst is retried.
GEMINI_SDK_RETRY_CODES = [500, 503]
GEMINI_SDK_ATTEMPTS = 3  # 1 try + 2 retries, waiting ~1 s and ~2 s in between
GEMINI_SDK_MIN_TIMEOUT = 10.0  # seconds; a server deadline under 10 s is rejected by Gemini (400)
GEMINI_RATE_LIMIT_RETRIES = 2
GEMINI_RATE_LIMIT_MAX_WAIT = 10.0  # seconds per wait
# A question waits at most GEMINI_MAX_WAIT_SECONDS (.env) for Gemini's first text (all retries
# included); then Groq answers (BackupChain) or the answer falls back to sources mode. Once text is
# arriving, only a pause longer than GEMINI_TIMEOUT_SECONDS (.env) or the end of the question's
# AI_BUDGET_SECONDS stops it.
GEMINI_BUSY = "Gemini is busy right now. Showing the matching passages instead. Please try again in a minute."
# Used-up Gemini quotas, kept until the app restarts: hash of (API key, model) -> message.
_used_up_quotas: dict[str, str] = {}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_BUSY = "Groq is busy right now. Showing the matching passages instead. Please try again in a minute."
GROQ_MIN_SECONDS = 1.0  # Groq is not started with less time than this left in the budget
# Gemini and Groq both failed (or there was no time left): the passages are shown with this note.
AI_BUSY = "AI is busy, showing the matching passages."
GEMINI_SKIP_AFTER_FAILURES = 2  # Gemini failures in a row before Groq answers directly
GEMINI_RETRY_EVERY = 5  # while skipped, Gemini is tried again on every 5th question
PROVIDER_NAMES = {"gemini": "Gemini", "groq": "Groq"}


def _quota_key(api_key: str, model: str) -> str:
    return hashlib.sha256(f"{api_key}|{model}".encode()).hexdigest()


class LLMError(Exception):
    """A friendly, user-facing error message (safe to show in the UI)."""


class LLM(Protocol):
    label: str

    def stream(self, system: str, user: str) -> Iterator[str]: ...


@dataclass
class LLMStatus:
    llm: LLM | None
    mode: str  # "claude" | "gemini" | "groq" | "ollama" | "sources"
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
class _NoReply(Exception):
    """Gemini sent nothing in time (raised by GeminiLLM._receive)."""


class GeminiLLM:
    def __init__(self, api_key: str, model: str, client=None, sleep: Callable[[float], None] = time.sleep,
                 timeout: float = 15.0, max_wait: float = 15.0):
        from google import genai
        from google.genai import errors, types

        self._errors = errors
        self._types = types
        # A busy server (500/503) is retried briefly; each request gives up after `timeout` seconds
        # without data (HttpOptions.timeout is in milliseconds). Gemini rejects a deadline under 10 s
        # with a 400 error, so the SDK never gets less; shorter waits (max_wait, the backup chain's
        # 5 s) are only our own clock in _receive().
        retry = types.HttpRetryOptions(attempts=GEMINI_SDK_ATTEMPTS, initial_delay=1.0, max_delay=2.0,
                                       http_status_codes=GEMINI_SDK_RETRY_CODES)
        sdk_seconds = max(timeout, GEMINI_SDK_MIN_TIMEOUT)
        http = types.HttpOptions(timeout=int(sdk_seconds * 1000), retry_options=retry)
        self._client = client or genai.Client(api_key=api_key, http_options=http)
        self.model = model
        self.label = f"Gemini ({model})"
        self.timeout = timeout
        self.max_wait = max_wait
        self._quota_key = _quota_key(api_key, model)
        self._sleep = sleep

    def stream(self, system: str, user: str, first_by: float | None = None,
               end_by: float | None = None) -> Iterator[str]:
        """first_by: time.monotonic() by which the first text must arrive (default: now + max_wait).
        end_by: the whole answer must be finished by then (the backup chain's budget; default: none)."""
        from .gemini_parser import quota_exhausted_message, suggested_delay

        e_ = self._errors
        config = self._types.GenerateContentConfig(
            system_instruction=system, temperature=0.1, max_output_tokens=2048
        )
        t0 = time.monotonic()
        deadline = first_by if first_by is not None else t0 + self.max_wait  # for the first text only

        def failed(reason: str) -> None:
            log.warning("Gemini failed after %.1f s: %s", time.monotonic() - t0, reason)

        for attempt in range(1, GEMINI_RATE_LIMIT_RETRIES + 2):
            started = False
            try:
                for text in self._receive(user, config, deadline, end_by):
                    started = True
                    yield text
                log.info("Gemini answered in %.1f s", time.monotonic() - t0)
                return
            except _NoReply as e:
                failed(str(e))
                raise LLMError(GEMINI_BUSY) from None
            except e_.ClientError as e:
                if e.code == 429 and not quota_exhausted_message(e) and not started \
                        and attempt <= GEMINI_RATE_LIMIT_RETRIES:
                    delay = suggested_delay(e)
                    delay = min(delay + 1 if delay is not None else 2.0 * attempt, GEMINI_RATE_LIMIT_MAX_WAIT)
                    if time.monotonic() + delay < deadline:  # a short burst: wait and retry
                        log.info("Gemini rate limit reached; waiting %.0f s before retrying (attempt %d)",
                                 delay, attempt)
                        self._sleep(delay)
                        continue
                failed(f"status {e.code}")
                if e.code in (401, 403) or (e.code == 400 and "API key" in (e.message or "")):
                    raise LLMError("The Gemini API key in .env is not valid. Check GEMINI_API_KEY.") from None
                if e.code == 404:
                    raise LLMError(f"Gemini model '{self.model}' was not found. Check GEMINI_MODEL in .env.") from None
                if e.code == 429:
                    quota = quota_exhausted_message(e)
                    if quota:  # no retry, and no more calls until the app restarts (see resolve_llm)
                        _used_up_quotas[self._quota_key] = quota
                        raise LLMError(quota) from None
                    # text was already shown, the last try, or waiting would pass the time limit
                    raise LLMError("Too many requests to Gemini right now. Please wait a minute and try again.") from None
                raise LLMError("Gemini could not process this request. Please try a shorter question.") from None
            except e_.ServerError as e:  # 500/503, already retried by the SDK
                failed(f"status {e.code}")
                raise LLMError(GEMINI_BUSY) from None
            except e_.APIError as e:
                failed(f"status {e.code}")
                raise LLMError("The Gemini service is having problems. Please try again shortly.") from None
            except Exception as e:  # httpx connection / timeout errors
                failed(type(e).__name__)
                if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__:
                    raise LLMError(GEMINI_BUSY) from None
                raise LLMError("Could not reach the Gemini API. Check the internet connection.") from None

    def _receive(self, user: str, config, deadline: float, end_by: float | None = None) -> Iterator[str]:
        """Gemini's text pieces, read in a background thread so a stuck request can't hang the app.

        Waits until `deadline` for the first piece, then at most self.timeout for each next piece, and
        never past `end_by`; raises _NoReply when that runs out. Errors of the request are re-raised
        here unchanged.
        """
        pieces: queue.Queue = queue.Queue()
        stop = threading.Event()

        def work() -> None:
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=self.model, contents=user, config=config
                ):
                    if stop.is_set():
                        return
                    if chunk.text:
                        pieces.put(("text", chunk.text))
                pieces.put(("end", None))
            except BaseException as e:
                pieces.put(("error", e))

        # daemon: a request we gave up on ends by itself (its own timeout) and never blocks shutdown
        threading.Thread(target=work, name="gemini-answer", daemon=True).start()
        first_wait = max(deadline - time.monotonic(), 0.0)
        first = True
        try:
            while True:
                wait = deadline - time.monotonic() if first else self.timeout
                if end_by is not None:
                    wait = min(wait, end_by - time.monotonic())
                try:
                    kind, value = pieces.get(timeout=max(wait, 0.0))
                except queue.Empty:
                    if end_by is not None and time.monotonic() >= end_by:
                        raise _NoReply("answer not finished within the AI time limit") from None
                    if first:
                        raise _NoReply(f"no reply within {first_wait:.1f} s") from None
                    raise _NoReply(f"no new text for {self.timeout:.0f} s") from None
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                first = False
                yield value
        finally:
            stop.set()


# --------------------------------------------------------------------------- #
# Groq (OpenAI-style chat API, called with httpx)
# --------------------------------------------------------------------------- #
def _call_before(fn: Callable[[], object], deadline: float, name: str):
    """fn() in a background thread; raises _NoReply if it has not finished by `deadline`."""
    box: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            box.put(("ok", fn()))
        except BaseException as e:
            box.put(("error", e))

    # daemon: a call we gave up on ends by itself (its own timeout) and never blocks shutdown
    threading.Thread(target=work, name=name, daemon=True).start()
    try:
        kind, value = box.get(timeout=max(deadline - time.monotonic(), 0.0))
    except queue.Empty:
        raise _NoReply("no reply within the time limit") from None
    if kind == "error":
        raise value
    return value


class GroqLLM:
    def __init__(self, api_key: str, model: str, client=None, timeout: float = 10.0):
        import httpx

        self._httpx = httpx
        self._api_key = api_key
        self._client = client  # tests pass a fake; None = a fresh httpx client per answer
        self.model = model
        self.label = f"Groq ({model})"
        self.timeout = timeout

    def __repr__(self) -> str:  # never shows the key
        return f"GroqLLM(model={self.model!r})"

    def answer(self, system: str, user: str, seconds: float | None = None) -> str:
        """The whole answer text, within `seconds` (default self.timeout). Raises LLMError."""
        seconds = self.timeout if seconds is None else seconds
        body = {"model": self.model, "temperature": 0.1, "max_completion_tokens": 2048,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"  # short thinking: answers in a few seconds
        t0 = time.monotonic()

        def failed(reason: str) -> None:
            log.warning("Groq failed after %.1f s: %s", time.monotonic() - t0, reason)

        def post():
            headers = {"Authorization": f"Bearer {self._api_key}"}
            if self._client is not None:
                return self._client.post(GROQ_URL, json=body, headers=headers, timeout=seconds)
            with self._httpx.Client() as client:
                return client.post(GROQ_URL, json=body, headers=headers, timeout=seconds)

        try:
            resp = _call_before(post, t0 + seconds, "groq-answer")
        except _NoReply as e:
            failed(str(e))
            raise LLMError(GROQ_BUSY) from None
        except Exception as e:  # httpx connection / timeout errors
            failed(type(e).__name__)
            if isinstance(e, self._httpx.TimeoutException):
                raise LLMError(GROQ_BUSY) from None
            raise LLMError("Could not reach the Groq API. Check the internet connection.") from None

        code = resp.status_code
        if code != 200:
            failed(f"status {code}")
            if code in (401, 403):
                raise LLMError("The Groq API key in .env is not valid. Check GROQ_API_KEY.")
            if code == 404:
                raise LLMError(f"Groq model '{self.model}' was not found. Check GROQ_MODEL in .env.")
            if code == 429 or code >= 500:
                raise LLMError(GROQ_BUSY)
            raise LLMError("Groq could not process this request. Please try a shorter question.")
        try:
            text = resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            text = ""
        if not text.strip():
            failed("empty answer")
            raise LLMError(GROQ_BUSY)
        log.info("Groq answered in %.1f s", time.monotonic() - t0)
        return text

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self.answer(system, user)


# --------------------------------------------------------------------------- #
# Gemini first, Groq as the backup
# --------------------------------------------------------------------------- #
@dataclass
class _GeminiHealth:
    failures_in_row: int = 0
    skipped: int = 0  # questions skipped since the last Gemini try


# Kept until the app restarts: hash of (Gemini API key, model) -> health.
_gemini_health: dict[str, _GeminiHealth] = {}
_health_lock = threading.Lock()


def _should_try_gemini(key: str) -> bool:
    with _health_lock:
        h = _gemini_health.setdefault(key, _GeminiHealth())
        if h.failures_in_row < GEMINI_SKIP_AFTER_FAILURES:
            return True
        h.skipped += 1
        if h.skipped >= GEMINI_RETRY_EVERY:  # time to see if Gemini works again
            h.skipped = 0
            return True
        return False


def _record_gemini(key: str, ok: bool) -> None:
    with _health_lock:
        h = _gemini_health.setdefault(key, _GeminiHealth())
        h.failures_in_row = 0 if ok else h.failures_in_row + 1
        h.skipped = 0


class BackupChain:
    """Gemini first (first text within `first_seconds`), then Groq with the time left of
    `budget_seconds`. answer() returns (text, "gemini" | "groq") or raises LLMError."""

    def __init__(self, gemini: GeminiLLM | None, groq: GroqLLM | None, first_seconds: float = 5.0,
                 budget_seconds: float = 10.0):
        self.gemini, self.groq = gemini, groq
        self.first_seconds, self.budget_seconds = first_seconds, budget_seconds
        if gemini and groq:
            self.label = f"{gemini.label} + {groq.label} backup"
        else:
            self.label = (gemini or groq).label

    def answer(self, system: str, user: str) -> tuple[str, str]:
        start = time.monotonic()
        end = start + self.budget_seconds
        errors: list[str] = []
        if self.gemini:
            key = self.gemini._quota_key
            if not self.groq or _should_try_gemini(key):  # without a backup, Gemini is always tried
                first_by = start + min(self.first_seconds, self.budget_seconds)
                try:
                    text = "".join(self.gemini.stream(system, user, first_by=first_by, end_by=end))
                    _record_gemini(key, True)
                    return text, "gemini"
                except LLMError as e:  # a half answer is dropped
                    _record_gemini(key, False)
                    errors.append(str(e))
                    if self.groq:
                        log.info("Gemini failed; asking Groq (%.1f s left)", end - time.monotonic())
            else:
                log.info("Gemini skipped (%d+ failures in a row); Groq answers", GEMINI_SKIP_AFTER_FAILURES)
        if self.groq:
            left = end - time.monotonic()
            if left >= GROQ_MIN_SECONDS:
                try:
                    return self.groq.answer(system, user, left), "groq"
                except LLMError as e:
                    errors.append(str(e))
            else:
                log.warning("No time left for Groq (%.1f s)", left)
        if self.gemini and self.groq:
            log.warning("Gemini and Groq both failed after %.1f s", time.monotonic() - start)
            raise LLMError(AI_BUSY)
        raise LLMError(errors[0] if errors else AI_BUSY)

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self.answer(system, user)[0]


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

    if provider in ("gemini", "groq"):
        groq = _groq(settings)
        gemini, why = None, "No Groq API key in .env yet."
        if provider == "gemini":
            gemini, why = _gemini(settings)
        if gemini is None and groq is None:
            return LLMStatus(None, "sources", "Sources mode", why)
        chain = BackupChain(gemini, groq, settings.gemini_max_wait_seconds, settings.ai_budget_seconds)
        return LLMStatus(chain, "gemini" if gemini else "groq", chain.label)

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
                     f"Unknown LLM_PROVIDER '{provider}' in .env (use anthropic, gemini, groq or ollama).")


def _gemini(settings: Settings) -> tuple[GeminiLLM | None, str]:
    """(Gemini client, "") or (None, why Gemini is off)."""
    if not settings.gemini_api_key.strip():
        return None, "No Gemini API key in .env yet."
    used_up = _used_up_quotas.get(_quota_key(settings.gemini_api_key, settings.gemini_model))
    if used_up:
        return None, f"{used_up} AI answers stay off until the app is restarted."
    try:
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model, timeout=settings.gemini_timeout_seconds,
                         max_wait=settings.gemini_max_wait_seconds), ""
    except Exception:
        log.exception("Could not create Gemini client")
        return None, "The Gemini client could not be started."


def _groq(settings: Settings) -> GroqLLM | None:
    if not settings.groq_api_key.strip():
        return None
    try:
        return GroqLLM(settings.groq_api_key, settings.groq_model, timeout=settings.ai_budget_seconds)
    except Exception:
        log.exception("Could not create Groq client")
        return None
