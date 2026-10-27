"""Read one page image with Gemini (PARSER=gemini) into faithful Markdown.

Only the page IMAGE is sent, never the PDF's own text layer, so a bad OCR layer
in a scanned PDF ("I I I" instead of 111) cannot leak into the parsed text.
Rate limits (HTTP 429) and temporary server errors are retried automatically,
waiting as long as Google asks (or with exponential back-off).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

BLANK_PAGE = "(blank page)"
MAX_ATTEMPTS = 8
MAX_WAIT_SECONDS = 120
LONG_WAIT_SECONDS = 600  # Google asking for a longer wait than this means a used-up (daily) quota

PARSE_PROMPT = f"""Transcribe this document page into Markdown, exactly as printed.

Rules:
- Copy every word, number, unit, symbol and formula exactly as shown, including Greek letters and
  superscripts (σ μ Θ π ², e.g. "σra = Pd / Pr x σr = 111 N/mm²"). Never summarise, translate,
  correct, calculate, convert or round anything.
- Tables: Markdown tables, with the header row, one table row per printed row.
- Formulas: plain text on one line, exactly as written.
- Keep numbered items ("9.", "(1)", "a)") with their numbers and text, in the printed order, and keep
  each item's sub-items and variable lines directly under it.
- Keep each question together with its answer.
- Diagrams, drawings and photos: one short line "[Diagram: what it shows; labels: ...]" listing the
  labels and dimensions printed on it.
- Include headers, stamps and handwriting only if they contain information.
- If the page is blank, reply with exactly: {BLANK_PAGE}
Reply with the Markdown only, no comments."""


class ParseError(Exception):
    """A friendly, user-facing parsing error (safe to show in the UI)."""


def suggested_delay(error) -> float | None:
    """Google's retryDelay in seconds, if the error carries one."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(getattr(error, "details", "")))
    return float(m.group(1)) if m else None


def quota_exhausted_message(error) -> str | None:
    """A message if the error is a used-up quota (e.g. the free tier's daily limit), not a short burst."""
    delay = suggested_delay(error)
    if "PerDay" in str(getattr(error, "details", "")) or (delay is not None and delay > LONG_WAIT_SECONDS):
        hours = f" (Google says: try again in about {delay / 3600:.0f} h)" if delay else ""
        return f"The Gemini quota for this API key is used up for today{hours}."
    return None


def _retry_delay(error, attempt: int) -> float:
    """Seconds to wait: Google's suggested retryDelay if present, else exponential back-off."""
    delay = suggested_delay(error)
    delay = delay + 1 if delay is not None else 5.0 * 2 ** (attempt - 1)
    return min(delay, MAX_WAIT_SECONDS)


def clean_markdown(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```[a-zA-Z]*\n(.*)\n```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    return "" if text == BLANK_PAGE else text


class GeminiPageReader:
    def __init__(self, api_key: str, model: str, client=None, sleep: Callable[[float], None] = time.sleep,
                 max_attempts: int = MAX_ATTEMPTS):
        from google import genai
        from google.genai import errors, types

        self._errors = errors
        self._types = types
        self._client = client or genai.Client(api_key=api_key)
        self.model = model
        self._sleep = sleep
        self.max_attempts = max_attempts

    def read(self, png: bytes, on_wait: Callable[[float, str], None] | None = None) -> str:
        """Markdown text of one page image. Raises ParseError."""
        e_, t = self._errors, self._types
        contents = [t.Part.from_bytes(data=png, mime_type="image/png"), PARSE_PROMPT]
        config = t.GenerateContentConfig(temperature=0.0, max_output_tokens=16384)
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                return clean_markdown(response.text or "")
            except e_.ClientError as e:
                if e.code == 429 and quota_exhausted_message(e):
                    raise ParseError(f"{quota_exhausted_message(e)} Pages read so far are saved: upload the "
                                     "file again after the quota resets to continue.") from None
                if e.code == 429:
                    delay, reason = _retry_delay(e, attempt), "Gemini rate limit reached"
                elif e.code in (401, 403) or (e.code == 400 and "API key" in (e.message or "")):
                    raise ParseError("The Gemini API key in .env is not valid. Check GEMINI_API_KEY.") from None
                elif e.code == 404:
                    raise ParseError(f"Gemini model '{self.model}' was not found. Check GEMINI_MODEL in .env.") from None
                else:
                    log.warning("Gemini parse error (status %s)", e.code)
                    raise ParseError(f"Gemini could not read this page (error {e.code}).") from None
            except e_.APIError as e:  # 5xx: temporary server problem
                delay, reason = _retry_delay(e, attempt), f"Gemini server busy (error {e.code})"
            except Exception as e:  # connection problems / timeouts
                log.warning("Gemini parse request failed: %s", type(e).__name__)
                delay, reason = _retry_delay(None, attempt), "Could not reach Gemini"
            if attempt == self.max_attempts:
                raise ParseError(f"{reason}; gave up after {self.max_attempts} attempts. "
                                 "Pages read so far are saved: upload the file again later to continue.")
            log.info("%s; waiting %.0f s before retrying (attempt %d)", reason, delay, attempt)
            if on_wait:
                on_wait(delay, reason)
            self._sleep(delay)
        raise ParseError("Gemini could not read this page.")  # not reached
