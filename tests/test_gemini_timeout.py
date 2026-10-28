"""Gemini must never hang the app: slow, stuck or busy Gemini -> the sources card with a clear
message within ~15 s (GEMINI_MAX_WAIT_SECONDS). Fake Gemini clients only: the real Gemini API
is never called."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import infodeck.config as config
from infodeck.engine import Answer
from infodeck.config import Settings
from infodeck.llm import GEMINI_BUSY, GeminiLLM, LLMError, LLMStatus, resolve_llm
from infodeck.ui import Trace, skeleton_html, sources_reason, sources_tag, tag_html, trail_steps
from tests.conftest import FakeEmbedder

FAKE_KEY = "AIza-THIS-IS-A-FAKE-GEMINI-KEY-0123456789"
DOC = "The coupling bolts are tightened to 540 N·m in a cross pattern."
QUESTION = "How are the coupling bolts tightened?"


class HangingGemini:
    """A request that never answers (until the test releases it)."""

    def __init__(self, first_texts=()):
        self.models = self
        self.first_texts = first_texts
        self.release = threading.Event()

    def generate_content_stream(self, model, contents, config):
        for t in self.first_texts:
            yield SimpleNamespace(text=t)
        self.release.wait(60)
        yield SimpleNamespace(text=" (too late)")


class FailingGemini:
    def __init__(self, error):
        self.models, self.error = self, error

    def generate_content_stream(self, model, contents, config):
        raise self.error
        yield  # a generator, like the real SDK


class SlowButSteadyGemini:
    """A long answer: a piece every `gap` seconds."""

    def __init__(self, pieces, gap):
        self.models, self.pieces, self.gap = self, pieces, gap

    def generate_content_stream(self, model, contents, config):
        for p in self.pieces:
            time.sleep(self.gap)
            yield SimpleNamespace(text=p)


def server_error(code=503):
    from google.genai import errors

    return errors.ServerError(code, {"error": {"code": code, "message": "The model is overloaded."}})


# ---- settings and the real client ----------------------------------------------- #
def test_timeout_settings_have_safe_defaults_and_can_be_changed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("GEMINI_MAX_WAIT_SECONDS", raising=False)
    env.write_text("", encoding="utf-8")
    s = config.get_settings()
    assert (s.gemini_timeout_seconds, s.gemini_max_wait_seconds) == (15, 15)
    env.write_text("GEMINI_TIMEOUT_SECONDS=12\nGEMINI_MAX_WAIT_SECONDS=20\n", encoding="utf-8")
    s = config.get_settings()
    assert (s.gemini_timeout_seconds, s.gemini_max_wait_seconds) == (12, 20)
    example = (config.ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_TIMEOUT_SECONDS=15" in example and "GEMINI_MAX_WAIT_SECONDS=15" in example


def test_gemini_client_gets_a_request_timeout_and_short_retries():
    llm = GeminiLLM(FAKE_KEY, "gemini-3.8-flash")  # builds the client; no request is sent
    assert (llm.timeout, llm.max_wait) == (15, 15)
    options = llm._client._api_client._http_options
    assert options.timeout == 15_000  # milliseconds
    retry = options.retry_options
    assert retry.attempts == 3 and retry.http_status_codes == [500, 503] and retry.max_delay <= 2


def test_settings_reach_the_gemini_client():
    s = replace(Settings(), llm_provider="gemini", gemini_api_key=FAKE_KEY,
                gemini_timeout_seconds=12, gemini_max_wait_seconds=20)
    llm = resolve_llm(s).llm
    assert (llm.timeout, llm.max_wait) == (12, 20)
    assert llm._client._api_client._http_options.timeout == 12_000


# ---- the time limit ----------------------------------------------------------------- #
def test_gemini_that_never_answers_fails_fast_with_busy_message(caplog):
    client = HangingGemini()
    llm = GeminiLLM(FAKE_KEY, "m", client=client, max_wait=0.5)
    start = time.monotonic()
    try:
        with pytest.raises(LLMError) as info:
            list(llm.stream("s", "u"))
    finally:
        client.release.set()
    assert time.monotonic() - start < 2
    assert str(info.value) == GEMINI_BUSY
    assert "Gemini failed after 0.5 s: no reply within" in caplog.text  # time and reason in the log


def test_request_timeout_and_503_give_the_busy_message(caplog):
    for error, reason in [(httpx.ReadTimeout("timed out"), "ReadTimeout"), (server_error(503), "status 503"),
                          (server_error(500), "status 500")]:
        llm = GeminiLLM(FAKE_KEY, "m", client=FailingGemini(error))
        start = time.monotonic()
        with pytest.raises(LLMError, match="Gemini is busy right now") as info:
            list(llm.stream("s", "u"))
        assert time.monotonic() - start < 1
        assert FAKE_KEY not in str(info.value)
        assert f"Gemini failed after 0.0 s: {reason}" in caplog.text  # time and reason in the log


def test_answer_that_stops_halfway_is_not_used(engine, tmp_path):
    client = HangingGemini(first_texts=["The bolts are tightened to "])
    llm = GeminiLLM(FAKE_KEY, "m", client=client, timeout=0.3)
    (tmp_path / "bolts.txt").write_text(DOC, encoding="utf-8")
    engine.ingest_file(tmp_path / "bolts.txt")
    try:
        ans = engine.answer(QUESTION, LLMStatus(llm, "gemini", llm.label))
    finally:
        client.release.set()
    assert ans.mode == "sources" and ans.ai_failed and ans.note == GEMINI_BUSY
    assert "tightened to " not in ans.text  # the half answer is dropped, the passages are shown
    assert ans.passages[0].file_name == "bolts.txt"


def test_long_healthy_answer_is_not_cut_off_by_the_time_limit(caplog):
    caplog.set_level("INFO", logger="infodeck.llm")
    pieces = ["The ", "bolts ", "are ", "tightened ", "to ", "540 N·m."]
    llm = GeminiLLM(FAKE_KEY, "m", client=SlowButSteadyGemini(pieces, gap=0.15), timeout=1.0, max_wait=0.3)
    assert "".join(llm.stream("s", "u")) == "".join(pieces)  # took ~0.9 s, longer than max_wait
    assert "Gemini answered in" in caplog.text


def test_rate_limit_wait_never_passes_the_time_limit():
    from google.genai import errors

    burst = errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "8s"}]}})
    waits = []
    llm = GeminiLLM(FAKE_KEY, "m", client=FailingGemini(burst), sleep=waits.append, max_wait=5)
    with pytest.raises(LLMError, match="Too many requests"):
        list(llm.stream("s", "u"))
    assert waits == []  # waiting 9 s would pass the 5 s limit: give up at once


# ---- what the user sees ---------------------------------------------------------------- #
def test_503_gives_sources_answer_with_the_message(engine, tmp_path):
    (tmp_path / "bolts.txt").write_text(DOC, encoding="utf-8")
    engine.ingest_file(tmp_path / "bolts.txt")
    llm = GeminiLLM(FAKE_KEY, "m", client=FailingGemini(server_error(503)))
    ans = engine.answer(QUESTION, LLMStatus(llm, "gemini", llm.label))
    assert ans.mode == "sources" and ans.ai_failed and ans.passages
    assert sources_tag(ans) == "AI answer failed"
    assert sources_reason(ans) == f"AI answer failed: {GEMINI_BUSY}"
    assert trail_steps(ans, Trace(retrieved=ans.passages))[-1] == ("AI answer failed", "warn")


def test_sources_card_text_when_ai_is_simply_off():
    off = Answer("", "sources", note="No Gemini API key in .env yet.")
    assert sources_tag(off) == "AI off"
    assert sources_reason(off) == "AI answers are off, showing matching passages. No Gemini API key in .env yet."


# ---- the app itself (Streamlit AppTest, fake engine models, fake Gemini) ----------------- #
def run_app_with_stuck_gemini(tmp_path: Path, monkeypatch, max_wait: float | None):
    """Ask a question in the real app.py while Gemini never answers; returns (app, seconds).

    max_wait goes into the app's .env as GEMINI_MAX_WAIT_SECONDS (None: the default is used).
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import infodeck.embeddings as embeddings
    import infodeck.llm as llm_module
    from infodeck.engine import InfoDeckEngine
    from infodeck.store import VectorStore

    env = tmp_path / ".env"
    env.write_text("\n".join([
        "LLM_PROVIDER=gemini", f"GEMINI_API_KEY={FAKE_KEY}", "RERANK_MODEL=none", "MIN_SCORE=0.25",
        f"CHROMA_DIR={tmp_path / 'chroma'}", f"PARSED_DIR={tmp_path / 'parsed'}",
        f"UPLOAD_DIR={tmp_path / 'uploads'}", f"SAMPLE_DIR={tmp_path / 'no_samples'}"]
        + ([f"GEMINI_MAX_WAIT_SECONDS={max_wait}"] if max_wait is not None else [])), encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", lambda name: FakeEmbedder())
    s = config.get_settings()
    (tmp_path / "bolts.txt").write_text(DOC, encoding="utf-8")
    InfoDeckEngine(VectorStore(s.chroma_dir, FakeEmbedder()), s).ingest_file(tmp_path / "bolts.txt")

    client = HangingGemini()
    real_init = GeminiLLM.__init__

    def stuck_gemini(self, api_key, model, client_=None, **kwargs):  # the app's settings, a stuck client
        real_init(self, api_key, model, client=client, **kwargs)

    monkeypatch.setattr(llm_module.GeminiLLM, "__init__", stuck_gemini)
    st.cache_resource.clear()  # a fresh engine for this app run
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=s.gemini_max_wait_seconds + 30)
    try:
        at.run()
        assert not at.exception
        start = time.monotonic()
        at.chat_input[0].set_value(QUESTION).run()
        seconds = time.monotonic() - start
    finally:
        client.release.set()
        st.cache_resource.clear()
    return at, seconds


def page_html(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def check_sources_card(at) -> None:
    assert not at.exception
    html = page_html(at)
    assert "Gemini is busy right now. Showing the matching passages instead." in html  # the sources card
    assert "bolts.txt" in html  # with the matching passage
    assert skeleton_html() not in html  # the loading skeleton is gone
    assert tag_html("AI answer", "ai") not in html  # and so is the live "AI answer" card
    assert tag_html("AI answer failed", "warn") in html


def test_app_shows_sources_card_and_no_skeleton_when_gemini_is_stuck(tmp_path, monkeypatch):
    at, seconds = run_app_with_stuck_gemini(tmp_path, monkeypatch, max_wait=1.0)
    check_sources_card(at)
    assert seconds < 10


@pytest.mark.slow
def test_app_shows_sources_card_in_about_15_seconds_with_the_default_limit(tmp_path, monkeypatch):
    at, seconds = run_app_with_stuck_gemini(tmp_path, monkeypatch, max_wait=None)
    check_sources_card(at)
    print(f"sources card shown after {seconds:.1f} s")
    assert 15 <= seconds < 20  # waited the full default 15 s, then the card came quickly
