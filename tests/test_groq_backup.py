"""Groq answers when Gemini is busy, times out or fails, within one 10 s budget per question.
Fake Gemini and Groq clients only: no real API is ever called."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

import infodeck.config as config
from infodeck.config import SecretRedactingFilter, Settings
from infodeck.engine import AI_UNCHECKED, cites_a_source
from infodeck.llm import (AI_BUSY, GEMINI_BUSY, GROQ_BUSY, GROQ_URL, BackupChain, GeminiLLM, GroqLLM, LLMError,
                          LLMStatus, resolve_llm)
from infodeck.ui import answered_by_html, sources_reason, trail_steps, Trace
from tests.test_gemini_timeout import DOC, FAKE_KEY, QUESTION, FailingGemini, HangingGemini, server_error

GROQ_KEY = "gsk_THIS-IS-A-FAKE-GROQ-KEY-0123456789"


class AnsweringGemini:
    def __init__(self, reply):
        self.models, self.reply, self.calls = self, reply, 0

    def generate_content_stream(self, model, contents, config):
        self.calls += 1
        yield SimpleNamespace(text=self.reply)


class FakeGroq:
    """Stands in for httpx.Client: post() returns a Groq-style reply, an error status, or raises."""

    def __init__(self, reply="", status=200, error=None, delay=0.0):
        self.reply, self.status, self.error, self.delay = reply, status, error, delay
        self.calls: list[dict] = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        time.sleep(self.delay)
        if self.error:
            raise self.error
        body = {"choices": [{"message": {"content": self.reply}}]} if self.status == 200 else \
            {"error": {"message": "nope"}}
        return httpx.Response(self.status, json=body, request=httpx.Request("POST", url))


@pytest.fixture
def bolts(engine, tmp_path):
    (tmp_path / "bolts.txt").write_text(DOC, encoding="utf-8")
    engine.ingest_file(tmp_path / "bolts.txt")
    return engine


def label(engine) -> str:
    return engine.retrieve(QUESTION)[0].label  # e.g. "bolts.txt"


def chain(gemini_client=None, groq_client=None, first=5.0, budget=10.0, gemini_timeout=15.0) -> LLMStatus:
    gemini = GeminiLLM(FAKE_KEY, "gemini-test", client=gemini_client, timeout=gemini_timeout) \
        if gemini_client else None
    groq = GroqLLM(GROQ_KEY, "openai/gpt-oss-120b", client=groq_client) if groq_client else None
    c = BackupChain(gemini, groq, first, budget)
    return LLMStatus(c, "gemini" if gemini else "groq", c.label)


# ---- Gemini fails -> Groq answers ------------------------------------------------------ #
def test_gemini_timeout_then_groq_answers_within_the_budget(bolts, caplog):
    caplog.set_level("INFO", logger="infodeck.llm")
    stuck = HangingGemini()
    groq = FakeGroq(f"The bolts are tightened to 540 N·m in a cross pattern [{label(bolts)}].")
    start = time.monotonic()
    try:
        ans = bolts.answer(QUESTION, chain(stuck, groq, first=0.5, budget=3.0))
    finally:
        stuck.release.set()
    seconds = time.monotonic() - start
    assert ans.mode == "groq" and ans.answered_by == "Groq" and "540 N·m" in ans.text
    assert ans.passages and ans.passages[0].file_name == "bolts.txt"
    assert seconds < 3.0
    assert 1.5 < groq.calls[0]["timeout"] < 2.6  # Groq got the time left of the 3 s budget (~2.5 s)
    assert "Gemini failed after 0.5 s" in caplog.text and "Groq answered in" in caplog.text


def test_gemini_503_and_halfway_stops_go_to_groq(bolts):
    reply = f"The bolts are tightened to 540 N·m [{label(bolts)}]."
    ans = bolts.answer(QUESTION, chain(FailingGemini(server_error(503)), FakeGroq(reply)))
    assert ans.mode == "groq" and ans.text == reply
    halfway = HangingGemini(first_texts=["The bolts are "])  # starts, then stops sending
    try:
        ans = bolts.answer(QUESTION, chain(halfway, FakeGroq(reply), first=0.5, budget=3.0, gemini_timeout=0.3))
    finally:
        halfway.release.set()
    assert ans.mode == "groq" and ans.text == reply  # the half Gemini answer is dropped


def test_healthy_gemini_answers_and_groq_is_never_called(bolts):
    gemini = AnsweringGemini(f"Tightened to 540 N·m [{label(bolts)}].")
    groq = FakeGroq("never used")
    ans = bolts.answer(QUESTION, chain(gemini, groq))
    assert ans.mode == "gemini" and ans.answered_by == "Gemini" and groq.calls == []
    assert answered_by_html(ans) == "<div class='id-reason'>Answered by Gemini</div>"
    assert ("Gemini wrote answer", "ok") in trail_steps(ans, Trace(retrieved=ans.passages))


def test_gemini_answer_that_takes_longer_than_the_budget_is_stopped():
    class Slow:
        def __init__(self):
            self.models = self

        def generate_content_stream(self, model, contents, config):
            for _ in range(20):
                time.sleep(0.2)
                yield SimpleNamespace(text="word ")

    start = time.monotonic()
    with pytest.raises(LLMError) as info:
        chain(Slow(), None, first=0.5, budget=1.0).llm.answer("s", "u")
    assert time.monotonic() - start < 1.6
    assert str(info.value) == GEMINI_BUSY


# ---- both fail -> passages ----------------------------------------------------------------- #
def test_both_fail_shows_the_passages_with_the_busy_note(bolts):
    ans = bolts.answer(QUESTION, chain(FailingGemini(server_error(503)), FakeGroq(status=503)))
    assert ans.mode == "sources" and ans.ai_failed and ans.passages[0].file_name == "bolts.txt"
    assert ans.note == AI_BUSY == "AI is busy, showing the matching passages."
    assert sources_reason(ans) == f"AI answer failed: {AI_BUSY}"


def test_groq_timeout_and_no_time_left_also_give_passages(bolts):
    stuck = HangingGemini()
    try:
        slow_groq = FakeGroq("too late", delay=5)
        start = time.monotonic()
        ans = bolts.answer(QUESTION, chain(stuck, slow_groq, first=0.3, budget=1.5))
        assert time.monotonic() - start < 2.5 and ans.note == AI_BUSY
        unused = FakeGroq("never used")  # Gemini used up the whole budget: Groq is not started
        ans = bolts.answer(QUESTION, chain(stuck, unused, first=1.0, budget=1.0))
        assert ans.note == AI_BUSY and unused.calls == []
    finally:
        stuck.release.set()


# ---- Groq answers get the same checks ----------------------------------------------------------- #
def test_groq_answer_that_fails_the_checks_is_not_shown(bolts, caplog):
    cite = f"[{label(bolts)}]"
    bad = {"wrong number": f"The bolts are tightened to 450 N·m {cite}.",  # 450 is not in the document
           "no source": "The bolts are tightened to 540 N·m in a cross pattern."}
    for why, reply in bad.items():
        ans = bolts.answer(QUESTION, chain(FailingGemini(server_error(503)), FakeGroq(reply)))
        assert ans.mode == "sources" and ans.ai_failed and ans.note == AI_UNCHECKED, why
        assert "450" not in ans.text and reply not in ans.text, why
    assert "Groq answer not shown, it failed the source check: 1 number(s) not in the passages" in caplog.text
    assert "Groq answer not shown, it failed the source check: no source cited" in caplog.text


def test_gemini_answers_get_the_same_strict_check(bolts):
    groq = FakeGroq("x")
    ans = bolts.answer(QUESTION, chain(AnsweringGemini(f"Tightened to 450 N·m [{label(bolts)}]."), groq))
    assert ans.mode == "sources" and ans.note == AI_UNCHECKED and groq.calls == []  # not passed on to Groq


def test_exact_value_card_still_comes_first_and_no_ai_is_called(engine, tmp_path):
    from tests.test_complex_docs import CALC_TEXT

    (tmp_path / "calc.txt").write_text(CALC_TEXT, encoding="utf-8")
    engine.ingest_file(tmp_path / "calc.txt")
    gemini, groq = AnsweringGemini("x"), FakeGroq("x")
    ans = engine.answer("What is the push-up distance?", chain(gemini, groq))
    assert ans.mode == "exact" and ans.text == "10.6 mm" and gemini.calls == 0 and groq.calls == []


def test_not_found_answers_stay_not_found(bolts):
    ans = bolts.answer(QUESTION, chain(FailingGemini(server_error(503)),
                                       FakeGroq("I could not find this in the documents.")))
    assert ans.mode == "not_found"


def test_citation_check_ignores_spaces():
    p = SimpleNamespace(label="KRT_Leave_Policy.pdf, p.2")
    assert cites_a_source("12 days [KRT_Leave_Policy.pdf, p. 2].", [p])
    assert not cites_a_source("12 days.", [p])


# ---- skipping Gemini after 2 failures ----------------------------------------------------------- #
def test_gemini_is_skipped_after_2_failures_and_tried_again_every_5_questions(bolts):
    reply = f"Tightened to 540 N·m [{label(bolts)}]."
    gemini = FailingGemini(server_error(503))
    calls = []
    real = gemini.generate_content_stream

    def counted(**kwargs):
        calls.append(1)
        return real(**kwargs)

    gemini.generate_content_stream = counted
    used = []
    for _ in range(12):
        ans = bolts.answer(QUESTION, chain(gemini, FakeGroq(reply)))
        assert ans.mode == "groq"
        used.append(len(calls))
    # questions 1-2 try Gemini (2 failures), 3-6 skip it, 7 tries again (fails), 8-11 skip, 12 tries
    assert used == [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4]


def test_gemini_working_again_is_used_again(bolts):
    reply = f"Tightened to 540 N·m [{label(bolts)}]."
    bad = FailingGemini(server_error(503))
    for _ in range(2):
        bolts.answer(QUESTION, chain(bad, FakeGroq(reply)))
    good = AnsweringGemini(reply)
    modes = [bolts.answer(QUESTION, chain(good, FakeGroq(reply))).mode for _ in range(6)]
    assert modes == ["groq"] * 4 + ["gemini", "gemini"]  # 5th question: Gemini works, back to normal


def test_gemini_without_a_backup_is_never_skipped(bolts):
    gemini = FailingGemini(server_error(503))
    for _ in range(4):
        ans = bolts.answer(QUESTION, chain(gemini, None))
        assert ans.note == GEMINI_BUSY  # its own message: there is no Groq to ask


# ---- Groq client ------------------------------------------------------------------------------- #
def test_groq_request_and_error_messages_never_contain_the_key(caplog):
    ok = FakeGroq("hello")
    assert GroqLLM(GROQ_KEY, "openai/gpt-oss-120b", client=ok).answer("sys", "user", 4) == "hello"
    call = ok.calls[0]
    assert call["url"] == GROQ_URL and call["timeout"] == 4
    assert call["headers"] == {"Authorization": f"Bearer {GROQ_KEY}"}
    assert call["json"]["model"] == "openai/gpt-oss-120b" and call["json"]["reasoning_effort"] == "low"
    assert [m["role"] for m in call["json"]["messages"]] == ["system", "user"]
    cases = [(FakeGroq(status=401), "Groq API key in .env is not valid"), (FakeGroq(status=404), "not found"),
             (FakeGroq(status=429), GROQ_BUSY), (FakeGroq(status=500), GROQ_BUSY),
             (FakeGroq(status=400), "could not process"), (FakeGroq(""), GROQ_BUSY),
             (FakeGroq(error=httpx.ReadTimeout("t")), GROQ_BUSY),
             (FakeGroq(error=httpx.ConnectError("c")), "Could not reach the Groq API")]
    for client, message in cases:
        with pytest.raises(LLMError) as info:
            GroqLLM(GROQ_KEY, "openai/gpt-oss-120b", client=client).answer("s", "u", 2)
        assert message in str(info.value) and GROQ_KEY not in str(info.value)
    assert GROQ_KEY not in caplog.text and GROQ_KEY not in repr(GroqLLM(GROQ_KEY, "m"))


def test_groq_key_is_masked_in_logs():
    f = SecretRedactingFilter(lambda: config._api_keys(replace(Settings(), groq_api_key=GROQ_KEY)))
    record = logging.LogRecord("x", logging.INFO, "", 0, "key %s", (GROQ_KEY,), None)
    f.filter(record)
    assert record.getMessage() == "key ***"


# ---- settings and choosing the providers ------------------------------------------------------ #
def test_groq_settings_and_defaults(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "AI_BUDGET_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    env.write_text("", encoding="utf-8")
    s = config.get_settings()
    assert (s.groq_api_key, s.groq_model, s.ai_budget_seconds) == ("", "openai/gpt-oss-120b", 10)
    env.write_text(f"GROQ_API_KEY={GROQ_KEY}\nGROQ_MODEL=llama-x\nAI_BUDGET_SECONDS=8\n", encoding="utf-8")
    s = config.get_settings()
    assert (s.groq_model, s.ai_budget_seconds) == ("llama-x", 8) and GROQ_KEY not in repr(s)
    example = (config.ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=\n" in example and "GROQ_MODEL=openai/gpt-oss-120b" in example
    assert "AI_BUDGET_SECONDS=10" in example


def test_gemini_sdk_deadline_is_never_under_10_seconds():
    """Gemini rejects a server deadline under 10 s (400). The 5 s limit is only our own clock."""
    for timeout in (1, 5, 9.9, 10, 15):
        llm = GeminiLLM(FAKE_KEY, "gemini-3.8-flash", timeout=timeout, max_wait=5)  # no request is sent
        assert llm._client._api_client._http_options.timeout >= 10_000, timeout
    status = resolve_llm(Settings(llm_provider="gemini", gemini_api_key=FAKE_KEY, groq_api_key=GROQ_KEY,
                                  gemini_timeout_seconds=3))
    sdk_ms = status.llm.gemini._client._api_client._http_options.timeout
    assert sdk_ms == 10_000 and status.llm.first_seconds == 5  # SDK 10 s, our own wait 5 s


def test_resolve_llm_builds_the_backup_chain():
    both = resolve_llm(Settings(llm_provider="gemini", gemini_api_key=FAKE_KEY, groq_api_key=GROQ_KEY))
    c = both.llm
    assert isinstance(c, BackupChain) and both.mode == "gemini" and c.gemini and c.groq
    assert (c.first_seconds, c.budget_seconds) == (5, 10)
    assert both.label == "Gemini (gemini-3.8-flash) + Groq (openai/gpt-oss-120b) backup"
    only_groq = resolve_llm(Settings(llm_provider="gemini", gemini_api_key="", groq_api_key=GROQ_KEY))
    assert only_groq.mode == "groq" and only_groq.llm.gemini is None
    assert resolve_llm(Settings(llm_provider="groq", groq_api_key=GROQ_KEY)).mode == "groq"
    off = resolve_llm(Settings(llm_provider="groq", groq_api_key=""))
    assert off.llm is None and "Groq" in off.reason


def test_used_up_gemini_quota_lets_groq_answer(monkeypatch):
    from infodeck import llm

    s = Settings(llm_provider="gemini", gemini_api_key=FAKE_KEY, groq_api_key=GROQ_KEY)
    monkeypatch.setitem(llm._used_up_quotas, llm._quota_key(FAKE_KEY, s.gemini_model), "Daily quota used up.")
    status = resolve_llm(s)
    assert status.mode == "groq" and status.llm.gemini is None
