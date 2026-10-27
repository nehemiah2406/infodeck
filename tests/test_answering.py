"""The "not found" rule, sources mode and LLM answers (LLM is always mocked)."""

from dataclasses import replace

from infodeck.config import Settings
from infodeck.engine import InfoDeckEngine, cited_passages
from infodeck.llm import LLMError, resolve_llm
from infodeck.prompts import NOT_FOUND, SOURCES_MODE_NOTE, SYSTEM_PROMPT
from infodeck.store import Passage
from tests.conftest import SOURCES_STATUS, FakeLLM, llm_status

QUESTION = "How many sick leave working days per year?"


# ---- "not found" rule ------------------------------------------------------ #
def test_weak_match_returns_not_found_without_calling_llm(loaded_engine):
    llm = FakeLLM()
    ans = loaded_engine.answer("zebra quantum volcano", llm_status(llm))
    assert ans.text == NOT_FOUND and ans.mode == "not_found"
    assert llm.calls == [], "the LLM must not be called when nothing relevant is found"


def test_threshold_controls_not_found(loaded_engine):
    llm = FakeLLM()
    strict = InfoDeckEngine(loaded_engine.store, replace(loaded_engine.settings, min_score=0.99))
    assert strict.answer(QUESTION, llm_status(llm)).mode == "not_found"
    assert llm.calls == []


def test_empty_library_returns_not_found(engine):
    assert engine.answer(QUESTION, llm_status(FakeLLM())).text == NOT_FOUND


def test_llm_saying_not_found_hides_sources(loaded_engine):
    ans = loaded_engine.answer(QUESTION, llm_status(FakeLLM(reply=NOT_FOUND)))
    assert ans.mode == "not_found" and ans.passages == []


def test_reranker_reorders_candidates_but_not_found_rule_uses_vector_score(loaded_engine):
    class ReverseReranker:
        def score(self, question, texts):
            return list(range(len(texts)))  # last vector candidate gets the highest score

    s = replace(loaded_engine.settings, top_k=5, rerank_candidates=20)
    plain = InfoDeckEngine(loaded_engine.store, s).retrieve(QUESTION)
    candidates = loaded_engine.store.search(QUESTION, 20)
    reranked = InfoDeckEngine(loaded_engine.store, s, ReverseReranker()).retrieve(QUESTION)
    assert len(reranked) == 5 and reranked[0].text == candidates[-1].text
    assert [p.text for p in plain] == [p.text for p in candidates[:5]]
    llm = FakeLLM()
    gated = InfoDeckEngine(loaded_engine.store, s, ReverseReranker())
    assert gated.answer("zebra quantum volcano", llm_status(llm)).mode == "not_found" and not llm.calls


# ---- LLM answers ------------------------------------------------------------- #
def test_llm_gets_system_prompt_and_labelled_context(loaded_engine):
    llm = FakeLLM()
    loaded_engine.answer(QUESTION, llm_status(llm))
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT and NOT_FOUND in system
    assert "[KRT_Leave_Policy.pdf, p." in user
    assert user.count("\n[KRT_") + user.startswith("<context>\n[") >= 1
    assert user.rstrip().endswith(f"Question: {QUESTION}")


def test_sources_show_only_cited_passages(loaded_engine):
    passages = loaded_engine.retrieve(QUESTION)
    target = passages[0]
    llm = FakeLLM(reply=f"You get 10 days [{target.label}].")
    ans = loaded_engine.answer(QUESTION, llm_status(llm))
    assert ans.mode == "claude"
    assert ans.passages and all(p.label == target.label for p in ans.passages)


def test_page_citation_match_is_exact():
    p1 = Passage("a", "doc.pdf", 1, "", 0.9, "x")
    p12 = Passage("b", "doc.pdf", 12, "", 0.8, "x")
    assert cited_passages("see [doc.pdf, p.12]", [p1, p12]) == [p12]


def test_llm_error_falls_back_to_sources_mode(loaded_engine):
    llm = FakeLLM(error=LLMError("Could not reach the Claude API."))
    ans = loaded_engine.answer(QUESTION, llm_status(llm))
    assert ans.mode == "sources" and ans.passages
    assert "Could not reach" in ans.note


# ---- sources mode -------------------------------------------------------------- #
def test_sources_mode_shows_passages_with_file_and_page(loaded_engine):
    ans = loaded_engine.answer(QUESTION, SOURCES_STATUS)
    assert ans.mode == "sources" and ans.text == SOURCES_MODE_NOTE
    assert 1 <= len(ans.passages) <= 3
    top = ans.passages[0]
    assert top.file_name == "KRT_Leave_Policy.pdf" and top.page >= 1
    assert top.label == f"KRT_Leave_Policy.pdf, p.{top.page}"
    assert "API key" in ans.note


def test_sources_mode_still_applies_not_found_rule(loaded_engine):
    assert loaded_engine.answer("zebra quantum volcano", SOURCES_STATUS).text == NOT_FOUND


def test_missing_api_key_means_sources_mode_not_a_crash(loaded_engine):
    status = resolve_llm(Settings(llm_provider="anthropic", anthropic_api_key=""))
    assert status.llm is None and status.mode == "sources"
    assert loaded_engine.answer(QUESTION, status).mode == "sources"


def test_ollama_not_running_means_sources_mode(monkeypatch):
    import infodeck.llm as llm_mod

    def down(host):
        raise ConnectionError("refused")

    monkeypatch.setattr(llm_mod, "_ollama_models", down)
    status = resolve_llm(Settings(llm_provider="ollama", ollama_model="llama3.2:3b"))
    assert status.mode == "sources" and "not running" in status.reason


def test_ollama_without_model_name_or_model_missing(monkeypatch):
    import infodeck.llm as llm_mod

    assert resolve_llm(Settings(llm_provider="ollama", ollama_model="")).mode == "sources"
    monkeypatch.setattr(llm_mod, "_ollama_models", lambda host: ["mistral:latest"])
    status = resolve_llm(Settings(llm_provider="ollama", ollama_model="llama3.2:3b"))
    assert status.mode == "sources" and "ollama pull" in status.reason
    monkeypatch.setattr(llm_mod, "_ollama_models", lambda host: ["llama3.2:3b"])
    assert resolve_llm(Settings(llm_provider="ollama", ollama_model="llama3.2:3b")).mode == "ollama"


def test_unknown_provider_means_sources_mode():
    status = resolve_llm(Settings(llm_provider="gpt"))
    assert status.mode == "sources" and "LLM_PROVIDER" in status.reason


def test_api_key_present_selects_claude_without_network_call():
    status = resolve_llm(Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test-123456789"))
    assert status.mode == "claude" and "claude-sonnet-5-5" in status.label


def test_gemini_without_key_means_sources_mode():
    status = resolve_llm(Settings(llm_provider="gemini", gemini_api_key=""))
    assert status.mode == "sources" and "Gemini" in status.reason


def test_gemini_key_present_selects_gemini_without_network_call():
    status = resolve_llm(Settings(llm_provider="gemini", gemini_api_key="AIza-test-123456789"))
    assert status.mode == "gemini" and "gemini-2.5-flash" in status.label
