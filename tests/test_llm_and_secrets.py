"""Claude/Ollama wrappers with mocked clients, and API-key safety."""

import logging
from types import SimpleNamespace

import anthropic
import httpx2  # the HTTP library used by the anthropic SDK
import pytest

import infodeck.config as config
from infodeck.config import SecretRedactingFilter, Settings
from infodeck.llm import ClaudeLLM, LLMError, OllamaLLM

FAKE_KEY = "sk-ant-api03-THIS-IS-A-FAKE-KEY-0123456789"


class FakeStream:
    def __init__(self, pieces, stop_reason="end_turn", error=None):
        self.text_stream = iter(pieces)
        self._stop_reason = stop_reason
        self._error = error

    def __enter__(self):
        if self._error:
            raise self._error
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop_reason)


class FakeAnthropicClient:
    def __init__(self, stream):
        self.kwargs = None
        self._stream = stream
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._make))

    def _make(self, **kwargs):
        self.kwargs = kwargs
        return self._stream


def test_claude_streams_text_and_sends_expected_request():
    client = FakeAnthropicClient(FakeStream(["You get ", "18 days ", "[Leave.pdf, p.1]"]))
    llm = ClaudeLLM(FAKE_KEY, "claude-sonnet-5-5", "low", client=client)
    assert "".join(llm.stream("SYSTEM", "USER")) == "You get 18 days [Leave.pdf, p.1]"
    k = client.kwargs
    assert k["model"] == "claude-sonnet-5-5" and k["system"] == "SYSTEM"
    assert k["messages"] == [{"role": "user", "content": "USER"}]
    assert k["output_config"] == {"effort": "low"}
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]


def test_claude_refusal_becomes_friendly_error():
    llm = ClaudeLLM(FAKE_KEY, "m", client=FakeAnthropicClient(FakeStream(["partial"], stop_reason="refusal")))
    with pytest.raises(LLMError, match="declined"):
        list(llm.stream("s", "u"))


def test_claude_connection_error_is_friendly_and_never_contains_the_key():
    request = httpx2.Request("POST", "https://api.anthropic.com")
    error = anthropic.APIConnectionError(request=request)
    llm = ClaudeLLM(FAKE_KEY, "m", client=FakeAnthropicClient(FakeStream([], error=error)))
    with pytest.raises(LLMError) as info:
        list(llm.stream("s", "u"))
    assert "internet" in str(info.value)
    assert FAKE_KEY not in str(info.value)


def test_claude_bad_key_message():
    response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com"))
    error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
    llm = ClaudeLLM(FAKE_KEY, "m", client=FakeAnthropicClient(FakeStream([], error=error)))
    with pytest.raises(LLMError, match="not valid"):
        list(llm.stream("s", "u"))


def test_ollama_streams_and_wraps_errors():
    class FakeOllama:
        def __init__(self, fail=False):
            self.fail = fail

        def chat(self, **kwargs):
            if self.fail:
                raise ConnectionError("connection refused")
            return iter([{"message": {"content": "Hello "}}, {"message": {"content": "world"}}])

    assert "".join(OllamaLLM("h", "llama3.2:3b", client=FakeOllama()).stream("s", "u")) == "Hello world"
    with pytest.raises(LLMError, match="Ollama"):
        list(OllamaLLM("h", "llama3.2:3b", client=FakeOllama(fail=True)).stream("s", "u"))


def test_settings_repr_never_shows_the_key():
    s = Settings(anthropic_api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(s) and FAKE_KEY not in str(s)


def test_log_filter_masks_the_key(caplog):
    logger = logging.getLogger("test.secret")
    f = SecretRedactingFilter(lambda: [FAKE_KEY])
    logger.addFilter(f)
    try:
        with caplog.at_level(logging.INFO, logger="test.secret"):
            logger.info("calling API with key %s", FAKE_KEY)
    finally:
        logger.removeFilter(f)
    assert FAKE_KEY not in caplog.text and "***" in caplog.text


def test_settings_are_read_from_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=ollama\nANTHROPIC_API_KEY=\nOLLAMA_MODEL=llama3.2:3b\nTOP_K=7\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = config.get_settings()
    assert s.llm_provider == "ollama" and s.ollama_model == "llama3.2:3b" and s.top_k == 7
    assert not s.has_api_key
    env.write_text("ANTHROPIC_API_KEY=abc123456789\n")  # pasting a key works without restart
    assert config.get_settings().has_api_key
