"""Shared test fixtures: a fake embedder and fake LLM, so tests need no model download,
no internet and no API key."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from infodeck.config import ROOT, Settings
from infodeck.engine import InfoDeckEngine
from infodeck.llm import LLMStatus
from infodeck.store import VectorStore

SAMPLE_DIR = ROOT / "data" / "sample_docs"
STOPWORDS = set("a an the is are do does i my me to of in on for and or what how many much can with be by at it this that".split())


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    dims = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in STOPWORDS:
                continue
            w = w.rstrip("s") if len(w) > 3 else w  # tiny stemming: days -> day
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    label = "Fake LLM"

    def __init__(self, reply: str = "Answer [KRT_Leave_Policy.pdf, p.1]", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def stream(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        for word in self.reply.split(" "):
            yield word + " "


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings(),
        anthropic_api_key="",
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        ocr_workers=1,
        parsed_dir=tmp_path / "parsed",
        parser="local",
        gemini_api_key="",
        sample_dir=SAMPLE_DIR,
        min_score=0.25,
    )


@pytest.fixture
def engine(settings) -> InfoDeckEngine:
    return InfoDeckEngine(VectorStore(settings.chroma_dir, FakeEmbedder()), settings)


@pytest.fixture
def loaded_engine(engine) -> InfoDeckEngine:
    engine.ingest_folder(SAMPLE_DIR)
    return engine


def llm_status(llm) -> LLMStatus:
    return LLMStatus(llm, "claude", llm.label)


SOURCES_STATUS = LLMStatus(None, "sources", "Sources mode", "No Claude API key in .env yet.")
