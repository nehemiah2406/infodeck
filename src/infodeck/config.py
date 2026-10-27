"""Settings, read from the .env file (and environment variables).

The .env file is re-read every time get_settings() is called, so pasting an
API key into .env takes effect after a browser refresh, without a restart.
The API key is never included in repr()/str() and is masked in all logs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "anthropic"
    anthropic_api_key: str = field(default="", repr=False)
    claude_model: str = "claude-sonnet-5-5"
    claude_effort: str = "low"
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.8-flash"
    parser: str = "local"  # "local" (pdfplumber + local OCR) or "gemini" (page images read by Gemini)
    ocr_workers: int = 6  # scanned pages OCR'd in parallel (worker processes); 1 = no extra processes
    ocr_dpi: int = 200  # page image resolution for local OCR (150 lost accuracy in the evaluation)
    ollama_model: str = ""
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # "" = no re-ranking
    rerank_candidates: int = 20
    chroma_dir: Path = ROOT / "storage" / "chroma"
    parsed_dir: Path = ROOT / "storage" / "parsed"  # parsed text per page (also the parse cache)
    upload_dir: Path = ROOT / "data" / "uploads"
    sample_dir: Path = ROOT / "data" / "sample_docs"
    top_k: int = 5
    min_score: float = 0.55
    fact_match_score: float = 0.80  # exact-value answers need at least this similarity
    chunk_size: int = 800
    chunk_overlap: int = 150

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key.strip())


def get_settings() -> Settings:
    # Values in .env win over the shell environment; os.environ is not modified.
    env = {**os.environ, **{k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None}}

    def get(name: str, default: str) -> str:
        value = (env.get(name) or "").strip()
        return value or default

    return Settings(
        llm_provider=get("LLM_PROVIDER", "anthropic").lower(),
        anthropic_api_key=get("ANTHROPIC_API_KEY", ""),
        claude_model=get("CLAUDE_MODEL", "claude-sonnet-5-5"),
        claude_effort=get("CLAUDE_EFFORT", "low"),
        gemini_api_key=get("GEMINI_API_KEY", ""),
        gemini_model=get("GEMINI_MODEL", "gemini-3.8-flash"),
        parser=get("PARSER", "local").lower(),
        ocr_workers=max(1, int(get("OCR_WORKERS", "6"))),
        ocr_dpi=int(get("OCR_DPI", "200")),
        ollama_model=get("OLLAMA_MODEL", ""),
        ollama_host=get("OLLAMA_HOST", "http://localhost:11434"),
        embedding_model=get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model="" if get("RERANK_MODEL", "x").lower() in ("none", "off") else
                     get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        rerank_candidates=int(get("RERANK_CANDIDATES", "20")),
        chroma_dir=_resolve(get("CHROMA_DIR", "storage/chroma")),
        parsed_dir=_resolve(get("PARSED_DIR", "storage/parsed")),
        upload_dir=_resolve(get("UPLOAD_DIR", "data/uploads")),
        sample_dir=_resolve(get("SAMPLE_DIR", "data/sample_docs")),
        top_k=int(get("TOP_K", "5")),
        min_score=float(get("MIN_SCORE", "0.55")),
        fact_match_score=float(get("FACT_MATCH_SCORE", "0.80")),
        chunk_size=int(get("CHUNK_SIZE", "800")),
        chunk_overlap=int(get("CHUNK_OVERLAP", "150")),
    )


def _api_keys(s: Settings) -> list[str]:
    return [s.anthropic_api_key, s.gemini_api_key]


class SecretRedactingFilter(logging.Filter):
    """Replaces any API key that slips into a log message with '***'."""

    def __init__(self, secrets_getter=lambda: _api_keys(get_settings())):
        super().__init__()
        self._secrets_getter = secrets_getter

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = [s for s in self._secrets_getter() if s and len(s) >= 8]
        if secrets:
            message = record.getMessage()
            for s in secrets:
                message = message.replace(s, "***")
            record.msg, record.args = message, None
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once, with the API-key redaction filter attached."""
    root = logging.getLogger()
    if any(isinstance(f, SecretRedactingFilter) for h in root.handlers for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactingFilter())
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "urllib3", "huggingface_hub",
                  "google_genai", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
