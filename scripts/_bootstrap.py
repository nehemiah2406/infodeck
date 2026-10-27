"""Shared setup for command-line scripts: makes `src/` importable and builds the engine."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infodeck.config import get_settings, setup_logging  # noqa: E402
from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder  # noqa: E402
from infodeck.engine import InfoDeckEngine  # noqa: E402
from infodeck.store import VectorStore  # noqa: E402


def build_engine() -> InfoDeckEngine:
    setup_logging()
    settings = get_settings()
    store = VectorStore(settings.chroma_dir, SentenceTransformerEmbedder(settings.embedding_model))
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank_model else None
    return InfoDeckEngine(store, settings, reranker)
