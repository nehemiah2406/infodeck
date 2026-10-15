"""Local embedding model (sentence-transformers). Text never leaves the laptop here."""

from __future__ import annotations

import logging
import os
from typing import Protocol

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger(__name__)

# bge models retrieve better when *questions* (not documents) get this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        try:  # use the downloaded copy first, so the app also works offline
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            log.info("Downloading embedding model %s (first run only)...", model_name)
            self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reads question + passage together and scores how well the passage answers it.

    More accurate than vector similarity alone, but slower, so it only re-orders
    the ~20 candidates that vector search found.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            log.info("Downloading re-ranking model %s (first run only)...", model_name)
            self.model = CrossEncoder(model_name)

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(question, t) for t in texts], show_progress_bar=False)]
