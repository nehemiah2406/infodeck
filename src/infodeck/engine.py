"""The RAG pipeline: ingest documents, retrieve passages, produce answers.

Answer flow for a question:
  1. Retrieve the top-k most similar chunks.
  2. If even the best match is weak (score < MIN_SCORE): reply NOT_FOUND, no LLM call.
  3. If no LLM is available: "sources mode", return the best matching passages.
  4. Otherwise stream an answer from the LLM, grounded in the retrieved chunks.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Reranker
from .llm import LLMError, LLMStatus
from .loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError, load_document
from .prompts import NOT_FOUND, SOURCES_MODE_NOTE, SYSTEM_PROMPT, build_user_prompt
from .store import Passage, VectorStore, file_hash

log = logging.getLogger(__name__)

SOURCES_MODE_MAX_PASSAGES = 3
SAMPLES_MARKER = ".samples_loaded"


@dataclass
class IngestResult:
    status: str  # "added" | "replaced" | "duplicate" | "no_text" | "error"
    file_name: str
    message: str
    pages: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("added", "replaced", "duplicate")


@dataclass
class Answer:
    text: str
    mode: str  # "claude" | "ollama" | "sources" | "not_found" | "error"
    passages: list[Passage] = field(default_factory=list)  # shown under "Sources"
    note: str = ""


def safe_file_name(name: str) -> str:
    name = Path(name).name  # drop any folder parts
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" .")
    return name[:120] or "document"


def cited_passages(answer_text: str, passages: list[Passage]) -> list[Passage]:
    """Passages whose label ('file, p.N') appears in the answer. Falls back to all passages."""
    cited = [p for p in passages if re.search(re.escape(p.label) + r"(?!\d)", answer_text)]
    return cited or passages


def is_not_found(text: str) -> bool:
    return text.strip().lower().startswith(NOT_FOUND.lower().rstrip("."))


class InfoDeckEngine:
    def __init__(self, store: VectorStore, settings: Settings, reranker: Reranker | None = None):
        self.store = store
        self.settings = settings
        self.reranker = reranker

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: Path, display_name: str | None = None) -> IngestResult:
        path = Path(path)
        name = display_name or path.name
        try:
            doc_id = file_hash(path)[:16]
            if self.store.has_document(doc_id):
                return IngestResult("duplicate", name, f"{name} is already loaded (same content).")
            doc = load_document(path, name)
        except (UnsupportedFileError, ValueError) as e:
            return IngestResult("error", name, str(e))
        except Exception:
            log.exception("Failed to read %s", name)
            return IngestResult("error", name, f"{name} could not be read. Is the file damaged?")

        if not doc.has_text:
            msg = doc.warnings[0] if doc.warnings else f"{name} contains no text."
            return IngestResult("no_text", name, msg, pages=doc.page_count, warnings=doc.warnings)

        chunks = chunk_pages(doc.pages, name, self.settings.chunk_size, self.settings.chunk_overlap)
        old_versions = [d for d in self.store.find_by_name(name) if d != doc_id]
        for old in old_versions:  # same file name, new content: replace the old version
            self.remove_document(old, delete_file=False)
        n = self.store.add_document(doc_id, chunks, doc.page_count, str(path))
        status = "replaced" if old_versions else "added"
        verb = "Updated" if old_versions else "Added"
        log.info("%s %s: %d pages, %d chunks", verb, name, doc.page_count, n)
        return IngestResult(status, name, f"{verb} {name} ({doc.page_count} pages).",
                            pages=doc.page_count, chunks=n, warnings=doc.warnings)

    def ingest_folder(self, folder: Path) -> list[IngestResult]:
        files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        return [self.ingest_file(p) for p in files]

    def save_and_ingest_upload(self, file_name: str, data: bytes) -> IngestResult:
        """Save an uploaded file into the uploads folder and index it."""
        name = safe_file_name(file_name)
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return IngestResult("error", name, f"{name}: only PDF, DOCX and TXT files are supported.")
        upload_dir = self.settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / f".incoming_{name}"
        tmp.write_bytes(data)
        try:
            doc_id = file_hash(tmp)[:16]
            if self.store.has_document(doc_id):
                return IngestResult("duplicate", name, f"{name} is already loaded (same content).")
            final = upload_dir / f"{doc_id[:8]}_{name}"
            shutil.move(tmp, final)
            result = self.ingest_file(final, display_name=name)
            if not result.ok:
                final.unlink(missing_ok=True)
            return result
        finally:
            tmp.unlink(missing_ok=True)

    def remove_document(self, doc_id: str, delete_file: bool = True) -> None:
        docs = {d.doc_id: d for d in self.store.list_documents()}
        info = docs.get(doc_id)
        self.store.remove_document(doc_id)
        if delete_file and info:
            src = Path(info.source_path)
            try:  # only delete files we saved ourselves (never the sample docs)
                if src.resolve().is_relative_to(self.settings.upload_dir.resolve()):
                    src.unlink(missing_ok=True)
            except OSError:
                log.warning("Could not delete uploaded file for %s", info.file_name)

    def load_samples_once(self) -> list[IngestResult]:
        """Load the sample PDFs on the very first start (not again if the user removes them)."""
        marker = self.settings.chroma_dir / SAMPLES_MARKER
        if marker.exists() or not self.settings.sample_dir.exists():
            return []
        results = self.ingest_folder(self.settings.sample_dir)
        marker.write_text("Sample documents were loaded once. Delete this file to load them again.\n")
        return results

    # ------------------------------------------------------------------ #
    # Answering
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str) -> list[Passage]:
        """Top-k passages, or [] if even the best match is too weak.

        Vector search fetches candidates; the "not found" rule uses the best vector
        score; the optional re-ranker then re-orders the candidates before taking top-k.
        """
        n_candidates = max(self.settings.top_k, self.settings.rerank_candidates if self.reranker else 0)
        passages = self.store.search(question, n_candidates)
        if not passages or passages[0].score < self.settings.min_score:
            best = passages[0].score if passages else 0.0
            log.info("No strong match (best score %.3f < %.2f)", best, self.settings.min_score)
            return []
        if self.reranker:
            scores = self.reranker.score(question, [p.text for p in passages])
            passages = [p for _, p in sorted(zip(scores, passages), key=lambda x: -x[0])]
        return passages[: self.settings.top_k]

    def sources_answer(self, passages: list[Passage], reason: str = "") -> Answer:
        return Answer(SOURCES_MODE_NOTE, "sources", passages[:SOURCES_MODE_MAX_PASSAGES], note=reason)

    def stream_llm(self, status: LLMStatus, question: str, passages: list[Passage]) -> Iterator[str]:
        """Yield answer text pieces from the LLM. Raises LLMError with a friendly message."""
        yield from status.llm.stream(SYSTEM_PROMPT, build_user_prompt(question, passages))

    def finish_llm_answer(self, status: LLMStatus, text: str, passages: list[Passage]) -> Answer:
        if is_not_found(text):
            return Answer(NOT_FOUND, "not_found")
        return Answer(text.strip(), status.mode, cited_passages(text, passages))

    def answer(self, question: str, status: LLMStatus) -> Answer:
        """Non-streaming answer (used by scripts and tests)."""
        question = question.strip()
        if not question:
            return Answer("Please type a question.", "error")
        passages = self.retrieve(question)
        if not passages:
            return Answer(NOT_FOUND, "not_found")
        if status.llm is None:
            return self.sources_answer(passages, status.reason)
        try:
            text = "".join(self.stream_llm(status, question, passages))
        except LLMError as e:
            return self.sources_answer(passages, str(e))
        return self.finish_llm_answer(status, text, passages)
