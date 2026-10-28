"""The RAG pipeline: ingest documents, retrieve passages, produce answers.

Answer flow for a question:
  1. Exact value: if the question matches a stored value or Q&A pair closely enough
     (score >= FACT_MATCH_SCORE), answer with it directly, no LLM call.
  2. Retrieve the top-k most similar chunks.
  3. If even the best match is weak (score < MIN_SCORE): reply NOT_FOUND, no LLM call.
  4. If no LLM is available: "sources mode", return the best matching passages.
  5. Otherwise stream an answer from the LLM, grounded in the retrieved chunks, then
     flag any number in the answer that does not appear in those chunks.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunker, chunk_pages
from .config import Settings
from .embeddings import Reranker
from .facts import Fact, FactExtractor, extract_facts, format_fact_answer, question_covered, symbol_in_question
from .gemini_parser import ParseError
from .jobs import IngestJob, IngestQueue, JobCancelled
from .llm import LLMError, LLMStatus
from .loaders import (HEADER_SAMPLE_PAGES, MIN_CHARS_PER_PAGE, SUPPORTED_EXTENSIONS, Page, Progress,
                      UnsupportedFileError, iter_pdf_pages, pdf_warnings, repeated_lines, strip_lines)
from .parsing import NUMBER_CHECK, is_partial, pages_to_recheck, parse_document, read_meta, write_meta
from .prompts import NOT_FOUND, SOURCES_MODE_NOTE, SYSTEM_PROMPT, build_user_prompt
from .store import Passage, VectorStore, file_hash

log = logging.getLogger(__name__)

SOURCES_MODE_MAX_PASSAGES = 3
SAMPLES_MARKER = ".samples_loaded"
COMMIT_EVERY_PAGES = 5  # while a big PDF loads, finished pages become searchable in batches of this size
FACT_AMBIGUITY_MARGIN = 0.02  # two facts this close in score with different values: not an exact match
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
BRACKETS_RE = re.compile(r"\[[^\[\]]*\]")  # citations like [file.pdf, p.3]


@dataclass
class IngestResult:
    status: str  # "added" | "replaced" | "duplicate" | "queued" | "loading" | "no_text" | "cancelled" | "error"
    file_name: str
    message: str
    pages: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    doc_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("added", "replaced", "duplicate")


@dataclass
class Answer:
    text: str
    mode: str  # "claude" | "gemini" | "ollama" | "exact" | "sources" | "not_found" | "error"
    passages: list[Passage] = field(default_factory=list)  # shown under "Sources"
    note: str = ""  # sources mode: why AI answers are off, or the AI error message
    ai_failed: bool = False  # sources mode because the AI answer failed (busy, timeout, quota...)
    fact: Fact | None = None  # the matched fact ("exact" mode)
    unverified: list[str] = field(default_factory=list)  # numbers in the answer not found in the sources
    loading: str = ""  # documents still being read when the question was asked


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


def unverified_numbers(answer: str, passages: list[Passage], question: str = "") -> list[str]:
    """Numbers in the answer that appear neither in the passages nor in the question."""
    sources = "\n".join(p.text for p in passages) + "\n" + question
    without_commas = sources.replace(",", "")
    missing: list[str] = []
    for n in NUMBER_RE.findall(BRACKETS_RE.sub(" ", answer)):
        n = n.rstrip(",.")
        if not n or n in missing:
            continue
        if re.search(rf"(?<![\d.,]){re.escape(n)}(?!\d)", sources) or \
                re.search(rf"(?<![\d.]){re.escape(n.replace(',', ''))}(?!\d)", without_commas):
            continue
        missing.append(n)
    return missing


class InfoDeckEngine:
    def __init__(self, store: VectorStore, settings: Settings, reranker: Reranker | None = None,
                 gemini_client=None):
        self.store = store
        self.settings = settings
        self.reranker = reranker
        self.gemini_client = gemini_client  # tests pass a fake; None = real client (PARSER=gemini)
        self.jobs = IngestQueue(self._run_job)  # uploads are read in the background, one at a time

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: Path, display_name: str | None = None,
                    progress: Progress | None = None) -> IngestResult:
        path = Path(path)
        name = display_name or path.name
        try:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise UnsupportedFileError(f"{name}: file type '{path.suffix}' is not supported.")
            doc_id = file_hash(path)[:16]
            if self.is_loaded(doc_id):
                return IngestResult("duplicate", name, f"{name} is already loaded (same content).")
            doc = parse_document(path, name, doc_id, self.settings, progress, self.gemini_client)
        except ParseError as e:
            return IngestResult("error", name, str(e))
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
        n_facts = self.store.add_facts(doc_id, extract_facts(doc.pages, name, doc_id))
        status = "replaced" if old_versions else "added"
        verb = "Updated" if old_versions else "Added"
        log.info("%s %s: %d pages, %d chunks, %d exact values", verb, name, doc.page_count, n, n_facts)
        return IngestResult(status, name, f"{verb} {name} ({doc.page_count} pages).",
                            pages=doc.page_count, chunks=n, warnings=doc.warnings)

    def ingest_folder(self, folder: Path) -> list[IngestResult]:
        files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        return [self.ingest_file(p) for p in files]

    def save_and_ingest_upload(self, file_name: str, data: bytes, progress: Progress | None = None) -> IngestResult:
        """Save an uploaded file into the uploads folder and index it."""
        name = safe_file_name(file_name)
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return IngestResult("error", name, f"{name}: only PDF, Word, Excel, PowerPoint, text and image "
                                               "files are supported.")
        upload_dir = self.settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / f".incoming_{name}"
        tmp.write_bytes(data)
        try:
            doc_id = file_hash(tmp)[:16]
            if self.is_loaded(doc_id):
                return IngestResult("duplicate", name, f"{name} is already loaded (same content).")
            final = upload_dir / f"{doc_id[:8]}_{name}"
            shutil.move(tmp, final)
            result = self.ingest_file(final, display_name=name, progress=progress)
            if not result.ok:
                final.unlink(missing_ok=True)
            return result
        finally:
            tmp.unlink(missing_ok=True)

    def queue_upload(self, file_name: str, data: bytes) -> IngestResult:
        """Save an uploaded file and queue it to be read in the background (one file at a time).

        The same content is never queued twice: a rerun, a browser refresh or a second upload
        of a file that is still being read just reports that it is already being read.
        """
        name = safe_file_name(file_name)
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return IngestResult("error", name, f"{name}: only PDF, Word, Excel, PowerPoint, text and image "
                                               "files are supported.")
        upload_dir = self.settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / f".incoming_{os.getpid()}_{name}"
        tmp.write_bytes(data)
        try:
            doc_id = file_hash(tmp)[:16]
            job = self.jobs.get(doc_id)
            if job and job.active:
                return IngestResult("loading", name, f"{name} is already being read.", doc_id=doc_id)
            if self.is_loaded(doc_id):
                return IngestResult("duplicate", name, f"{name} is already loaded (same content).", doc_id=doc_id)
            final = upload_dir / f"{doc_id[:8]}_{name}"
            os.replace(tmp, final)
            _, new = self.jobs.submit(doc_id, name, final)
            if not new:
                return IngestResult("loading", name, f"{name} is already being read.", doc_id=doc_id)
            return IngestResult("queued", name, f"{name} is queued to be read.", doc_id=doc_id)
        finally:
            tmp.unlink(missing_ok=True)

    def is_loaded(self, doc_id: str) -> bool:
        """True if the document is fully indexed.

        Chunks left over from an upload that was interrupted (app closed while reading) are
        removed, so the document is read again; pages read before come from the parse cache.
        """
        if not self.store.has_document(doc_id):
            return False
        job = self.jobs.get(doc_id)
        if job and job.active:
            return False
        if is_partial(self.settings, doc_id):
            log.info("Removing the half-indexed leftover of an interrupted upload (%s)", doc_id)
            self.store.remove_document(doc_id)
            return False
        return True

    def _run_job(self, job: IngestJob) -> IngestResult:
        """Read one queued upload (runs in the background thread). Never raises."""
        try:
            if self.settings.parser == "local" and job.path.suffix.lower() == ".pdf":
                result = self._ingest_pdf_streaming(job)
            else:
                result = self.ingest_file(job.path, job.file_name, progress=job.update)
                job.check_cancelled()
        except JobCancelled:
            self.store.remove_document(job.doc_id)
            result = IngestResult("cancelled", job.file_name, f"Stopped reading {job.file_name}.")
        except (ParseError, UnsupportedFileError, ValueError) as e:
            self.store.remove_document(job.doc_id)
            result = IngestResult("error", job.file_name, str(e))
        except Exception:
            log.exception("Failed to read %s", job.file_name)
            self.store.remove_document(job.doc_id)
            result = IngestResult("error", job.file_name, f"{job.file_name} could not be read. Is the file damaged?")
        if not result.ok:
            self._delete_upload(job.path)
        result.doc_id = job.doc_id
        return result

    def _ingest_pdf_streaming(self, job: IngestJob) -> IngestResult:
        """Read a PDF page by page (scans OCR'd in parallel) and index finished pages in small
        batches, so questions can be asked about the first pages while the rest is still read."""
        s, name, path, doc_id = self.settings, job.file_name, job.path, job.doc_id
        out = s.parsed_dir / doc_id
        meta = read_meta(out)
        if meta.get("parser") != "local":  # parsed before with another parser: start again
            shutil.rmtree(out, ignore_errors=True)
            meta = {}
        if meta.get("complete"):  # read completely before: index it from the parse cache
            stale = pages_to_recheck(out, meta)
            if not stale:
                return self.ingest_file(path, name, progress=job.update)
            # Read before broken OCR numbers ("10).6") were re-read: read just those pages again.
            log.info("Reading %d page(s) of %s again (broken OCR numbers)", len(stale), name)
            for n in stale:
                (out / f"page_{n}.md").unlink(missing_ok=True)
            write_meta(out, {**meta, "complete": False})
        out.mkdir(parents=True, exist_ok=True)
        known = {}  # pages read before the app was closed
        for f in out.glob("page_*.md"):
            known[int(f.stem.removeprefix("page_"))] = f.read_text(encoding="utf-8")

        chunker = Chunker(name, s.chunk_size, s.chunk_overlap)
        fact_extractor = FactExtractor(name, doc_id)
        old_versions = [d for d in self.store.find_by_name(name) if d != doc_id]
        all_pages: list[Page] = []
        kept: list[Page] = []
        ocr_pages: list[int] = []
        sample: list[Page] = []  # first pages, used to find repeated headers/footers
        batch: list[Page] = []
        repeated: set[str] | None = None
        total = 0

        def commit(pages: list[Page]) -> None:
            job.check_cancelled()
            for p in pages:
                (out / f"page_{p.number}.md").write_text(p.text, encoding="utf-8")
            pages = [p for p in pages if len(p.text.strip()) >= MIN_CHARS_PER_PAGE]
            if pages and old_versions and not kept:  # same name, new content: replace the old version
                for old in old_versions:
                    self.remove_document(old, delete_file=False)
            chunks = [c for p in pages for c in chunker.add_page(p)]
            facts = [f for p in pages for f in fact_extractor.add_page(p)]
            self.store.add_document(doc_id, chunks, total, str(path))
            self.store.add_facts(doc_id, facts, start=fact_extractor.count - len(facts))
            kept.extend(pages)

        for pdf_page in iter_pdf_pages(path, name, s.ocr_dpi, s.ocr_workers, known):
            job.check_cancelled()
            if not total:
                total = pdf_page.total
                write_meta(out, {"file_name": name, "parser": "local", "complete": False, "page_count": total})
            if pdf_page.ocr:
                ocr_pages.append(pdf_page.number)
            page = Page(pdf_page.number, pdf_page.text)
            all_pages.append(page)
            if repeated is None:
                sample.append(page)
                if len(sample) >= min(HEADER_SAMPLE_PAGES, total):
                    repeated = repeated_lines(sample)
                    for p in sample:
                        p.text = strip_lines(p.text, repeated)
                    commit(sample)
            else:
                page.text = strip_lines(page.text, repeated)
                batch.append(page)
                if len(batch) >= COMMIT_EVERY_PAGES:
                    commit(batch)
                    batch = []
            job.update(pdf_page.number, total)
        commit(batch)
        job.check_cancelled()

        warnings = pdf_warnings(name, all_pages, ocr_pages)
        write_meta(out, {"file_name": name, "parser": "local", "complete": True, "page_count": total,
                         "pages": [p.number for p in kept], "warnings": warnings, "number_check": NUMBER_CHECK})
        if not kept:
            msg = warnings[0] if warnings else f"{name} contains no text."
            return IngestResult("no_text", name, msg, pages=total, warnings=warnings)
        verb = "Updated" if old_versions else "Added"
        log.info("%s %s: %d pages, %d chunks, %d exact values", verb, name, total, chunker.count,
                 fact_extractor.count)
        return IngestResult("replaced" if old_versions else "added", name, f"{verb} {name} ({total} pages).",
                            pages=total, chunks=chunker.count, warnings=warnings)

    def loading_note(self) -> str:
        """'X is still loading (page 57 of 204)...' while documents are being read, else ''."""
        notes = []
        for job in self.jobs.active():
            if job.status == "running" and job.total:
                notes.append(f"{job.file_name} is still loading (page {job.done} of {job.total})")
            else:
                notes.append(f"{job.file_name} is waiting to be read")
        if not notes:
            return ""
        return "; ".join(notes) + ". Pages not read yet are not searched."

    def remove_document(self, doc_id: str, delete_file: bool = True) -> None:
        job = self.jobs.cancel(doc_id)  # stops a document that is still being read
        docs = {d.doc_id: d for d in self.store.list_documents()}
        info = docs.get(doc_id)
        self.store.remove_document(doc_id)
        if not delete_file:
            return
        if job and job.status == "running":
            return  # the job has the file open; it deletes the file itself when it stops
        if job:
            self._delete_upload(job.path)
        elif info:
            self._delete_upload(Path(info.source_path))

    def _delete_upload(self, src: Path) -> None:
        try:  # only delete files we saved ourselves (never the sample docs)
            if src.resolve().is_relative_to(self.settings.upload_dir.resolve()):
                src.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not delete uploaded file %s", src.name)

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
    def match_fact(self, question: str) -> tuple[Fact, float] | None:
        """The stored fact that answers the question exactly, or None if there is no safe match.

        A fact matches if the question names its symbol (e.g. "What is σra?"), or if its name is
        similar enough (score >= FACT_MATCH_SCORE) AND every content word of the question appears
        in the fact. Two close matches with different values are treated as no match.
        """
        tokens = [t for t in re.findall(r"[^\s?!,;:()]+", question) if len(t) >= 2]
        by_symbol = [f for f in self.store.facts_with_symbol(tokens) if symbol_in_question(question, f.symbol)]
        if by_symbol and len({(f.value, f.unit) for f in by_symbol}) == 1:
            return by_symbol[0], 1.0
        candidates = [(f, s) for f, s in self.store.search_facts(question, 5)
                      if s >= self.settings.fact_match_score and question_covered(question, f)]
        if not candidates:
            return None
        best, score = candidates[0]
        if any(score - s <= FACT_AMBIGUITY_MARGIN and (f.value, f.unit) != (best.value, best.unit)
               for f, s in candidates[1:]):
            log.info("Exact-value match is ambiguous for %r", question)
            return None
        return best, score

    def exact_answer(self, question: str) -> Answer | None:
        match = self.match_fact(question)
        if match is None:
            return None
        fact, score = match
        source = Passage(fact.line, fact.file_name, fact.page, fact.item_title, score, fact.doc_id)
        return Answer(format_fact_answer(fact), "exact", [source], fact=fact)

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

    def sources_answer(self, passages: list[Passage], reason: str = "", ai_failed: bool = False) -> Answer:
        return Answer(SOURCES_MODE_NOTE, "sources", passages[:SOURCES_MODE_MAX_PASSAGES], note=reason,
                      ai_failed=ai_failed)

    def stream_llm(self, status: LLMStatus, question: str, passages: list[Passage]) -> Iterator[str]:
        """Yield answer text pieces from the LLM. Raises LLMError with a friendly message."""
        yield from status.llm.stream(SYSTEM_PROMPT, build_user_prompt(question, passages))

    def finish_llm_answer(self, status: LLMStatus, text: str, passages: list[Passage],
                          question: str = "") -> Answer:
        if is_not_found(text):
            return Answer(NOT_FOUND, "not_found")
        return Answer(text.strip(), status.mode, cited_passages(text, passages),
                      unverified=unverified_numbers(text, passages, question))

    def answer(self, question: str, status: LLMStatus) -> Answer:
        """Non-streaming answer (used by scripts and tests)."""
        question = question.strip()
        if not question:
            return Answer("Please type a question.", "error")
        loading = self.loading_note()
        ans = self._answer(question, status)
        ans.loading = loading
        return ans

    def _answer(self, question: str, status: LLMStatus) -> Answer:
        exact = self.exact_answer(question)
        if exact:
            return exact
        passages = self.retrieve(question)
        if not passages:
            return Answer(NOT_FOUND, "not_found")
        if status.llm is None:
            return self.sources_answer(passages, status.reason)
        try:
            text = "".join(self.stream_llm(status, question, passages))
        except LLMError as e:
            return self.sources_answer(passages, str(e), ai_failed=True)
        return self.finish_llm_answer(status, text, passages, question)
