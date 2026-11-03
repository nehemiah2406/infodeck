"""InfoDeck - ask questions about your company documents.

Run with run.bat, or:  .venv\\Scripts\\python -m streamlit run app.py

Everything that runs the app is inside main(), called only when this file is not run as
"__mp_main__". Streamlit makes this file the main module, and on Windows every new OCR worker
process first re-runs the main module under that name: without the guard each worker started
a whole second InfoDeck (AI models, database, sample documents) before reading a single page.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from infodeck import APP_NAME  # noqa: E402
from infodeck.config import get_settings, setup_logging  # noqa: E402
from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder  # noqa: E402
from infodeck.engine import Answer, InfoDeckEngine  # noqa: E402
from infodeck.llm import BackupChain, LLMError, resolve_llm  # noqa: E402
from infodeck.prompts import NOT_FOUND  # noqa: E402
from infodeck.qa_log import log_answer  # noqa: E402
from infodeck.store import VectorStore  # noqa: E402
from infodeck.ui import (CardView, Trace, card_html, card_view, copy_button_html, esc, skeleton_html,  # noqa: E402
                         topbar_html)

log = logging.getLogger("infodeck.app")

HEADLINE = "Every answer, straight from your documents"
TAGLINE = "Ask questions about your company policies and files. Each answer cites the page it came from."
UPLOAD_HINT = "PDF, Word, Excel, PowerPoint, text or image files"
MAX_UPLOAD_MB = 200
SMALL_FILE_WAIT_SECONDS = 5  # after an upload, wait this long so quick files are ready at once
UPLOAD_TYPES = ["pdf", "docx", "txt", "xlsx", "csv", "pptx", "png", "jpg", "jpeg"]
SAMPLE_QUESTIONS = [
    "How many earned leave days do I get per year?",
    "What is the hotel limit per night in Mumbai?",
    "What should I do if I lose my laptop?",
]


# --------------------------------------------------------------------------- #
# Engine (shared by all users; created once per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_engine() -> InfoDeckEngine:
    setup_logging()
    settings = get_settings()
    store = VectorStore(settings.chroma_dir, SentenceTransformerEmbedder(settings.embedding_model))
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank_model else None
    engine = InfoDeckEngine(store, settings, reranker)
    for r in engine.load_samples_once():
        log.info("Sample document: %s", r.message)
    engine.self_test_unchecked()  # documents loaded before the exact-value self-test existed
    return engine


# --------------------------------------------------------------------------- #
# The answer card: answer line, file + page, exact quote, Copy
# --------------------------------------------------------------------------- #
def render_answer(ans: Answer, trace: Trace, idx: int, still_loading: bool) -> CardView:
    view = card_view(ans, trace.question, get_settings().min_score)
    with st.container(key=f"idcard_{view.kind}_{idx}"):
        st.markdown(card_html(view), unsafe_allow_html=True)
        if view.copy_text:
            st.html(copy_button_html("Copy", view.copy_text), width="content", unsafe_allow_javascript=True)
    if ans.loading and still_loading:  # once the documents are read, the old note would be wrong
        st.caption(f"⏳ {ans.loading}")
    return view


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def upload_errors(r) -> list[str]:
    """Messages of an IngestResult worth showing: only errors (no success or info messages)."""
    return [] if r.status in ("added", "replaced", "duplicate", "loading", "cancelled", "queued") else [r.message]


def job_text(job) -> str:
    if job.status == "running" and job.total:
        return f"Reading {job.file_name}: page {job.done} of {job.total}"
    if job.status == "running":  # started; the page count comes with the first page
        return f"Reading {job.file_name}..."
    return f"Waiting: {job.file_name} (files are read one at a time)"


def ingest_files(engine: InfoDeckEngine, files) -> list[str]:
    """Queue uploaded files to be read in the background, one at a time. Returns error messages.

    Quick files are waited for a few seconds so they are ready right away; big ones keep loading
    (progress in the sidebar) while the user can already ask about finished pages.
    """
    errors, queued = [], []
    for f in files:
        try:
            if f.size > MAX_UPLOAD_MB * 1024 * 1024:
                errors.append(f"{f.name} is larger than {MAX_UPLOAD_MB} MB.")
                continue
            r = engine.queue_upload(f.name, f.getvalue())
        except Exception:
            log.exception("Upload failed for %s", f.name)
            errors.append(f"{f.name} could not be added. Please try again.")
            continue
        if r.status == "queued":
            queued.append(r.doc_id)
        else:
            errors += upload_errors(r)
    if queued:
        bar = st.progress(0.0, text="Reading...")
        deadline = time.monotonic() + SMALL_FILE_WAIT_SECONDS
        while time.monotonic() < deadline and any(engine.jobs.get(d).active for d in queued):
            running = next((j for j in engine.jobs.active() if j.status == "running"), None)
            if running:
                bar.progress(running.done / max(running.total, 1), text=job_text(running))
            time.sleep(0.25)
        bar.empty()
        for d in queued:
            job = engine.jobs.get(d)
            if job.active:
                ss.watch.add(d)
            else:
                errors += upload_errors(job.result)
    return errors


@st.fragment(run_every=2)
def loading_panel(engine: InfoDeckEngine) -> None:
    """Live 'page X of N' progress for documents being read; refreshes the page when one finishes."""
    jobs = engine.jobs.active()
    for job in jobs:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        c1.progress(job.done / job.total if job.total else 0.0, text=job_text(job))
        if c2.button("✕", key=f"stop_{job.doc_id}", help=f"Stop reading {job.file_name} and remove it"):
            engine.remove_document(job.doc_id)
            st.rerun(scope="app")
    finished = [d for d in ss.watch if not engine.jobs.get(d).active]
    for d in finished:
        ss.watch.discard(d)
        result = engine.jobs.get(d).result
        if result is not None:
            ss.upload_errors = ss.upload_errors + upload_errors(result)
    # Refresh the whole page when a job starts, finishes, or its first pages become searchable.
    signature = (engine.jobs.finished_count,
                 tuple((j.doc_id, j.status, engine.store.has_document(j.doc_id)) for j in jobs))
    changed = ss.get("jobs_signature") not in (None, signature)
    ss.jobs_signature = signature
    if finished or changed:
        st.rerun(scope="app")


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #
def answer_question(engine: InfoDeckEngine, question: str, skeleton) -> tuple[Answer, Trace]:
    start = time.monotonic()
    trace = Trace(question=question, docs=len(docs))
    loading = engine.loading_note()  # documents still being read are only partly searchable
    ans = _answer_question(engine, question, skeleton, trace)
    ans.loading = loading
    trace.seconds = time.monotonic() - start
    return ans, trace


def _answer_question(engine: InfoDeckEngine, question: str, skeleton, trace: Trace) -> Answer:
    """Exact value if one matches; else retrieve, then answer with the LLM (streamed) or sources mode."""
    exact = engine.exact_answer(question)
    if exact:
        return exact
    status = resolve_llm(get_settings())
    passages = engine.retrieve(question)
    trace.retrieved = passages
    if not passages:
        return Answer(NOT_FOUND, "not_found")
    if status.llm is None:
        return engine.sources_answer(passages, status.reason)
    if isinstance(status.llm, BackupChain):  # Gemini / Groq: checked first, then shown (no live text)
        try:
            return engine.chain_answer(status, question, passages)
        except LLMError as e:
            return engine.sources_answer(passages, str(e), ai_failed=True)
        finally:
            skeleton.empty()

    errors: list[str] = []

    def stream():
        first = True
        try:
            for piece in engine.stream_llm(status, question, passages):
                if first:  # the skeleton makes way for the text as it is written
                    skeleton.empty()
                    first = False
                yield piece
        except LLMError as e:
            errors.append(str(e))

    live = st.empty()
    try:
        with live.container():
            with st.container(key="idcard_live"):
                text = st.write_stream(stream())
    finally:  # whatever happened, the loading card and live text make way for the final card
        skeleton.empty()
        live.empty()
    if errors:
        return engine.sources_answer(passages, errors[0], ai_failed=True)
    return engine.finish_llm_answer(status, text if isinstance(text, str) else "".join(text), passages, question)


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def main() -> None:
    global ss, docs  # used by the helpers above
    st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="centered",
                       initial_sidebar_state="expanded", menu_items={})
    st.markdown(f"<style>{(Path(__file__).parent / 'app.css').read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)
    try:
        with st.spinner("Starting InfoDeck... (the first start can take up to a minute)"):
            engine = get_engine()
    except Exception:
        log.exception("Start-up failed")
        st.error("InfoDeck could not start. Please check the terminal window for details and try again.")
        st.stop()

    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("uploader_key", 0)
    ss.setdefault("upload_errors", [])  # shown in the sidebar until the next question
    ss.setdefault("watch", set())  # documents this browser tab uploaded that are still being read

    st.markdown(topbar_html(APP_NAME), unsafe_allow_html=True)

    # Sidebar: logo, Upload, file types (+ progress and errors while uploading)
    with st.sidebar:
        st.markdown(f"<div class='id-brand'>{APP_NAME}</div>", unsafe_allow_html=True)
        files = st.file_uploader("Upload", type=UPLOAD_TYPES, accept_multiple_files=True,
                                 key=f"uploader_{ss.uploader_key}", label_visibility="collapsed")
        st.markdown(f"<div class='id-types'>{UPLOAD_HINT}</div>", unsafe_allow_html=True)
        if files:
            ss.upload_errors = ingest_files(engine, files)
            ss.uploader_key += 1  # clears the upload box
            st.rerun()
        for msg in ss.upload_errors:
            st.error(msg)
        loading_panel(engine)

    docs = engine.store.list_documents()
    still_loading = bool(engine.jobs.active())

    # Main area: chat
    welcome = st.empty()  # headline + example questions, only while the chat is empty
    with welcome.container():
        if not ss.messages:
            st.markdown(f"<div class='id-hero'><div class='id-headline'>{HEADLINE}</div>"
                        f"<div class='id-tagline'>{TAGLINE}</div></div>", unsafe_allow_html=True)
        if not docs:
            st.info("Click + in the chat box (or Upload in the sidebar) to add a PDF, Word, Excel, PowerPoint, "
                    "text or image file.")
        elif not ss.messages and any(d.file_name.startswith("KRT_") for d in docs):
            with st.container(key="samples", horizontal=True):
                for q in SAMPLE_QUESTIONS:
                    if st.button(q, key=f"sample_{q}"):
                        ss.pending_question = q
                        st.rerun()

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(esc(m["content"]), unsafe_allow_html=True)
            else:
                render_answer(m["answer"], m.get("trace") or Trace(), i, still_loading)

    typed = st.chat_input("Ask a question or attach a document...", accept_file="multiple",
                          file_type=UPLOAD_TYPES)
    if typed and typed.files:
        # Index the attached files first; a question typed with them is answered on the next run.
        ss.upload_errors = ingest_files(engine, typed.files)
        if typed.text.strip():
            ss.pending_question = typed.text.strip()
        st.rerun()

    question = ((typed.text if typed else "") or ss.pop("pending_question", "") or "").strip()
    if question and not docs and not engine.jobs.active():
        st.warning("Please add a document first (use the + button).")
    elif question:
        welcome.empty()
        ss.upload_errors = []  # upload errors are only shown while uploading
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(esc(question), unsafe_allow_html=True)
        with st.chat_message("assistant"):
            skeleton = st.empty()
            skeleton.markdown(skeleton_html(), unsafe_allow_html=True)
            try:
                ans, trace = answer_question(engine, question, skeleton)
            except Exception:
                log.exception("Answering failed")
                ans = Answer("Sorry, something went wrong while answering. Please try again.", "error")
                trace = Trace(question=question)
            skeleton.empty()
            view = render_answer(ans, trace, len(ss.messages), bool(ans.loading))
        ss.messages.append({"role": "assistant", "answer": ans, "trace": trace})
        log_answer(get_settings().qa_log_file, question, view, trace.seconds)


# An OCR worker process re-runs this file as "__mp_main__" (Windows "spawn"): it must not start the app.
if __name__ != "__mp_main__":
    main()
