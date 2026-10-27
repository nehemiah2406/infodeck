"""InfoDeck - ask questions about your company documents.

Run with run.bat, or:  .venv\\Scripts\\python -m streamlit run app.py
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
from infodeck.llm import LLMError, resolve_llm  # noqa: E402
from infodeck.preview import can_preview, page_preview  # noqa: E402
from infodeck.prompts import NOT_FOUND, SOURCES_MODE_NOTE  # noqa: E402
from infodeck.store import Passage, VectorStore  # noqa: E402
from infodeck.ui import (AI_MODES, Trace, card_head_html, cited_html, confidence, copy_button_html,  # noqa: E402
                         copy_label, copy_text, esc, exact_html, follow_ups, has_numbers, is_quota,
                         not_found_html, preview_skeleton_html, skeleton_html, snippet, source_cards_html,
                         source_line_html, tag_html, topbar_html, trail_html, trail_steps, unique_sources)

log = logging.getLogger("infodeck.app")

HEADLINE = "Every answer, straight from your documents"
TAGLINE = "Ask questions about your company policies and files. Each answer cites the page it came from."
MAX_UPLOAD_MB = 200
SMALL_FILE_WAIT_SECONDS = 5  # after an upload, wait this long so quick files show their result at once
UPLOAD_TYPES = ["pdf", "docx", "txt", "xlsx", "csv", "pptx", "png", "jpg", "jpeg"]
SAMPLE_QUESTIONS = [
    "How many earned leave days do I get per year?",
    "What is the hotel limit per night in Mumbai?",
    "What should I do if I lose my laptop?",
]

st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="centered",
                   initial_sidebar_state="expanded", menu_items={})
st.markdown(f"<style>{(Path(__file__).parent / 'app.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

deferred = []  # page previews to load after everything else on the page (they must not slow answers down)


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
    return engine


def ai_pill(status) -> tuple[str, str, str]:
    """(label, tone, tooltip) for the AI status pill in the top bar."""
    if status.llm is None:
        return ("Quota used up" if is_quota(status.reason) else "AI off"), "warn", status.reason
    return f"{AI_MODES.get(status.mode, status.label)} ready", "ok", status.label


def render_topbar(slot, engine: InfoDeckEngine) -> None:
    docs_now = engine.store.list_documents()
    pages = 0
    for d in docs_now:
        job = engine.jobs.get(d.doc_id)
        pages += job.done if job and job.active else d.pages  # a document still loading: pages read so far
    label, tone, tip = ai_pill(resolve_llm(get_settings()))
    slot.markdown(topbar_html(APP_NAME, len(docs_now), pages, label, tone, tip), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Answer cards
# --------------------------------------------------------------------------- #
def render_passages(passages: list[Passage]) -> None:
    for p in passages:
        section = f" · {esc(p.section)}" if p.section else ""
        st.markdown(f"<div class='id-source'>{esc(p.label)}{section} · relevance {p.score:.2f}</div>"
                    f"<div class='id-passage'>{esc(p.text)}</div>", unsafe_allow_html=True)


def source_path(doc_id: str) -> str:
    return next((d.source_path for d in docs if d.doc_id == doc_id), "")


def preview_caption(file_name: str, page: int) -> str:
    where = f" · p. {page}" if page else ""
    return f"<div class='id-prevcap'><b>{esc(file_name)}</b>{where}</div>"


def show_preview(slot, doc_id: str, page: int, target: str, file_name: str, cached_only: bool) -> bool:
    """Fill the slot with the page crop. Returns False if it is not cached yet (cached_only)."""
    key = (doc_id, page, target)
    pv = None
    if key not in ss.preview_failed:
        try:
            pv = page_preview(source_path(doc_id), page, target, get_settings().parsed_dir / doc_id, cached_only)
        except Exception:
            log.warning("Page preview failed for %s p.%s", file_name, page, exc_info=True)
            ss.preview_failed.add(key)
        if pv is None and cached_only and key not in ss.preview_failed:
            return False
    with slot.container():
        if pv:
            st.image(pv.crop, width="stretch")
        st.markdown(preview_caption(file_name, page), unsafe_allow_html=True)
    return True


@st.dialog("Source page", width="large")
def open_page(doc_id: str, page: int, target: str, file_name: str) -> None:
    try:
        pv = page_preview(source_path(doc_id), page, target, get_settings().parsed_dir / doc_id)
    except Exception:
        log.warning("Page preview failed for %s p.%s", file_name, page, exc_info=True)
        pv = None
    if pv:
        st.image(pv.full, width="stretch")
        st.caption(f"{file_name}" + (f", page {page}" if page else "") +
                   (" · the matched line is boxed in green" if pv.boxed else ""))
    else:
        st.info("The page picture is not available for this file.")


def action_row(ans: Answer, idx: int, target: Passage | None, line: str, passages: bool = True) -> None:
    with st.container(key=f"idact_{idx}", horizontal=True):
        st.html(copy_button_html(copy_label(ans), copy_text(ans)), width="content", unsafe_allow_javascript=True)
        if target is not None:
            ok = can_preview(source_path(target.doc_id), target.page)
            if st.button("Open page", key=f"open_{idx}", disabled=not ok,
                         help=None if ok else "No page picture for this file type"):
                open_page(target.doc_id, target.page, line, target.file_name)
        if passages and ans.passages:
            with st.popover("Passages", key=f"pp_{idx}"):
                render_passages(ans.passages)
        st.feedback("thumbs", key=f"fb_{idx}")


def render_exact(ans: Answer, idx: int, level: str) -> None:
    f = ans.fact
    with st.container(key=f"idcard_exact_{idx}"):
        st.markdown(card_head_html([tag_html("Exact value", "ok")], level), unsafe_allow_html=True)
        if can_preview(source_path(f.doc_id), f.page):
            left, right = st.columns([1.55, 1], gap="medium")
            left.markdown(exact_html(ans), unsafe_allow_html=True)
            with right.container(key=f"idprev_{idx}"):
                slot = st.empty()
            if not show_preview(slot, f.doc_id, f.page, f.line, f.file_name, cached_only=True):
                slot.markdown(preview_skeleton_html(), unsafe_allow_html=True)
                deferred.append(lambda: show_preview(slot, f.doc_id, f.page, f.line, f.file_name, cached_only=False))
        else:
            st.markdown(exact_html(ans) + preview_caption(f.file_name, f.page), unsafe_allow_html=True)
        action_row(ans, idx, ans.passages[0] if ans.passages else None, f.line)


def render_ai(ans: Answer, idx: int, level: str, trace: Trace) -> None:
    sources = unique_sources(ans.passages)
    tags = [tag_html("AI answer", "ai")]
    if has_numbers(ans.text):
        tags.append(tag_html("Check value in source", "warn") if ans.unverified else tag_html("Numbers verified", "ok"))
    with st.container(key=f"idcard_ai_{idx}"):
        st.markdown(card_head_html(tags, level), unsafe_allow_html=True)
        st.markdown(cited_html(ans.text, sources), unsafe_allow_html=True)
        if ans.unverified:
            st.markdown(f"<div class='id-unverified'>Not found in the passages: {esc(', '.join(ans.unverified))}. "
                        "Check these in the source.</div>", unsafe_allow_html=True)
        if sources:
            st.markdown(source_cards_html(sources), unsafe_allow_html=True)
        best = sources[0] if sources else None
        action_row(ans, idx, best, snippet(best.text, trace.question)[1] if best else "")
    questions = follow_ups(trace.question, trace.retrieved)
    if questions:
        with st.container(key=f"idfu_{idx}", horizontal=True):
            for j, q in enumerate(questions):
                if st.button(q, key=f"fu_{idx}_{j}"):
                    ss.pending_question = q
                    st.rerun()


def render_sources(ans: Answer, idx: int, level: str, trace: Trace) -> None:
    tag = "Quota used up" if is_quota(ans.note) else "AI off"
    best = ans.passages[0]
    body, line = snippet(best.text, trace.question)
    with st.container(key=f"idcard_off_{idx}"):
        st.markdown(card_head_html([tag_html(tag, "warn")], level), unsafe_allow_html=True)
        st.markdown(f"{source_line_html(best)}<div class='id-snippet'>{body}</div>", unsafe_allow_html=True)
        with st.expander(f"All passages ({len(ans.passages)}, full text)"):
            render_passages(ans.passages)
        reason = SOURCES_MODE_NOTE + (f" {ans.note}" if ans.note else "")
        st.markdown(f"<div class='id-reason'>{esc(reason)}</div>", unsafe_allow_html=True)
        action_row(ans, idx, best, line, passages=False)


def render_answer(ans: Answer, trace: Trace, idx: int) -> None:
    st.markdown(trail_html(trail_steps(ans, trace), trace.seconds), unsafe_allow_html=True)
    s = get_settings()
    if ans.mode == "exact" and ans.fact:
        score = ans.passages[0].score if ans.passages else 1.0
        render_exact(ans, idx, confidence(score, True, s.min_score, s.fact_match_score))
    elif ans.mode in AI_MODES or (ans.mode == "sources" and ans.passages):
        pool = trace.retrieved or ans.passages
        level = confidence(max(p.score for p in pool), False, s.min_score, s.fact_match_score) if pool else None
        if ans.mode == "sources":
            render_sources(ans, idx, level, trace)
        else:
            render_ai(ans, idx, level, trace)
    elif ans.mode in ("not_found", "sources"):
        with st.container(key=f"idcard_nf_{idx}"):
            st.markdown(card_head_html([tag_html("Not found", "bad")], None) + not_found_html(),
                        unsafe_allow_html=True)
    else:
        with st.container(key=f"idcard_err_{idx}"):
            st.markdown(card_head_html([tag_html("Error", "bad")], None) +
                        f"<div class='id-nf'>{esc(ans.text)}</div>", unsafe_allow_html=True)
    if ans.loading:
        st.caption(f"⏳ {ans.loading}")


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def result_messages(r) -> list[tuple[str, str]]:
    """(message kind, text) pairs for an IngestResult."""
    kind = ("success" if r.status in ("added", "replaced") else
            "info" if r.status in ("duplicate", "loading", "cancelled") else "error")
    return [(kind, r.message)] + [("warning", w) for w in r.warnings if w != r.message]


def job_text(job) -> str:
    if job.status == "running" and job.total:
        return f"Reading {job.file_name}: page {job.done} of {job.total}"
    return f"Waiting: {job.file_name} (files are read one at a time)"


def ingest_files(engine: InfoDeckEngine, files) -> list[tuple[str, str]]:
    """Queue uploaded files to be read in the background, one at a time.

    Quick files are waited for a few seconds so their result shows right away; big ones keep
    loading (progress in the sidebar) while the user can already ask about finished pages.
    """
    results, queued = [], []
    for f in files:
        try:
            if f.size > MAX_UPLOAD_MB * 1024 * 1024:
                results.append(("error", f"{f.name} is larger than {MAX_UPLOAD_MB} MB."))
                continue
            r = engine.queue_upload(f.name, f.getvalue())
        except Exception:
            log.exception("Upload failed for %s", f.name)
            results.append(("error", f"{f.name} could not be added. Please try again."))
            continue
        if r.status == "queued":
            queued.append(r.doc_id)
        else:
            results += result_messages(r)
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
                results.append(("info", f"Reading {job.file_name} in the background (progress below). "
                                        "You can already ask about the pages that are done."))
            else:
                results += result_messages(job.result)
    return results


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
            ss.upload_results = ss.upload_results + result_messages(result)
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
    with live.container():
        with st.container(key="idcard_live"):
            st.markdown(card_head_html([tag_html("AI answer", "ai")], None), unsafe_allow_html=True)
            text = st.write_stream(stream())
    live.empty()  # replaced by the final, formatted answer card
    if errors:
        return engine.sources_answer(passages, f"AI answer failed: {errors[0]}")
    return engine.finish_llm_answer(status, text if isinstance(text, str) else "".join(text), passages, question)


# --------------------------------------------------------------------------- #
# Start-up
# --------------------------------------------------------------------------- #
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
ss.setdefault("upload_results", [])
ss.setdefault("watch", set())  # documents this browser tab uploaded that are still being read
ss.setdefault("preview_failed", set())  # (doc_id, page, line) whose page preview could not be made

topbar = st.empty()
render_topbar(topbar, engine)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"<div class='id-brand'><span class='glyph'></span>{APP_NAME}</div>", unsafe_allow_html=True)

    status = resolve_llm(get_settings())
    if status.llm is None:
        st.markdown(f"<div class='id-pill-reason'>AI answers are off: {esc(status.reason)}</div>",
                    unsafe_allow_html=True)

    st.subheader("Add documents")
    settings = get_settings()
    st.caption("Reading pages with Gemini (page images are sent to Google)" if settings.parser == "gemini"
               else "Reading pages on this computer (scans with local OCR)")
    files = st.file_uploader("PDF, Word, Excel, PowerPoint, text or image files", type=UPLOAD_TYPES,
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
                             label_visibility="collapsed")
    if files:
        ss.upload_results = ingest_files(engine, files)
        ss.uploader_key += 1  # clears the upload box
        st.rerun()
    for kind, msg in ss.upload_results:
        getattr(st, kind)(msg)
    loading_panel(engine)

    docs = engine.store.list_documents()
    st.subheader(f"Documents ({len(docs)})")
    if not docs:
        st.caption("No documents loaded yet.")
    for d in docs:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        pages = f"📄 {d.pages} page{'s' if d.pages != 1 else ''}"
        job = engine.jobs.get(d.doc_id)
        if job and job.active:
            pages = f"⏳ loading, page {job.done} of {job.total}"
        c1.markdown(f"<div class='id-doc'>{esc(d.file_name)}<br><small>{pages}</small></div>",
                    unsafe_allow_html=True)
        if c2.button("✕", key=f"ask_rm_{d.doc_id}", help=f"Remove {d.file_name}"):
            ss.confirm_remove = d.doc_id
            st.rerun()
        if ss.get("confirm_remove") == d.doc_id:
            st.caption(f"Remove {d.file_name} for everyone?")
            y, n = st.columns(2)
            if y.button("Remove", key=f"rm_{d.doc_id}", type="primary", use_container_width=True):
                engine.remove_document(d.doc_id)
                ss.confirm_remove = None
                ss.upload_results = [("info", f"Removed {d.file_name}.")]
                st.rerun()
            if n.button("Cancel", key=f"cancel_rm_{d.doc_id}", use_container_width=True):
                ss.confirm_remove = None
                st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True, disabled=not ss.messages):
        ss.messages = []
        st.rerun()
    st.caption("Answers come only from the loaded documents. Always check the cited source.")

# --------------------------------------------------------------------------- #
# Main area: chat
# --------------------------------------------------------------------------- #
welcome = st.empty()  # headline + example questions, only while the chat is empty
with welcome.container():
    if not ss.messages:
        st.markdown(
            f"<div class='id-hero'><span class='id-badge'><span class='glyph'></span>{APP_NAME}</span>"
            f"<div class='id-headline'>{HEADLINE}</div><div class='id-tagline'>{TAGLINE}</div></div>",
            unsafe_allow_html=True,
        )
    if not docs:
        st.info("Click + in the chat box (or use the sidebar) to add a PDF, Word, Excel, PowerPoint, text or image file.")
    elif not ss.messages and any(d.file_name.startswith("KRT_") for d in docs):
        st.markdown("<div class='id-hint'>Try one of these</div>", unsafe_allow_html=True)
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
            render_answer(m["answer"], m.get("trace") or Trace(), i)

for kind, msg in ss.pop("chat_upload_results", []):
    getattr(st, kind)(msg)

typed = st.chat_input("Ask a question or attach a document...", accept_file="multiple",
                      file_type=UPLOAD_TYPES)
if typed and typed.files:
    # Index the attached files first; a question typed with them is answered on the next run.
    ss.upload_results = ss.chat_upload_results = ingest_files(engine, typed.files)
    if typed.text.strip():
        ss.pending_question = typed.text.strip()
    st.rerun()

question = ((typed.text if typed else "") or ss.pop("pending_question", "") or "").strip()
if question and not docs and not engine.jobs.active():
    st.warning("Please add a document first (use the + button).")
elif question:
    welcome.empty()
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
            ans, trace = Answer("Sorry, something went wrong while answering. Please try again.", "error"), Trace()
        skeleton.empty()
        render_answer(ans, trace, len(ss.messages))
    ss.messages.append({"role": "assistant", "answer": ans, "trace": trace})
    render_topbar(topbar, engine)  # the AI status may have changed (e.g. quota used up)

for load in deferred:  # last: page previews never hold up the answer
    load()
