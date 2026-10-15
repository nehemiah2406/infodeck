"""InfoDeck - ask questions about your company documents.

Run with run.bat, or:  .venv\\Scripts\\python -m streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from infodeck import APP_NAME  # noqa: E402
from infodeck.config import get_settings, setup_logging  # noqa: E402
from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder  # noqa: E402
from infodeck.engine import Answer, InfoDeckEngine  # noqa: E402
from infodeck.llm import LLMError, resolve_llm  # noqa: E402
from infodeck.prompts import NOT_FOUND, SOURCES_MODE_NOTE  # noqa: E402
from infodeck.store import Passage, VectorStore  # noqa: E402

log = logging.getLogger("infodeck.app")

TAGLINE = "Ask questions about your company documents"
MAX_UPLOAD_MB = 50
SAMPLE_QUESTIONS = [
    "How many earned leave days do I get per year?",
    "What is the hotel limit per night in Mumbai?",
    "What should I do if I lose my laptop?",
]

st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="centered",
                   initial_sidebar_state="expanded", menu_items={})

st.markdown(
    """
<style>
/* Hide Streamlit branding: menu, toolbar, Deploy button, footer, top colour bar */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stMainMenu"] {
    display: none !important;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 820px; }
.id-title { font-size: 2.1rem; font-weight: 700; color: #1F4E79; margin-bottom: 0; }
.id-tagline { color: #5b6573; margin-top: .1rem; margin-bottom: 1.2rem; }
.id-status { font-size: .85rem; padding: .45rem .6rem; border-radius: .4rem; margin: .3rem 0 .8rem 0; }
.id-status.on  { background: #e8f4ec; color: #1d5e33; }
.id-status.off { background: #fff4e0; color: #7a4d00; }
.id-passage { background: #f5f7fa; border-left: 3px solid #1F4E79; padding: .55rem .75rem;
    border-radius: .3rem; font-size: .9rem; line-height: 1.45; margin: .2rem 0 .8rem 0; }
.id-source { font-size: .85rem; font-weight: 600; color: #1F4E79; }
.id-meta { font-weight: 400; color: #6b7480; }
.id-doc { font-size: .9rem; line-height: 1.25; }
.id-doc small { color: #6b7480; }
</style>
""",
    unsafe_allow_html=True,
)


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


def esc(text: str) -> str:
    """Escape document text for safe HTML display (no injected HTML/markdown/LaTeX)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def render_passages(passages: list[Passage]) -> None:
    for p in passages:
        section = f" · {esc(p.section)}" if p.section else ""
        st.markdown(
            f"<div class='id-source'>{esc(p.label)}<span class='id-meta'>{section}"
            f" · relevance {p.score:.2f}</span></div><div class='id-passage'>{esc(p.text)}</div>",
            unsafe_allow_html=True,
        )


def render_answer(ans: Answer) -> None:
    if ans.mode == "not_found":
        st.markdown(NOT_FOUND)
        st.caption("Try rephrasing the question, or upload the document that contains the answer.")
    elif ans.mode == "sources":
        st.markdown(f"**{SOURCES_MODE_NOTE}**")
        if ans.note:
            st.caption(ans.note)
        with st.expander(f"Sources ({len(ans.passages)} passages)", expanded=True):
            render_passages(ans.passages)
    elif ans.mode == "error":
        st.warning(ans.text)
    else:
        st.markdown(ans.text.replace("$", "\\$"))
        if ans.passages:
            with st.expander(f"Sources ({len(ans.passages)})"):
                render_passages(ans.passages)


def answer_question(engine: InfoDeckEngine, question: str) -> Answer:
    """Retrieve, then answer with the LLM (streamed), or fall back to sources mode."""
    status = resolve_llm(get_settings())
    passages = engine.retrieve(question)
    if not passages:
        return Answer(NOT_FOUND, "not_found")
    if status.llm is None:
        return engine.sources_answer(passages, status.reason)

    errors: list[str] = []

    def stream():
        try:
            yield from engine.stream_llm(status, question, passages)
        except LLMError as e:
            errors.append(str(e))

    live = st.empty()
    with live.container():
        with st.spinner("Thinking..."):
            text = st.write_stream(stream())
    live.empty()  # replaced by the final, formatted answer below
    if errors:
        return engine.sources_answer(passages, f"AI answer failed: {errors[0]}")
    return engine.finish_llm_answer(status, text if isinstance(text, str) else "".join(text), passages)


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

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"<div class='id-title' style='font-size:1.5rem'>📘 {APP_NAME}</div>", unsafe_allow_html=True)

    status = resolve_llm(get_settings())
    if status.llm is not None:
        st.markdown(f"<div class='id-status on'>● AI mode: {esc(status.label)}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='id-status off'>● AI mode: Sources mode<br><small>{esc(status.reason)}</small></div>",
                    unsafe_allow_html=True)

    st.subheader("Add documents")
    files = st.file_uploader("PDF, Word or text files", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
                             label_visibility="collapsed")
    if files:
        results = []
        with st.spinner("Reading and indexing..."):
            for f in files:
                try:
                    if f.size > MAX_UPLOAD_MB * 1024 * 1024:
                        results.append(("error", f"{f.name} is larger than {MAX_UPLOAD_MB} MB."))
                        continue
                    r = engine.save_and_ingest_upload(f.name, f.getvalue())
                    results.append(("success" if r.status in ("added", "replaced") else
                                    "info" if r.status == "duplicate" else "error", r.message))
                    results += [("warning", w) for w in r.warnings if w != r.message]
                except Exception:
                    log.exception("Upload failed for %s", f.name)
                    results.append(("error", f"{f.name} could not be added. Please try again."))
        ss.upload_results = results
        ss.uploader_key += 1  # clears the upload box
        st.rerun()
    for kind, msg in ss.upload_results:
        getattr(st, kind)(msg)

    docs = engine.store.list_documents()
    st.subheader(f"Documents ({len(docs)})")
    if not docs:
        st.caption("No documents loaded yet.")
    for d in docs:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        pages = f"{d.pages} page{'s' if d.pages != 1 else ''}"
        c1.markdown(f"<div class='id-doc'>📄 {esc(d.file_name)}<br><small>{pages}</small></div>",
                    unsafe_allow_html=True)
        with c2.popover("✕", help=f"Remove {d.file_name}"):
            st.write(f"Remove **{d.file_name}** for everyone?")
            if st.button("Remove", key=f"rm_{d.doc_id}", type="primary"):
                engine.remove_document(d.doc_id)
                ss.upload_results = [("info", f"Removed {d.file_name}.")]
                st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True, disabled=not ss.messages):
        ss.messages = []
        st.rerun()
    st.caption("Answers come only from the loaded documents. Always check the cited source.")

# --------------------------------------------------------------------------- #
# Main area: chat
# --------------------------------------------------------------------------- #
st.markdown(f"<div class='id-title'>{APP_NAME}</div><div class='id-tagline'>{TAGLINE}</div>",
            unsafe_allow_html=True)

if not docs:
    st.info("Upload a PDF, Word or text file in the sidebar to get started.")
elif not ss.messages and any(d.file_name.startswith("KRT_") for d in docs):
    st.caption("Try one of these:")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}"):
            ss.pending_question = q
            st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"], avatar="📘" if m["role"] == "assistant" else None):
        if m["role"] == "user":
            st.markdown(esc(m["content"]), unsafe_allow_html=True)
        else:
            render_answer(m["answer"])

typed = st.chat_input("Ask a question about your documents...", disabled=not docs)
question = (typed or ss.pop("pending_question", "") or "").strip()
if question:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(esc(question), unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="📘"):
        try:
            ans = answer_question(engine, question)
        except Exception:
            log.exception("Answering failed")
            ans = Answer("Sorry, something went wrong while answering. Please try again.", "error")
        render_answer(ans)
    ss.messages.append({"role": "assistant", "answer": ans})
