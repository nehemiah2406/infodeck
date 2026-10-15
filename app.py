"""InfoDeck - ask questions about your company documents.

Run with run.bat, or:  .venv\\Scripts\\python -m streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import re
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
# Citations written by the LLM, e.g. [KRT_Leave_Policy.pdf, p.2] or [a.pdf, p.1; b.pdf, p.3]
CITATION_RE = re.compile(r"\[([^\[\]]+?\.(?:pdf|docx|txt)\b[^\[\]]*)\]", re.IGNORECASE)

st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="centered",
                   initial_sidebar_state="expanded", menu_items={})

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

:root {
    --ink: #0b0f1a;
    --glass: rgba(255, 255, 255, 0.045);
    --glass-strong: rgba(255, 255, 255, 0.07);
    --glass-border: rgba(255, 255, 255, 0.11);
    --text: #e7eaf3;
    --muted: #9aa3b8;
    --violet: #a78bfa;
    --blue: #60a5fa;
    --teal: #2dd4bf;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stMainMenu"] {
    display: none !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
/* Don't grey out the page while it updates */
[data-stale="true"] { opacity: 1 !important; }

/* ---------- Base + aurora background ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}
.stApp { background: var(--ink); color: var(--text); }
.stApp::before {
    content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
    background:
        radial-gradient(38% 32% at 18% 22%, rgba(139, 92, 246, 0.42), transparent 70%),
        radial-gradient(34% 30% at 82% 14%, rgba(59, 130, 246, 0.34), transparent 70%),
        radial-gradient(40% 34% at 62% 88%, rgba(20, 184, 166, 0.30), transparent 70%);
    filter: blur(48px);
    animation: aurora 24s ease-in-out infinite alternate;
}
@keyframes aurora {
    0%   { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
    50%  { transform: translate3d(3%, -2%, 0) rotate(8deg) scale(1.08); }
    100% { transform: translate3d(-3%, 2%, 0) rotate(-6deg) scale(1.03); }
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"],
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }
.block-container { padding-top: 2.6rem; max-width: 840px; }

/* ---------- Title ---------- */
.id-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 3rem;
    letter-spacing: -0.03em; line-height: 1.1; margin: 0; display: inline-block;
    background: linear-gradient(92deg, #c4b5fd 0%, #93c5fd 50%, #5eead4 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.id-tagline { color: var(--muted); font-size: 1.05rem; margin: .35rem 0 1.6rem 0; }
.id-side-title { font-size: 1.7rem; margin-bottom: .2rem; }

/* ---------- Sidebar (glass) ---------- */
[data-testid="stSidebar"] {
    background: rgba(13, 18, 32, 0.72) !important;
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] h3 { font-size: .8rem; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass) !important; border: 1px dashed rgba(167, 139, 250, 0.45) !important; border-radius: 14px;
}
.id-doc { font-size: .88rem; line-height: 1.3; color: var(--text); overflow-wrap: anywhere; }
[data-testid="stSidebar"] [class*="st-key-ask_rm_"] button {
    padding: .1rem .5rem !important; min-height: 2rem; color: var(--muted) !important;
}
[data-testid="stSidebar"] [class*="st-key-ask_rm_"] button:hover { color: #fca5a5 !important; border-color: rgba(248, 113, 113, .6) !important; }
.id-doc small { color: var(--muted); }

/* AI-mode pill */
.id-pill {
    display: inline-flex; align-items: center; gap: .45rem; font-size: .8rem; font-weight: 600;
    padding: .32rem .75rem; border-radius: 999px; margin: .4rem 0 .2rem 0;
}
.id-pill .dot { width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }
.id-pill.on  { background: rgba(45, 212, 191, 0.13); border: 1px solid rgba(45, 212, 191, 0.45); color: #5eead4; }
.id-pill.off { background: rgba(251, 191, 36, 0.12); border: 1px solid rgba(251, 191, 36, 0.42); color: #fcd34d; }
.id-pill-reason { font-size: .78rem; color: var(--muted); margin-bottom: .8rem; }

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stPopover"] button {
    background: var(--glass) !important; color: var(--text) !important;
    border: 1px solid var(--glass-border) !important; border-radius: 10px !important;
    transition: all .2s ease;
}
.stButton > button:hover { border-color: rgba(167, 139, 250, 0.6) !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(92deg, #7c3aed, #2563eb) !important; border: none !important; color: white !important;
}
/* Example questions: glowing pills */
.st-key-samples .stButton > button {
    border-radius: 999px !important; padding: .45rem 1.05rem !important;
    border: 1px solid rgba(167, 139, 250, 0.5) !important;
    background: rgba(139, 92, 246, 0.10) !important; color: #ddd6fe !important;
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.22);
}
.st-key-samples .stButton > button:hover {
    box-shadow: 0 0 28px rgba(139, 92, 246, 0.50); transform: translateY(-1px);
    border-color: rgba(196, 181, 253, 0.9) !important;
}
.id-hint { color: var(--muted); font-size: .85rem; margin-bottom: .4rem; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--glass); border: 1px solid var(--glass-border); border-radius: 18px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    padding: 1rem 1.1rem; margin-bottom: .8rem;
    animation: fadeUp .45s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(139, 92, 246, 0.12); border-color: rgba(167, 139, 250, 0.30);
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.6; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background: rgba(20, 26, 43, 0.85) !important; border-radius: 16px !important;
    border-color: rgba(167, 139, 250, 0.35) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }

/* Citation chips */
.chip {
    display: inline-block; font-size: .72rem; font-weight: 500; line-height: 1.5; white-space: nowrap;
    padding: .05rem .55rem; margin: 0 .12rem; border-radius: 999px; vertical-align: 1px;
    background: rgba(96, 165, 250, 0.14); border: 1px solid rgba(96, 165, 250, 0.38); color: #bfdbfe;
}
.chip.score { background: rgba(45, 212, 191, 0.10); border-color: rgba(45, 212, 191, 0.32); color: #99f6e4; }

/* Sources expander as glass card */
[data-testid="stExpander"] details {
    background: rgba(255, 255, 255, 0.035) !important; border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important; backdrop-filter: blur(12px);
}
[data-testid="stExpander"] summary { color: var(--muted); font-size: .88rem; }
.id-source { margin: .2rem 0 .3rem 0; font-size: .82rem; color: var(--muted); }
.id-passage {
    background: rgba(255, 255, 255, 0.03); border-left: 2px solid var(--violet);
    padding: .6rem .8rem; border-radius: 8px; font-size: .88rem; line-height: 1.55;
    color: #d5dae7; margin: 0 0 .9rem 0;
}
.id-notice { color: #fcd34d; font-weight: 600; }

/* Progress bar + spinner */
[data-testid="stProgress"] div[role="progressbar"] > div > div {
    background-image: linear-gradient(90deg, #8b5cf6, #3b82f6, #14b8a6) !important;
}
[data-testid="stSpinner"] { color: var(--muted); }
[data-testid="stAlert"] { background: var(--glass) !important; border: 1px solid var(--glass-border); border-radius: 12px; }

@media (prefers-reduced-motion: reduce) {
    .stApp::before, [data-testid="stChatMessage"] { animation: none !important; }
    .st-key-samples .stButton > button:hover { transform: none; }
}
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


def answer_html(text: str) -> str:
    """LLM answer as markdown-safe HTML, with [file, p.N] citations turned into chips."""
    safe = html.escape(text, quote=False).replace("$", "&#36;")

    def chips(m: re.Match) -> str:
        return "".join(f"<span class='chip'>{p.strip()}</span>" for p in m.group(1).split(";") if p.strip())

    return CITATION_RE.sub(chips, safe)


def render_passages(passages: list[Passage]) -> None:
    for p in passages:
        section = f" &nbsp;{esc(p.section)}" if p.section else ""
        st.markdown(
            f"<div class='id-source'><span class='chip'>{esc(p.label)}</span>"
            f"<span class='chip score'>relevance {p.score:.2f}</span>{section}</div>"
            f"<div class='id-passage'>{esc(p.text)}</div>",
            unsafe_allow_html=True,
        )


def render_answer(ans: Answer) -> None:
    if ans.mode == "not_found":
        st.markdown(NOT_FOUND)
        st.caption("Try rephrasing the question, or upload the document that contains the answer.")
    elif ans.mode == "sources":
        st.markdown(f"<span class='id-notice'>{SOURCES_MODE_NOTE}</span>", unsafe_allow_html=True)
        if ans.note:
            st.caption(ans.note)
        with st.expander(f"Sources ({len(ans.passages)} passages)", expanded=True):
            render_passages(ans.passages)
    elif ans.mode == "error":
        st.warning(ans.text)
    else:
        st.markdown(answer_html(ans.text), unsafe_allow_html=True)
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
    st.markdown(f"<div class='id-title id-side-title'>{APP_NAME}</div>", unsafe_allow_html=True)

    status = resolve_llm(get_settings())
    if status.llm is not None:
        st.markdown(f"<div class='id-pill on'><span class='dot'></span>{esc(status.label)}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='id-pill off'><span class='dot'></span>Sources mode</div>"
                    f"<div class='id-pill-reason'>{esc(status.reason)}</div>", unsafe_allow_html=True)

    st.subheader("Add documents")
    files = st.file_uploader("PDF, Word or text files", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
                             label_visibility="collapsed")
    if files:
        results = []
        bar = st.progress(0.0, text="Preparing...")
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Indexing {f.name} ({i + 1}/{len(files)})")
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
        bar.progress(1.0, text="Done")
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
        c1.markdown(f"<div class='id-doc'>{esc(d.file_name)}<br><small>📄 {pages}</small></div>",
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
st.markdown(f"<div class='id-title'>{APP_NAME}</div><div class='id-tagline'>{TAGLINE}</div>",
            unsafe_allow_html=True)

if not docs:
    st.info("Upload a PDF, Word or text file in the sidebar to get started.")
elif not ss.messages and any(d.file_name.startswith("KRT_") for d in docs):
    st.markdown("<div class='id-hint'>Try one of these</div>", unsafe_allow_html=True)
    with st.container(key="samples", horizontal=True):
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
