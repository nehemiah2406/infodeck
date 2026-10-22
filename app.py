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

HEADLINE = "Every answer, straight from your documents"
TAGLINE = "Ask questions about your company policies and files. Each answer cites the page it came from."
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --ink: #050507;
    --glass: rgba(255, 255, 255, 0.035);
    --glass-strong: rgba(255, 255, 255, 0.06);
    --line: rgba(255, 255, 255, 0.09);
    --line-strong: rgba(255, 255, 255, 0.18);
    --text: #ededef;
    --muted: #8e8e98;
    --orange: #ff7a45;
    --red: #e2455a;
    --violet: #8b6cff;
    --blue: #5b7cfa;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stMainMenu"] {
    display: none !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
/* Don't grey out the page while it updates */
[data-stale="true"] { opacity: 1 !important; }

/* ---------- Base + "new era" particle ring ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}
.stApp { background: var(--ink); color: var(--text); }
/* Ring of fine light strands: a conic colour wheel masked to a thin, streaky ring */
.stApp::before {
    content: ""; position: fixed; left: 50%; top: 38%; z-index: 0; pointer-events: none;
    width: min(1150px, 150vw); aspect-ratio: 1; translate: -50% -50%;
    background: conic-gradient(from 210deg, var(--orange), var(--red), var(--violet), var(--blue),
                               var(--violet), var(--red), var(--orange));
    -webkit-mask: radial-gradient(closest-side, transparent 56%, #000 63%, #000 70%, transparent 82%),
                  repeating-conic-gradient(#000 0deg .3deg, transparent .3deg 1.05deg);
    -webkit-mask-composite: source-in;
    mask: radial-gradient(closest-side, transparent 56%, #000 63%, #000 70%, transparent 82%),
          repeating-conic-gradient(#000 0deg .3deg, transparent .3deg 1.05deg);
    mask-composite: intersect;
    opacity: .5; filter: blur(.4px);
    animation: drift 140s linear infinite;
}
/* Soft glow behind the strands, plus darkness in the middle so text stays readable */
.stApp::after {
    content: ""; position: fixed; left: 50%; top: 38%; z-index: 0; pointer-events: none;
    width: min(1150px, 150vw); aspect-ratio: 1; translate: -50% -50%;
    background:
        radial-gradient(closest-side, var(--ink) 48%, transparent 70%),
        conic-gradient(from 210deg, rgba(255, 122, 69, .35), rgba(226, 69, 90, .3), rgba(139, 108, 255, .3),
                       rgba(91, 124, 250, .35), rgba(139, 108, 255, .3), rgba(226, 69, 90, .3), rgba(255, 122, 69, .35));
    border-radius: 50%; filter: blur(60px); opacity: .55;
    animation: breathe 12s ease-in-out infinite alternate;
}
@keyframes drift   { to { rotate: 360deg; } }
@keyframes breathe { from { scale: .96; opacity: .45; } to { scale: 1.04; opacity: .6; } }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"],
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }
.block-container { padding-top: 3rem; max-width: 820px; }

/* ---------- Hero ---------- */
.id-hero { text-align: center; margin: 3.2rem 0 2.2rem 0; }
.id-hero.compact { margin: .4rem 0 1.4rem 0; }
.id-badge {
    display: inline-flex; align-items: center; gap: .5rem; font-size: .8rem; color: #d4d4d8;
    padding: .34rem .85rem; border-radius: 999px; background: rgba(255, 255, 255, .04);
    border: 1px solid var(--line-strong); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.id-badge .glyph { width: .55rem; height: .55rem; border-radius: 2px; rotate: 45deg;
    background: linear-gradient(135deg, var(--orange), var(--blue)); }
.id-headline {
    font-weight: 300; font-size: clamp(2.2rem, 5.4vw, 3.6rem); line-height: 1.08; letter-spacing: -0.035em;
    margin: 1.2rem 0 0 0;
    background: linear-gradient(180deg, #ffffff 30%, #a1a1aa 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.id-hero.compact .id-headline { font-size: 1.7rem; margin-top: .8rem; }
.id-tagline { color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin: 1rem auto 0 auto; max-width: 520px; }
.id-hero.compact .id-tagline { display: none; }

/* ---------- Sidebar (dark glass) ---------- */
[data-testid="stSidebar"] {
    background: rgba(8, 8, 11, 0.78) !important;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--line);
}
.id-brand { display: flex; align-items: center; gap: .6rem; font-size: 1.35rem; font-weight: 600;
    letter-spacing: -0.02em; color: #fafafa; margin-bottom: .3rem; }
.id-brand .glyph { width: .8rem; height: .8rem; border-radius: 3px; rotate: 45deg;
    background: linear-gradient(135deg, var(--orange), var(--blue)); box-shadow: 0 0 16px rgba(255, 122, 69, .45); }
[data-testid="stSidebar"] h3 { font-size: .74rem; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass) !important; border: 1px dashed var(--line-strong) !important; border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(255, 122, 69, .55) !important; }
.id-doc { font-size: .88rem; line-height: 1.3; color: var(--text); overflow-wrap: anywhere; }
[data-testid="stSidebar"] [class*="st-key-ask_rm_"] button {
    padding: .1rem .5rem !important; min-height: 2rem; color: var(--muted) !important;
}
[data-testid="stSidebar"] [class*="st-key-ask_rm_"] button:hover { color: #fca5a5 !important; border-color: rgba(248, 113, 113, .6) !important; }
.id-doc small { color: var(--muted); }

/* AI-mode pill */
.id-pill {
    display: inline-flex; align-items: center; gap: .45rem; font-size: .78rem; font-weight: 500;
    padding: .3rem .75rem; border-radius: 999px; margin: .4rem 0 .2rem 0;
}
.id-pill .dot { width: .45rem; height: .45rem; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }
.id-pill.on  { background: rgba(91, 124, 250, 0.12); border: 1px solid rgba(91, 124, 250, 0.42); color: #b4c3ff; }
.id-pill.off { background: rgba(255, 122, 69, 0.10); border: 1px solid rgba(255, 122, 69, 0.42); color: #ffb08f; }
.id-pill-reason { font-size: .78rem; color: var(--muted); margin-bottom: .8rem; }

/* ---------- Buttons: outlined pills ---------- */
.stButton > button, [data-testid="stPopover"] button {
    background: rgba(255, 255, 255, .03) !important; color: var(--text) !important;
    border: 1px solid var(--line-strong) !important; border-radius: 999px !important;
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}
.stButton > button:hover {
    border-color: rgba(255, 255, 255, .42) !important; box-shadow: 0 0 22px rgba(255, 122, 69, .16);
}
.stButton > button[kind="primary"] {
    background: #f4f4f5 !important; border: 1px solid #f4f4f5 !important; color: #09090b !important; font-weight: 500;
}
.stButton > button[kind="primary"]:hover { background: #ffffff !important; }
/* Example questions */
.st-key-samples { justify-content: center; }
.st-key-samples .stButton > button { padding: .5rem .7rem .5rem 1.1rem !important; color: #e4e4e7 !important; }
.st-key-samples .stButton > button p::after {
    content: "↗"; display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem; margin-left: .6rem;
    border-radius: 50%; border: 1px solid var(--line-strong); font-size: .75rem; color: #d4d4d8;
}
.st-key-samples .stButton > button:hover { transform: translateY(-1px); }
.id-hint { color: var(--muted); font-size: .82rem; margin-bottom: .6rem; text-align: center; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: rgba(12, 12, 16, .55); border: 1px solid var(--line); border-radius: 20px;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    padding: 1rem 1.15rem; margin-bottom: .8rem;
    animation: fadeUp .45s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(255, 122, 69, 0.06); border-color: rgba(255, 140, 100, 0.22);
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.65; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background: rgba(12, 12, 16, 0.85) !important; border-radius: 22px !important;
    border-color: var(--line-strong) !important;
}
[data-testid="stChatInput"]:focus-within > div { border-color: rgba(255, 122, 69, .55) !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }

/* Citation chips */
.chip {
    display: inline-block; font-size: .72rem; font-weight: 500; line-height: 1.5; white-space: nowrap;
    padding: .05rem .55rem; margin: 0 .12rem; border-radius: 999px; vertical-align: 1px;
    background: rgba(91, 124, 250, 0.13); border: 1px solid rgba(91, 124, 250, 0.38); color: #c7d2fe;
}
.chip.score { background: rgba(255, 122, 69, 0.10); border-color: rgba(255, 122, 69, 0.34); color: #ffc2a8; }

/* Sources expander */
[data-testid="stExpander"] details {
    background: rgba(255, 255, 255, 0.025) !important; border: 1px solid var(--line) !important;
    border-radius: 16px !important; backdrop-filter: blur(12px);
}
[data-testid="stExpander"] summary { color: var(--muted); font-size: .88rem; }
.id-source { margin: .2rem 0 .3rem 0; font-size: .82rem; color: var(--muted); }
.id-passage {
    background: rgba(255, 255, 255, 0.025); border-left: 2px solid var(--orange);
    padding: .6rem .8rem; border-radius: 8px; font-size: .88rem; line-height: 1.55;
    color: #d4d4d8; margin: 0 0 .9rem 0;
}
.id-notice { color: #ffb08f; font-weight: 500; }

/* Progress bar + spinner */
[data-testid="stProgress"] div[role="progressbar"] > div > div {
    background-image: linear-gradient(90deg, var(--orange), var(--red), var(--violet), var(--blue)) !important;
}
[data-testid="stSpinner"] { color: var(--muted); }
[data-testid="stAlert"] { background: var(--glass) !important; border: 1px solid var(--line); border-radius: 14px; }

@media (prefers-reduced-motion: reduce) {
    .stApp::before, .stApp::after, [data-testid="stChatMessage"] { animation: none !important; }
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
    st.markdown(f"<div class='id-brand'><span class='glyph'></span>{APP_NAME}</div>", unsafe_allow_html=True)

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
st.markdown(
    f"<div class='id-hero{' compact' if ss.messages else ''}'>"
    f"<span class='id-badge'><span class='glyph'></span>{APP_NAME}</span>"
    f"<div class='id-headline'>{HEADLINE}</div><div class='id-tagline'>{TAGLINE}</div></div>",
    unsafe_allow_html=True,
)

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
