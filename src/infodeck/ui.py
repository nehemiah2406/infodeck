"""HTML pieces for the answer UI (top bar, the answer card, snippets, Copy button).

Pure functions, so they can be tested without Streamlit. Every piece of document or
answer text goes through esc() (or _highlight / _key_numbers, which escape too): numbers,
units and quotes are shown exactly as they are in the Answer, never changed.

Every answer uses one card, in this order: the answer line (key value in maroon), file name +
page, "Exact text from the document" with the quote, and a Copy button. Nothing else.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from .engine import NUMBER_RE, Answer
from .facts import QUESTION_STOPWORDS, Fact
from .prompts import NOT_FOUND
from .store import Passage

# Citations written by the LLM, e.g. [KRT_Leave_Policy.pdf, p.2], [a.pdf, p.1; b.pdf, p.3] or, from some
# models, with full-width brackets: 【KRT_Leave_Policy.pdf, p.2】
CITATION_RE = re.compile(r"[\[【]([^\[\]【】]+?\.(?:pdf|docx|txt|xlsx|csv|pptx|png|jpe?g)\b[^\[\]【】]*)[\]】]",
                         re.IGNORECASE)
AI_MODES = {"claude": "Claude", "gemini": "Gemini", "groq": "Groq", "ollama": "Ollama"}
EXACT_HIGH_MARGIN = 0.10  # exact value: High if the match score is this far above FACT_MATCH_SCORE
# Passages: High / Medium if the best score is this far above MIN_SCORE. Measured with bge-small on
# the sample docs: clear matches score 0.70-0.79, off-topic questions that still pass MIN_SCORE 0.58-0.59.
PASSAGE_HIGH_MARGIN = 0.15
PASSAGE_MEDIUM_MARGIN = 0.05
QUOTE_LABEL = "Exact text from the document"
LONG_QUOTE = 200  # an exact value's source line longer than this is cut to the sentence with the value
ONE_SENTENCE_MIN = 25  # characters: shorter first "sentences" ("U.S.") are joined with the next
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=\S)|\s+(?=•\s)")
SNIPPET_LINES = 3
LONG_LINE = 300  # a "line" longer than this is split into sentences for the snippet
WORD_RE = re.compile(r"\w+(?:[.,]\d+)*")


def esc(text: str) -> str:
    """Escape document text for safe HTML display (no injected HTML/markdown/LaTeX)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def _attr(text: str) -> str:
    return html.escape(text, quote=True).replace("$", "&#36;")


@dataclass
class Trace:
    """What happened while answering one question."""
    question: str = ""
    docs: int = 0  # documents searched
    retrieved: list[Passage] = field(default_factory=list)  # all passages retrieval returned
    seconds: float = 0.0


def topbar_html(app_name: str) -> str:
    return f"<div class='id-topbar'><div class='id-logo'>{esc(app_name)}</div></div>"


# --------------------------------------------------------------------------- #
# Sidebar: the loaded documents
# --------------------------------------------------------------------------- #
READING = "Reading..."


def document_list(docs, active_jobs) -> list[tuple[str, bool]]:
    """(file name, still being read) per document: every loaded document (also those loaded before a
    refresh or restart), then files still waiting or being read that have no searchable pages yet."""
    reading = {j.doc_id: j.file_name for j in active_jobs}
    loaded = {d.doc_id for d in docs}
    return ([(d.file_name, d.doc_id in reading) for d in docs] +
            [(name, True) for doc_id, name in reading.items() if doc_id not in loaded])


def document_list_html(items: list[tuple[str, bool]]) -> str:
    if not items:
        return ""
    rows = "".join(f"<div class='id-doc'><span class='name'>{esc(name)}</span>"
                   + (f"<span class='reading'>{READING}</span>" if reading else "") + "</div>"
                   for name, reading in items)
    return f"<div class='id-docs'>{rows}</div>"


# --------------------------------------------------------------------------- #
# Confidence: is a passage strong enough to show on its own?
# --------------------------------------------------------------------------- #
def confidence(score: float, exact: bool, min_score: float, fact_match_score: float) -> str:
    """High / Medium / Low from the retrieval score that already decided the answer (no new model)."""
    # rounded: 0.55 + 0.15 is 0.7000000000000001 in floating point, and 0.70 must count as High
    if exact:
        return "High" if score >= round(fact_match_score + EXACT_HIGH_MARGIN, 6) else "Medium"
    if score >= round(min_score + PASSAGE_HIGH_MARGIN, 6):
        return "High"
    return "Medium" if score >= round(min_score + PASSAGE_MEDIUM_MARGIN, 6) else "Low"


def is_strong(p: Passage, min_score: float) -> bool:
    """A passage is shown on its own (no AI answer) only with a High match: score >= MIN_SCORE + 0.15
    (0.70 with the default 0.55), where clear matches start; off-topic questions score 0.58-0.59."""
    return confidence(p.score, False, min_score, 0.0) == "High"


# --------------------------------------------------------------------------- #
# The answer card
# --------------------------------------------------------------------------- #
@dataclass
class CardView:
    """Everything one answer card shows (and what the Q&A log records)."""
    kind: str  # "exact" | "ai" | "passage" | "not_found" | "error"
    answer_html: str = ""  # the answer line (exact and AI answers only)
    answer_text: str = ""  # the same, as plain text
    file_name: str = ""
    page: int = 0
    quote_html: str = ""
    quote_text: str = ""
    message: str = ""  # not found / error: the one line shown

    @property
    def copy_text(self) -> str:
        return self.answer_text or self.quote_text


def card_view(ans: Answer, question: str, min_score: float) -> CardView:
    if ans.mode == "exact" and ans.fact:
        f: Fact = ans.fact
        return CardView("exact", exact_answer_html(f), ans.text, f.file_name, f.page, *exact_quote(f))
    if ans.mode in AI_MODES:
        text = one_sentence(clean_ai_text(ans.text))
        view = CardView("ai", _key_numbers(text), text)
        if ans.passages:
            numbers = _numbers(text)  # the cited passage that holds the answer's numbers, else the first
            best = next((p for p in ans.passages if any(_has_number(p.text, n) for n in numbers)), ans.passages[0])
            view.file_name, view.page = best.file_name, best.page
            view.quote_html, view.quote_text = answer_quote(best.text, question, text)
        return view
    if ans.mode == "sources" and ans.passages and is_strong(ans.passages[0], min_score):
        best = ans.passages[0]  # AI off, failed or not checked: the best passage, without any message
        quote_html, quote_text = answer_quote(best.text, question)
        return CardView("passage", file_name=best.file_name, page=best.page, quote_html=quote_html,
                        quote_text=quote_text)
    if ans.mode in ("not_found", "sources"):
        return CardView("not_found", message=NOT_FOUND)
    return CardView("error", message=ans.text)


def card_html(view: CardView) -> str:
    if view.kind in ("not_found", "error"):
        return f"<div class='id-nf'>{esc(view.message)}</div>"
    parts = []
    if view.answer_html:
        parts.append(f"<div class='id-answer'>{view.answer_html}</div>")
    if view.file_name:
        page = f"<span class='sep'>·</span>p. {view.page}" if view.page else ""
        parts.append(f"<div class='id-where'><span class='file'>{esc(view.file_name)}</span>{page}</div>")
    if view.quote_html:
        parts.append(f"<div class='id-qlabel'>{QUOTE_LABEL}</div><div class='id-quote'>{view.quote_html}</div>")
    return "".join(parts)


def exact_answer_html(f: Fact) -> str:
    """The answer line of an exact value: its name, then the value (and unit) in maroon."""
    if f.kind == "qa":
        return _key_numbers(f.value)
    unit = f" {esc(f.unit)}" if f.unit else ""
    name = f.name or f.item_title
    label = f"<span class='name'>{esc(name)}</span> " if name else ""
    return f"{label}<span class='key'>{esc(f.value)}{unit}</span>"


def mark_value(line: str, value: str) -> str:
    """The source line escaped, with the first occurrence of the value highlighted."""
    i = line.find(value) if value else -1
    if i < 0:
        return esc(line)
    return f"{esc(line[:i])}<mark class='val'>{esc(value)}</mark>{esc(line[i + len(value):])}"


def exact_quote(f: Fact) -> tuple[str, str]:
    """(HTML, text) of the exact value's source line, with the value marked; a long line (a whole
    paragraph of prose) is cut down to the sentence that holds the value."""
    line = f.line
    if len(line) > LONG_QUOTE:
        line = next((s for s in _sentences(line) if f.value and f.value in s), line)
    return mark_value(line, f.value), line


def answer_quote(text: str, question: str, answer: str = "") -> tuple[str, str]:
    """(HTML, text) of only the sentence(s) of a passage that hold the answer: those with the answer's
    numbers (at most two), else the one sentence that matches the question best. Exact document text."""
    sentences = _sentences(text) or [text.strip()]
    words = _content_words(question)
    picked = [s for s in sentences if any(_has_number(s, n) for n in _numbers(answer))][:2]
    if not picked:
        hits = [len({_stem(w) for w in WORD_RE.findall(s)} & words) for s in sentences]
        picked = [sentences[max(range(len(sentences)), key=lambda i: (hits[i], -i))]]
    return "<br>".join(_highlight(s, words) for s in picked), "\n".join(picked)


def _sentences(text: str) -> list[str]:
    """The sentences and list items of a text, each exactly as written (lines are split further at
    '. ' and before bullets; '10.6' is not split)."""
    out = []
    for line in text.splitlines():
        out += [s.strip() for s in SENTENCE_RE.split(line) if s.strip() and s.strip() != "•"]
    return out


def _numbers(text: str) -> list[str]:
    return list(dict.fromkeys(n.rstrip(",.") for n in NUMBER_RE.findall(text) if n.rstrip(",.")))


def _has_number(text: str, number: str) -> bool:
    return bool(re.search(rf"(?<![\d.,]){re.escape(number)}(?!\d)", text))


def one_sentence(text: str) -> str:
    """The first sentence of an AI answer (the answer line is one short sentence). A first "sentence"
    shorter than ONE_SENTENCE_MIN characters (e.g. "U.S.") is not cut off."""
    flat = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", flat)
    first = ""
    for part in parts:
        first = f"{first} {part}".strip()
        if len(first) >= ONE_SENTENCE_MIN:
            return first
    return first


def clean_ai_text(text: str) -> str:
    """The AI answer without [file, p.N] citations (the card shows the source) and without
    markdown bold markers. Digits, units and words are not changed."""
    text = CITATION_RE.sub("", text).replace("**", "")
    text = re.sub(r"[ \t]+([.,;:!?])(?=\s|$)", r"\1", text)  # "540 N·m [a.pdf]." -> "540 N·m."
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _key_numbers(text: str) -> str:
    """Escaped text with every number in maroon (the number itself is not changed)."""
    out, last = [], 0
    for m in NUMBER_RE.finditer(text):
        number = m.group().rstrip(",")
        out.append(esc(text[last:m.start()]) + f"<span class='key'>{esc(number)}</span>")
        last = m.start() + len(number)
    return "".join(out) + esc(text[last:])


# --------------------------------------------------------------------------- #
# Copy button
# --------------------------------------------------------------------------- #
# One click handler for every copy button on the page (Streamlit strips onclick="..." attributes,
# but runs scripts in st.html(..., unsafe_allow_javascript=True)). The guard installs it only once.
COPY_SCRIPT = """<script>
if (!window.__idCopy) {
  window.__idCopy = true;
  document.addEventListener('click', (e) => {
    const b = e.target.closest && e.target.closest('.id-copy');
    if (!b) return;
    navigator.clipboard.writeText(b.dataset.copy).then(() => {
      b.textContent = 'Copied ✓';
      setTimeout(() => { b.textContent = b.dataset.label; }, 1500);
    });
  });
}
</script>"""


def copy_button_html(label: str, text: str) -> str:
    return (f"<button class='id-copy' type='button' data-copy='{_attr(text)}' data-label='{_attr(label)}'>"
            f"{esc(label)}</button>{COPY_SCRIPT}")


# --------------------------------------------------------------------------- #
# The quote: best passage trimmed around the match, matched words highlighted
# --------------------------------------------------------------------------- #
def _stem(word: str) -> str:
    w = word.lower()
    return w.rstrip("s") if len(w) > 3 else w


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in WORD_RE.findall(text)
            if w.lower() not in QUESTION_STOPWORDS and (len(w) >= 3 or any(c.isdigit() for c in w))}


def _highlight(text: str, words: set[str]) -> str:
    out = []
    for piece in re.split(r"(\w+(?:[.,]\d+)*)", text):
        if piece and _stem(piece) in words and WORD_RE.fullmatch(piece):
            out.append(f"<mark>{esc(piece)}</mark>")
        else:
            out.append(esc(piece))
    return "".join(out)


def _units(text: str) -> list[str]:
    units = []
    for line in (ln for ln in text.split("\n") if ln.strip()):
        if len(line) > LONG_LINE:
            units += [s for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
        else:
            units.append(line)
    return units


def _window(text: str, question: str) -> tuple[list[str], int, int, list[int], set[str]]:
    """(lines of the passage, start of the best ~3-line window, its size, hits per line, question words)."""
    words = _content_words(question)
    units = _units(text) or [text]
    hits = [len({_stem(w) for w in WORD_RE.findall(u)} & words) for u in units]
    n = min(SNIPPET_LINES, len(units))
    start = max(range(len(units) - n + 1), key=lambda i: (sum(hits[i:i + n]), -i))
    return units, start, n, hits, words


def snippet(text: str, question: str) -> tuple[str, str]:
    """(HTML of about 3 lines around the best match, the best matching line).

    The lines are shown exactly as in the passage; '…' marks text left out.
    """
    units, start, n, hits, words = _window(text, question)
    window = units[start:start + n]
    focus = window[max(range(n), key=lambda j: hits[start + j])]
    body = "<br>".join(_highlight(u, words) for u in window)
    before = "<span class='more'>…</span><br>" if start > 0 else ""
    after = "<br><span class='more'>…</span>" if start + n < len(units) else ""
    return before + body + after, focus


def snippet_text(text: str, question: str) -> str:
    """The same lines as snippet(), as plain text (for Copy and the Q&A log)."""
    units, start, n, _, _ = _window(text, question)
    return "\n".join(units[start:start + n])


def skeleton_html() -> str:
    return ("<div class='id-skelcard'><div class='id-skel'><div class='sk w30'></div><div class='sk w90'></div>"
            "<div class='sk w75'></div><div class='sk w55'></div></div></div>")
