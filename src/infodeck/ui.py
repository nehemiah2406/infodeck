"""HTML pieces for the answer UI (top bar, process trail, answer cards, citations, snippets).

Pure functions, so they can be tested without Streamlit. Every piece of document or
answer text goes through esc() (or _highlight, which escapes too): numbers, units and
quotes are shown exactly as they are in the Answer, never changed.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from .engine import BRACKETS_RE, NUMBER_RE, Answer
from .facts import QUESTION_STOPWORDS, Fact
from .prompts import NOT_FOUND, SOURCES_MODE_NOTE
from .store import Passage

# Citations written by the LLM, e.g. [KRT_Leave_Policy.pdf, p.2] or [a.pdf, p.1; b.pdf, p.3]
CITATION_RE = re.compile(r"\[([^\[\]]+?\.(?:pdf|docx|txt|xlsx|csv|pptx|png|jpe?g)\b[^\[\]]*)\]", re.IGNORECASE)
AI_MODES = {"claude": "Claude", "gemini": "Gemini", "ollama": "Ollama"}
EXACT_HIGH_MARGIN = 0.10  # exact value: High if the match score is this far above FACT_MATCH_SCORE
# Passages: High / Medium if the best score is this far above MIN_SCORE. Measured with bge-small on
# the sample docs: clear matches score 0.70-0.79, off-topic questions that still pass MIN_SCORE 0.58-0.59.
PASSAGE_HIGH_MARGIN = 0.15
PASSAGE_MEDIUM_MARGIN = 0.05
SNIPPET_LINES = 3
LONG_LINE = 300  # a "line" longer than this is split into sentences for the snippet
MAX_FOLLOW_UPS = 3
FOLLOW_UP_WORDS = 7
NOT_FOUND_TIPS = [
    "Use the words the document uses (item name, symbol or heading).",
    "Check that the document with the answer is uploaded.",
    "Ask about one value or topic at a time.",
]
WORD_RE = re.compile(r"\w+(?:[.,]\d+)*")


def esc(text: str) -> str:
    """Escape document text for safe HTML display (no injected HTML/markdown/LaTeX)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def _attr(text: str) -> str:
    return html.escape(text, quote=True).replace("$", "&#36;")


@dataclass
class Trace:
    """What happened while answering one question (shown as the process trail)."""
    question: str = ""
    docs: int = 0  # documents searched
    retrieved: list[Passage] = field(default_factory=list)  # all passages retrieval returned
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Top bar, trail, tags, confidence
# --------------------------------------------------------------------------- #
def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def topbar_html(app_name: str, docs: int, pages: int, ai_label: str, ai_tone: str, ai_tip: str = "") -> str:
    return (f"<div class='id-topbar'><div class='id-logo'><span class='glyph'></span>{esc(app_name)}</div>"
            f"<div class='id-livepills'><span class='id-live'>{plural(docs, 'doc')}</span>"
            f"<span class='id-live'>{plural(pages, 'page')} indexed</span>"
            f"<span class='id-live {ai_tone}' title='{_attr(ai_tip)}'><span class='dot'></span>{esc(ai_label)}</span>"
            f"</div></div>")


def trail_steps(ans: Answer, trace: Trace) -> list[tuple[str, str]]:
    """(step text, tone) pairs; tone is ok | warn | bad."""
    searched = (f"Searched {plural(trace.docs, 'document')}", "ok")
    if ans.mode == "exact" and ans.fact:
        f = ans.fact
        matched = f"Matched item {f.item}" if f.item else f"Matched “{_short(f.name or f.item_title, 6)}”"
        verified = ("Value verified in source", "ok") if f.value and f.value in f.line else ("Value from source line", "ok")
        return [searched, (matched, "ok"), verified]
    if ans.mode in AI_MODES:
        checked = ("Numbers checked", "ok") if not ans.unverified else ("Numbers to check", "warn")
        return [searched, (f"Read {plural(len(trace.retrieved), 'passage')}", "ok"),
                (f"{AI_MODES[ans.mode]} wrote answer", "ok"), checked]
    if ans.mode == "sources":
        why = sources_tag(ans)
        # The passages shown (sources mode shows the best few), so the count matches "All passages (N)"
        return [searched, (f"Found {plural(len(ans.passages), 'passage')}", "ok"), (why, "warn")]
    if ans.mode == "not_found":
        return [searched, ("No match in the documents", "bad")]
    return [("Something went wrong", "bad")]


def trail_html(steps: list[tuple[str, str]], seconds: float) -> str:
    parts = [f"<span class='id-step {tone}'><span class='dot'></span>{esc(text)}</span>" for text, tone in steps]
    return (f"<div class='id-trail'>{'<span class=sep>›</span>'.join(parts)}"
            f"<span class='id-time'>· {seconds:.1f} s</span></div>")


def is_quota(note: str) -> bool:
    return "quota" in note.lower()


def sources_tag(ans: Answer) -> str:
    """Label of a sources-mode card: why there is no AI answer."""
    return "Quota used up" if is_quota(ans.note) else "AI answer failed" if ans.ai_failed else "AI off"


def sources_reason(ans: Answer) -> str:
    """The message under a sources-mode card."""
    if ans.ai_failed:  # e.g. "Gemini is busy right now. Showing the matching passages instead. ..."
        return f"AI answer failed: {ans.note}"
    return SOURCES_MODE_NOTE + (f" {ans.note}" if ans.note else "")


def tag_html(text: str, tone: str) -> str:
    return f"<span class='id-tag {tone}'><span class='dot'></span>{esc(text)}</span>"


def confidence(score: float, exact: bool, min_score: float, fact_match_score: float) -> str:
    """High / Medium / Low from the retrieval score that already decided the answer (no new model)."""
    if exact:
        return "High" if score >= fact_match_score + EXACT_HIGH_MARGIN else "Medium"
    if score >= min_score + PASSAGE_HIGH_MARGIN:
        return "High"
    return "Medium" if score >= min_score + PASSAGE_MEDIUM_MARGIN else "Low"


def meter_html(level: str) -> str:
    tip = "How closely the best source matches your question (from the search scores)"
    return (f"<span class='id-meter {level.lower()}' title='{tip}'><i></i><i></i><i></i>"
            f"<b>{level}</b></span>")


def card_head_html(tags: list[str], level: str | None) -> str:
    return (f"<div class='id-head'><div class='id-tags'>{''.join(tags)}</div>"
            f"{meter_html(level) if level else ''}</div>")


# --------------------------------------------------------------------------- #
# Exact value
# --------------------------------------------------------------------------- #
def mark_value(line: str, value: str) -> str:
    """The source line escaped, with the first occurrence of the value highlighted."""
    i = line.find(value) if value else -1
    if i < 0:
        return esc(line)
    return f"{esc(line[:i])}<mark class='val'>{esc(value)}</mark>{esc(line[i + len(value):])}"


def exact_html(ans: Answer) -> str:
    """Body of the exact-value card: label line, the value, 'matches' line, quote block."""
    f: Fact = ans.fact
    if f.kind == "qa":
        label = esc(f.name)
        value = f"<div class='id-xvalue qa'>{esc(f.value)}</div>"
    else:
        bits = [esc(f.name or f.item_title)]
        if f.symbol:
            bits.append(f"<span class='sym'>{esc(f.symbol)}</span>")
        if f.item:
            bits.append(f"Item {esc(f.item)}")
        label = " · ".join(bits)
        unit = f"<span class='unit'>{esc(f.unit)}</span>" if f.unit else ""
        value = f"<div class='id-xvalue'><span class='num'>{esc(f.value)}</span>{unit}</div>"
    formula = f"<div class='id-formula'>{esc(f.formula)}</div>" if f.formula else ""
    verified = ("<div class='id-verified'>✓ Matches the source word for word</div>" if f.value and f.value in f.line
                else "<div class='id-verified'>Taken from the source line below</div>")
    return (f"<div class='id-xlabel'>{label}</div>{value}{formula}{verified}"
            f"<div class='id-quote'>{mark_value(f.line, f.value)}</div>")


def copy_label(ans: Answer) -> str:
    if ans.mode == "exact" and ans.fact and ans.fact.kind != "qa":
        return f"Copy {_short(ans.text, 4)}"
    return "Copy"


def copy_text(ans: Answer) -> str:
    if ans.mode == "sources" and ans.passages:
        return ans.passages[0].text
    return ans.text


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
# AI answer: numbered citations and source cards
# --------------------------------------------------------------------------- #
def unique_sources(passages: list[Passage]) -> list[Passage]:
    """One entry per citation label (file + page), best score first kept in order."""
    seen: dict[str, Passage] = {}
    for p in passages:
        if p.label not in seen or p.score > seen[p.label].score:
            seen[p.label] = p
    order = list(dict.fromkeys(p.label for p in passages))
    return [seen[label] for label in order]


def cited_html(text: str, sources: list[Passage]) -> str:
    """LLM answer (markdown-safe) with [file, p.N] citations turned into numbered badges."""
    numbers = {p.label: i for i, p in enumerate(sources, 1)}
    safe = html.escape(text, quote=False).replace("$", "&#36;")

    def badges(m: re.Match) -> str:
        out = []
        for part in (x.strip() for x in m.group(1).split(";")):
            if not part:
                continue
            n = numbers.get(html.unescape(part))
            out.append(f"<span class='cite' title='{_attr(html.unescape(part))}'>{n}</span>" if n
                       else f"<span class='chip'>{part}</span>")
        return "".join(out)

    return CITATION_RE.sub(badges, safe)


def has_numbers(text: str) -> bool:
    return bool(NUMBER_RE.search(BRACKETS_RE.sub(" ", text)))


def source_cards_html(sources: list[Passage]) -> str:
    cards = []
    for i, p in enumerate(sources, 1):
        where = [f"p. {p.page}"] if p.page else []
        if p.section:
            where.append(esc(_short(p.section, 6)))
        width = max(0.0, min(1.0, p.score)) * 100
        cards.append(f"<div class='id-src'><div class='top'><span class='n'>{i}</span>"
                     f"<span class='file' title='{_attr(p.file_name)}'>{esc(p.file_name)}</span></div>"
                     f"<div class='where'>{' · '.join(where) or '&nbsp;'}</div>"
                     f"<div class='bar' title='relevance {p.score:.2f}'><i style='width:{width:.0f}%'></i></div></div>")
    return f"<div class='id-srcgrid'>{''.join(cards)}</div>"


def follow_ups(question: str, passages: list[Passage]) -> list[str]:
    """Up to 3 short questions from the headings of the retrieved passages (no LLM call).

    Headings whose words are all already in the question are skipped (that is the topic
    just asked about).
    """
    asked = _content_words(question)
    out: list[str] = []
    seen: set[frozenset] = set()
    for p in passages:
        heading = _clean_heading(p.section)
        words = _content_words(heading)
        if not words or words <= asked or frozenset(words) in seen:
            continue
        seen.add(frozenset(words))
        out.append(f"What about {heading}?")
        if len(out) == MAX_FOLLOW_UPS:
            break
    return out


def _clean_heading(section: str) -> str:
    h = re.sub(r"^\s*(?:(?:section|item|part)\s+)?(?:\d+(?:\.\d+)*\.?|\(\w{1,3}\)|[A-Z]\.)\s+", "",
               section.strip(), flags=re.IGNORECASE)
    first = re.split(r"\s*[,:;(]\s*", h, maxsplit=1)[0]
    if len(first.split()) >= 2:
        h = first
    h = _short(h.rstrip(" .:-"), FOLLOW_UP_WORDS)
    # Title Case headings read oddly inside a question: lower-case the words, keep acronyms like "IT"
    return " ".join(w if len(w) > 1 and w.isupper() else w.lower() for w in h.split())


def _short(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


# --------------------------------------------------------------------------- #
# Sources mode: best passage trimmed around the match, matched words highlighted
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


def snippet(text: str, question: str) -> tuple[str, str]:
    """(HTML of about 3 lines around the best match, the best matching line).

    The lines are shown exactly as in the passage; '…' marks text left out.
    """
    words = _content_words(question)
    units = _units(text) or [text]
    hits = [len({_stem(w) for w in WORD_RE.findall(u)} & words) for u in units]
    n = min(SNIPPET_LINES, len(units))
    start = max(range(len(units) - n + 1), key=lambda i: (sum(hits[i:i + n]), -i))
    window = units[start:start + n]
    focus = window[max(range(n), key=lambda j: hits[start + j])]
    body = "<br>".join(_highlight(u, words) for u in window)
    before = "<span class='more'>…</span><br>" if start > 0 else ""
    after = "<br><span class='more'>…</span>" if start + n < len(units) else ""
    return before + body + after, focus


def source_line_html(p: Passage) -> str:
    where = [esc(p.file_name)] + ([f"p. {p.page}"] if p.page else []) + ([esc(p.section)] if p.section else [])
    return f"<div class='id-srcline'>{' · '.join(where)}</div>"


# --------------------------------------------------------------------------- #
# Not found, loading
# --------------------------------------------------------------------------- #
def not_found_html() -> str:
    tips = "".join(f"<li>{esc(t)}</li>" for t in NOT_FOUND_TIPS)
    return f"<div class='id-nf'>{esc(NOT_FOUND)}</div><ul class='id-tips'>{tips}</ul>"


def skeleton_html() -> str:
    return ("<div class='id-skelcard'><div class='id-skel'><div class='sk w30'></div><div class='sk w90'></div>"
            "<div class='sk w75'></div><div class='sk w55'></div></div></div>")


def preview_skeleton_html() -> str:
    return "<div class='id-prevskel'><div class='sk'></div></div>"
