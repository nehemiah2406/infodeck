"""Exact-value and Q&A index.

Finds lines that state one value, and question/answer pairs, and keeps their
text exactly as written (no number is parsed into a float, so "1,118,355"
stays "1,118,355"):
- sub-items:        "(1) Push-up distance, 10.6 mm"
- formula results:  "(2) Grip stress corresponding to Pd, N/mm²   σra = Pd / Pr x σr = 111 N/mm²"
- value on the next line: "(2) Grip stress corresponding to Pd, N/mm²" then "σra = Pd / Pr x σr = 114 N/mm²"
- variable lines:   "A : Interfacial surface area ... A= 1,118,355"
- named values:     "Design pressure (P) = 25 bar", "Design pressure: 25 bar"
- table rows:       "Grade: S355; Yield strength (MPa): 355" (local) or Markdown table rows (Gemini)
- Q&A pairs:        "Q: ...?" / "A: ...", or a question line followed by its answer

A question that matches a fact's name closely enough is answered straight from
the fact, without the LLM (see InfoDeckEngine.exact_answer). match_quality() is the safety
check every exact answer must pass: the question's main subject must be in the fact's own name.
Comparison lines ("σe = 279 N/mm² ≤ σa = 345 N/mm²") give no fact: the value belongs to another symbol.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from .chunking import ITEM_RE, MD_HEADING_RE, QUESTION_RE, SUBITEM_RE, bare, is_question
from .loaders import Page

MAX_FACTS_PER_DOCUMENT = 5000

NUM = r"[-+−]?(?:\d{1,3}(?:[.,]\d{3}){2,}|\d[\d,]*(?:\.\d+)?)(?:\s*[x×]\s*10\^?[-−]?\d+)?"
UNIT = r"[^\s\d=,;()|.][^\s=,;|]*(?:\s*/\s*[^\s=,;|]+)?"
VALUE_AT_END_RE = re.compile(rf"=\s*(?P<value>{NUM})\s*(?P<unit>{UNIT})?\s*$")
NAME_VALUE_RE = re.compile(rf"^(?P<name>[^\d=:;|][^=:;|]{{1,100}}?)\s*[,:]\s*(?P<value>{NUM})\s*(?P<unit>{UNIT})?\s*$")
VARIABLE_RE = re.compile(r"^(?P<symbol>[^\s:=()]{1,12})\s+:\s+(?P<rest>.+=.+)$")
SYMBOL_IN_NAME_RE = re.compile(r"\((?P<symbol>[^\s()]{1,12})\)\s*$")
UNIT_IN_HEADER_RE = re.compile(r"^(?P<name>.*?)\s*[(\[](?P<unit>[^()\[\]]{1,12})[)\]]\s*$")
COMPARISON_RE = re.compile(r"[≤≥<>]")
GAP_RE = re.compile(r"\s{3,}")  # OCR joins side-by-side columns with 3 spaces
HELPER_RE = re.compile(r"^(?:where|∴|therefore|hence|thus)\b\s*,?\s*|^∴\s*", re.IGNORECASE)  # "where, f = ..."
ANSWER_RE = re.compile(r"^(?:A\s*\d*\s*[.:)]|Answer\s*\d*\s*[.:)])\s*", re.IGNORECASE)
VALUE_CELL_RE = re.compile(rf"^(?P<value>{NUM})(?:\s*(?P<unit>{UNIT}))?$")
KNOWN_UNITS = {"mm", "cm", "m", "km", "in", "ft", "kg", "g", "t", "n", "kn", "mn", "pa", "kpa", "mpa", "gpa",
               "bar", "psi", "%", "°c", "°f", "k", "s", "min", "h", "hz", "rpm", "kw", "w", "mw", "v", "a",
               "nm", "knm", "mm²", "mm2", "m²", "m2", "mm³", "m³", "l", "ml", "deg", "°", "rad", "days", "years",
               "kn.m", "n.m", "t.m", "ton.m", "kgf.m"}

# Words that say nothing about WHICH value is asked for.
QUESTION_STOPWORDS = set("""a an the is are was were be been do does did i my me we our you your it its this that these
those to of in on for and or what which who whom how much many when where why can could should would will shall may
must please tell give show find value values number amount according document documents sheet calc calculation
calculated per there their with by at from as about state stated listed list mentioned used use given""".split())


@dataclass
class Fact:
    kind: str  # "value" | "qa"
    name: str  # what the value is (or the question, for Q&A)
    value: str  # exact value text (or the answer, for Q&A)
    unit: str = ""
    symbol: str = ""
    formula: str = ""
    item: str = ""  # e.g. "9.(2)"
    item_title: str = ""
    line: str = ""  # the source line, exactly as read
    file_name: str = ""
    page: int = 0
    doc_id: str = ""
    unsafe: bool = False  # the upload self-test found it can be confused with another value: never shown
    checked: bool = False  # the upload self-test has run for this fact

    @property
    def label(self) -> str:
        return f"{self.file_name}, p.{self.page}" if self.page else self.file_name

    @property
    def embed_text(self) -> str:
        """The text that is embedded and compared with questions: the name (or the question)."""
        if self.kind == "qa":
            return self.name
        name = self.name or self.item_title
        return f"{name} ({self.symbol})" if self.symbol and self.symbol not in name else name

    def to_metadata(self) -> dict:
        return asdict(self)


def _unit_like(text: str) -> bool:
    t = text.strip()
    return bool(t) and " " not in t and len(t) <= 10 and (t.lower() in KNOWN_UNITS or bool(re.search(r"[/²³°%·]", t)))


def _split_name_unit(name: str) -> tuple[str, str]:
    """'Grip stress corresponding to Pd, N/mm²' -> ('Grip stress corresponding to Pd', 'N/mm²')"""
    head, sep, last = name.rpartition(",")
    last = last.strip()
    if last.startswith("(") and last.endswith(")"):  # "Interfacial surface area, (mm²)"
        last = last[1:-1]
    if sep and _unit_like(last):
        return head.strip(), last
    return name.strip(), ""


def _split_symbol(name: str) -> tuple[str, str]:
    """'Design pressure (P)' -> ('Design pressure', 'P')"""
    m = SYMBOL_IN_NAME_RE.search(name)
    return (name[:m.start()].strip(" ,"), m.group("symbol")) if m else (name, "")


def _clean_unit(unit: str | None) -> str:
    return (unit or "").strip().rstrip(".")


def parse_value_line(text: str) -> Fact | None:
    """Parse one line (without its "(n)" marker) into a value fact, or None."""
    text = text.strip().rstrip(".").strip()
    if COMPARISON_RE.search(text):  # "σe = 279 N/mm² ≤ σyt = 345 N/mm²": 345 is not σe
        return None
    text = HELPER_RE.sub("", text)  # "where, f = 0.534": "where" is not a name
    m = VALUE_AT_END_RE.search(text)
    if m and m.start() > 0:
        value, unit = m.group("value"), _clean_unit(m.group("unit"))
        left = text[:m.start()].rstrip()
        if left.count("(") > left.count(")"):  # "345 N/mm² (= 35 kgf/mm²)": the same value in another unit
            return None
        eq = left.find("=")
        gaps = [g.end() for g in GAP_RE.finditer(left) if eq >= 0 and g.start() > eq]
        if gaps:  # "t = { ... } + 1   ① = 2.8": two equations side by side; 2.8 is ①, not t
            return parse_value_line(left[gaps[-1]:] + " " + text[m.start():])
        lhs = left[:eq].rstrip() if eq >= 0 else left
        tokens = lhs.split()
        if not tokens:
            return None
        token = tokens[-1]
        if re.fullmatch(r"[-+−]?[\d.,]+", token):  # "angle,   0.03 =0": a number is not a symbol
            return None
        symbol = token.strip("()")  # "Design pressure (P) = 25 bar"
        formula = f"{symbol} {left[eq:].strip()}" if eq >= 0 else ""
        name, unit_hint = _split_name_unit(lhs[: len(lhs) - len(token)].strip(" ,:"))
        if not name and len(symbol) > 12:  # "Long name = 5 mm": no symbol, the whole left side is the name
            name, symbol = lhs, ""
        name, symbol_in_name = _split_symbol(name)
        symbol = symbol or symbol_in_name
        return Fact("value", name, value, unit or unit_hint, symbol, formula)
    m = NAME_VALUE_RE.match(text)
    if m:
        name, symbol = _split_symbol(m.group("name").strip())
        return Fact("value", name, m.group("value"), _clean_unit(m.group("unit")), symbol)
    return None


def _header_parts(text: str) -> tuple[str, str, str]:
    """'Grip stress corresponding to Pd, N/mm²' -> (name, symbol, unit) for an item line without a value.

    'Diameter interference, (Id), mm' -> ('Diameter interference', 'Id', 'mm'); a remark after ' : '
    ('..., N/mm² : from LR rule of') is dropped.
    """
    text = text.split(" : ")[0].strip(" ,:")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return "", "", ""
    unit = parts.pop() if len(parts) > 1 and _unit_like(parts[-1]) else ""
    symbol = ""
    if len(parts) > 1 and re.fullmatch(r"\(\S{1,12}\)", parts[-1]):
        symbol = parts.pop().strip("()")
    name, symbol_in_name = _split_symbol(", ".join(parts))
    return name, symbol or symbol_in_name, unit


def _similar_symbol(a: str, b: str) -> bool:
    """'σe' and 'oe' (OCR often reads σ as o): equal, or equal except the first character."""
    a, b = _norm(a), _norm(b)
    return a == b or (len(a) == len(b) >= 2 and a[1:] == b[1:])


def _formula_for_header(header: tuple[str, str, str], text: str) -> Fact | None:
    """The value of a header line ('(2) Grip stress ..., N/mm²') from the formula on the next line
    ('σra = Pd / Pr x σr = 114 N/mm²'), only if the formula clearly belongs to it: the units agree,
    or the symbols agree. Helper lines ('where, ...') and named lines never count."""
    name, symbol, unit = header
    if not name or "=" not in text or HELPER_RE.match(text):
        return None
    f = parse_value_line(text)
    if f is None or f.name or not f.formula:
        return None
    if unit and f.unit and _norm(unit) != _norm(f.unit):
        return None
    if symbol and f.symbol and not _similar_symbol(symbol, f.symbol):
        return None
    if not ((unit and f.unit) or (symbol and f.symbol)):
        return None  # nothing ties this line to the header
    return Fact("value", name, f.value, f.unit or unit, f.symbol or symbol, f.formula)


def _variable_fact(text: str) -> Fact | None:
    """'A : Interfacial surface area ... A= 1,118,355' or 'Pr : Contact pressure, N/mm² = 45.2'"""
    m = VARIABLE_RE.match(text.strip())
    if not m:
        return None
    symbol, rest = m.group("symbol"), m.group("rest")
    end = VALUE_AT_END_RE.search(rest)
    if not end:
        return None
    left = rest[:end.start()].rstrip()
    if re.search(r"\d$", left):  # "θ : Cone taper half angle,   0.03 =0": two numbers, which one is θ?
        return None
    if left.endswith(symbol):  # "... A= 1,118,355": drop the repeated symbol
        left = left[: -len(symbol)]
    if "=" in left:  # "Interfacial surface area, (mm²), A = π x dm x Lt": the name ends before the formula
        left = left[: left.find("=")].rstrip()
        left = left[: -len(symbol)] if left.endswith(symbol) else left
    name, unit_hint = _split_name_unit(re.sub(r"(\s*(\.{2,}|…+))+\s*$", "", left).strip(" ,:"))
    return Fact("value", name, end.group("value"), _clean_unit(end.group("unit")) or unit_hint, symbol)


def _table_row_facts(cells: list[tuple[str, str]]) -> list[Fact]:
    """Facts from one table row given (header, cell) pairs: one per numeric cell."""
    def value_cell(text: str):
        m = VALUE_CELL_RE.match(text)
        return m if m and (not m.group("unit") or _unit_like(m.group("unit"))) else None

    label_header, label = next(((h, v) for h, v in cells if v and not value_cell(v)), ("", ""))
    label = f"{label_header} {label}".strip()  # "Bolt size M20"
    facts = []
    for pos, (header, cell) in enumerate(cells):
        m = value_cell(cell)
        if not m or not header:
            continue
        h = UNIT_IN_HEADER_RE.match(header)
        name, unit = (h.group("name"), h.group("unit")) if h and _unit_like(h.group("unit")) else (header, "")
        unit = _clean_unit(m.group("unit")) or unit
        if not unit and pos + 1 < len(cells) and cells[pos + 1][0].lower().startswith("unit"):
            unit = cells[pos + 1][1]  # a "Unit" column right after the value
        facts.append(Fact("value", f"{label} {name}".strip() if label else name, m.group("value"), unit))
    return facts


def _markdown_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


class FactExtractor:
    """Extracts facts page by page (in page order), remembering the current numbered item and
    table header across pages. Used while a big PDF is still loading."""

    def __init__(self, file_name: str, doc_id: str = ""):
        self.file_name, self.doc_id = file_name, doc_id
        self.item_no, self.item_title = "", ""
        self.table_header: list[str] = []
        self.pending: tuple | None = None  # (line, sub-item, (name, symbol, unit), lines seen) of a header without value
        self.count = 0  # facts found so far in this document

    def add_page(self, page: Page) -> list[Fact]:
        facts: list[Fact] = []

        def add(fact: Fact | None, line: str, sub: str = "") -> None:
            if fact is None or not (fact.name or fact.symbol or self.item_title):
                return
            if self.count >= MAX_FACTS_PER_DOCUMENT:
                return
            item_no = self.item_no
            fact.item = f"{item_no}.({sub})" if item_no and sub else (f"{item_no}." if item_no else "")
            fact.item_title, fact.line = self.item_title, line.strip()
            fact.file_name, fact.page, fact.doc_id = self.file_name, page.number, self.doc_id
            facts.append(fact)
            self.count += 1

        lines = [line.strip() for line in page.text.splitlines()]
        i = 0
        while i < len(lines):
            line, b = lines[i], bare(lines[i])
            i += 1
            if not b:
                continue
            # The value of the item line before, on this line ("σra = Pd / Pr x σr = 114 N/mm²")
            if self.pending:
                header_line, sub_no, header, gap = self.pending
                if ITEM_RE.match(b) or SUBITEM_RE.match(b) or is_question(line) or b.startswith("|"):
                    self.pending = None
                elif "=" in b:
                    self.pending = None
                    fact = _formula_for_header(header, b)
                    if fact:
                        add(fact, f"{header_line}\n{line}", sub_no)
                        continue
                elif gap < 2:  # e.g. "Pt5, Ch19, 1.6.5" continuing the header
                    self.pending = (header_line, sub_no, header, gap + 1)
                else:
                    self.pending = None
            # Q&A: the question and every line until the next question/item/heading
            if is_question(line) and not b.startswith("|") and not VARIABLE_RE.match(b):  # "Q : Shear force = ..." is a variable
                answer = []
                while i < len(lines) and lines[i] and not (is_question(lines[i]) or ITEM_RE.match(bare(lines[i]))
                                                        or MD_HEADING_RE.match(lines[i])):
                    answer.append(lines[i])
                    i += 1
                if answer:
                    question = re.sub(r"^(?:Q\s*\d*\s*[.:)]|Question\s*\d*\s*[.:)])\s*", "", b, flags=re.I)
                    first = ANSWER_RE.sub("", bare(answer[0]), count=1)
                    text = "\n".join([first] + [bare(a) for a in answer[1:]]).strip()
                    add(Fact("qa", question.strip(), text), line)
                continue
            # Markdown tables (Gemini output)
            if b.startswith("|") or line.startswith("|"):
                cells = _markdown_cells(line)
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                    continue
                if not self.table_header or (i >= 2 and not lines[i - 2].startswith("|")):
                    self.table_header = cells
                    continue
                for f in _table_row_facts(list(zip(self.table_header, cells))):
                    add(f, line)
                continue
            self.table_header = []
            # Numbered item "9. Title": remember it; the title itself may hold a value
            m = re.match(r"^(\d{1,3})\.\s+(.+)$", b)
            if m and ITEM_RE.match(b):
                self.item_no, self.item_title = m.group(1), m.group(2).strip()
                if "=" in self.item_title:
                    add(parse_value_line(self.item_title), line)
                else:  # "7. Diameter interference, (Id), mm": its value may follow on the next line
                    self.pending = (line, "", _header_parts(self.item_title), 0)
                continue
            if MD_HEADING_RE.match(line):
                self.item_no, self.item_title = "", b
                continue
            # Sub-item "(1) Push-up distance, 10.6 mm"
            m = SUBITEM_RE.match(b)
            if m:
                sub = m.group(0).strip("() ")
                fact = parse_value_line(b[m.end():])
                if fact:
                    add(fact, line, sub)
                else:  # "(2) Grip stress corresponding to Pd, N/mm²": the value may follow on the next line
                    self.pending = (line, sub, _header_parts(b[m.end():]), 0)
                continue
            # Local table rows "Header: value; Header: value"
            if ";" in b and b.count(":") >= 2:
                pairs = [p.split(":", 1) for p in b.split(";") if ":" in p]
                for f in _table_row_facts([(h.strip(), v.strip()) for h, v in pairs]):
                    add(f, line)
                continue
            add(_variable_fact(b) or parse_value_line(b), line)
        return facts


def extract_facts(pages: list[Page], file_name: str, doc_id: str = "") -> list[Fact]:
    extractor = FactExtractor(file_name, doc_id)
    return [fact for page in pages for fact in extractor.add_page(page)]


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
# "grip stress CORRESPONDING TO Pd", "push-up force BASED ON Pd", "k = La / h": what follows is only a
# reference; the words before it are the question's main subject.
REFERENCE_RE = re.compile(r"\b(?:corresponding to|corresponds to|according to|based on|due to|related to|"
                          r"relative to|for|at|of|in|on|with|from|under|when|during)\b|=", re.IGNORECASE)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()  # NFKC: the micro sign µ becomes μ


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\w²³°%]+", _norm(text)) if w not in QUESTION_STOPWORDS]


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "al", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _same_word(a: str, b: str) -> bool:
    """'proposed' ~ 'proposal', 'coresponding' (OCR) ~ 'corresponding'. Words only: numbers must be equal."""
    if a == b:
        return True
    if any(c.isdigit() for c in a + b):
        return False
    if _stem(a) == _stem(b):
        return True
    return min(len(a), len(b)) >= 6 and SequenceMatcher(None, a, b).ratio() >= 0.85


def _in(word: str, vocabulary: list[str]) -> bool:
    return any(_same_word(word, v) for v in vocabulary)


def question_subject(question: str) -> list[str]:
    """The main subject words: the content words before 'corresponding to', 'for', 'of', '=' ...
    ('What is the grip stress corresponding to Pd?' -> ['grip', 'stress'])."""
    m = REFERENCE_RE.search(question)
    subject = _content_words(question[: m.start()]) if m else []
    return subject or _content_words(question)


def match_quality(question: str, fact: Fact) -> str | None:
    """The safety check for an exact answer: "tight", "loose", or None (this fact must not answer).

    1. Every content word of the question appears in the fact (name, symbol, unit, item title, formula).
    2. The main subject (question_subject) is in the fact's own name or item title, and at least one
       subject word is in its OWN name or symbol. So 'grip stress corresponding to Pd' never matches
       the item 'Push-up distance, Pd': Pd is only a reference.
    "tight": every word of the fact's own name is also in the question, or the question is just the
    fact's symbol ("What is Pd?"). Tight matches are preferred over "loose" ones.
    """
    own = _content_words(fact.name) + ([_norm(fact.symbol)] if fact.symbol else [])
    if fact.kind == "qa":
        own += _content_words(fact.value)  # "How many years ...?" is answered by "... 10 years ..."
    title = _content_words(f"{fact.item_title} {fact.item}")
    everything = own + title + _content_words(f"{fact.unit} {fact.formula}")
    words = _content_words(question)
    if not words or not all(_in(w, everything) for w in words):
        return None
    subject = question_subject(question)
    if not all(_in(w, own + title) for w in subject) or not any(_in(w, own) for w in subject):
        return None
    if fact.symbol and subject == [_norm(fact.symbol)]:  # "What is Pd?": this item IS Pd
        return "tight"
    name_words = _content_words(fact.name)
    return "tight" if name_words and all(_in(w, words) for w in name_words) else "loose"


def symbol_in_question(question: str, symbol: str) -> bool:
    """'What is σra?' names the symbol σra exactly (symbols of 2+ characters only)."""
    return len(symbol) >= 2 and re.search(rf"(?<![\w]){re.escape(symbol)}(?![\w])", question) is not None


def format_fact_answer(fact: Fact) -> str:
    if fact.kind == "qa":
        return fact.value
    return f"{fact.value} {fact.unit}".strip()
