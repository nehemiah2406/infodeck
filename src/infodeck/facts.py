"""Exact-value and Q&A index.

Finds lines that state one value, and question/answer pairs, and keeps their
text exactly as written (no number is parsed into a float, so "1,118,355"
stays "1,118,355"):
- sub-items:        "(1) Push-up distance, 10.6 mm"
- formula results:  "(2) Grip stress corresponding to Pd, N/mm²   σra = Pd / Pr x σr = 111 N/mm²"
- variable lines:   "A : Interfacial surface area ... A= 1,118,355"
- named values:     "Design pressure (P) = 25 bar", "Design pressure: 25 bar"
- table rows:       "Grade: S355; Yield strength (MPa): 355" (local) or Markdown table rows (Gemini)
- Q&A pairs:        "Q: ...?" / "A: ...", or a question line followed by its answer

A question that matches a fact's name closely enough is answered straight from
the fact, without the LLM (see InfoDeckEngine.exact_answer).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .chunking import ITEM_RE, MD_HEADING_RE, QUESTION_RE, SUBITEM_RE, bare, is_question
from .loaders import Page

MAX_FACTS_PER_DOCUMENT = 5000

NUM = r"[-+−]?\d[\d,]*(?:\.\d+)?(?:\s*[x×]\s*10\^?[-−]?\d+)?"
UNIT = r"[^\s\d=,;()|][^\s=,;|]*(?:\s*/\s*[^\s=,;|]+)?"
VALUE_AT_END_RE = re.compile(rf"=\s*(?P<value>{NUM})\s*(?P<unit>{UNIT})?\s*$")
NAME_VALUE_RE = re.compile(rf"^(?P<name>[^\d=:;|][^=:;|]{{1,100}}?)\s*[,:]\s*(?P<value>{NUM})\s*(?P<unit>{UNIT})?\s*$")
VARIABLE_RE = re.compile(r"^(?P<symbol>[^\s:=()]{1,12})\s+:\s+(?P<rest>.+=.+)$")
SYMBOL_IN_NAME_RE = re.compile(r"\((?P<symbol>[^\s()]{1,12})\)\s*$")
UNIT_IN_HEADER_RE = re.compile(r"^(?P<name>.*?)\s*[(\[](?P<unit>[^()\[\]]{1,12})[)\]]\s*$")
ANSWER_RE = re.compile(r"^(?:A\s*\d*\s*[.:)]|Answer\s*\d*\s*[.:)])\s*", re.IGNORECASE)
VALUE_CELL_RE = re.compile(rf"^(?P<value>{NUM})(?:\s*(?P<unit>{UNIT}))?$")
KNOWN_UNITS = {"mm", "cm", "m", "km", "in", "ft", "kg", "g", "t", "n", "kn", "mn", "pa", "kpa", "mpa", "gpa",
               "bar", "psi", "%", "°c", "°f", "k", "s", "min", "h", "hz", "rpm", "kw", "w", "mw", "v", "a",
               "nm", "knm", "mm²", "mm2", "m²", "m2", "mm³", "m³", "l", "ml", "deg", "°", "rad", "days", "years"}

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
    if sep and _unit_like(last):
        return head.strip(), last.strip()
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
    m = VALUE_AT_END_RE.search(text)
    if m and m.start() > 0:
        value, unit = m.group("value"), _clean_unit(m.group("unit"))
        left = text[:m.start()].rstrip()
        eq = left.find("=")
        lhs = left[:eq].rstrip() if eq >= 0 else left
        tokens = lhs.split()
        if not tokens:
            return None
        token = tokens[-1]
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
    if left.endswith(symbol):  # "... A= 1,118,355": drop the repeated symbol
        left = left[: -len(symbol)]
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


def extract_facts(pages: list[Page], file_name: str, doc_id: str = "") -> list[Fact]:
    facts: list[Fact] = []
    item_no, item_title = "", ""
    table_header: list[str] = []

    def add(fact: Fact | None, line: str, page: Page, sub: str = "") -> None:
        if fact is None or not (fact.name or fact.symbol or item_title):
            return
        fact.item = f"{item_no}.({sub})" if item_no and sub else (f"{item_no}." if item_no else "")
        fact.item_title, fact.line = item_title, line.strip()
        fact.file_name, fact.page, fact.doc_id = file_name, page.number, doc_id
        facts.append(fact)

    for page in pages:
        lines = [line.strip() for line in page.text.splitlines()]
        i = 0
        while i < len(lines):
            line, b = lines[i], bare(lines[i])
            i += 1
            if not b:
                continue
            # Q&A: the question and every line until the next question/item/heading
            if is_question(line) and not b.startswith("|"):
                answer = []
                while i < len(lines) and lines[i] and not (is_question(lines[i]) or ITEM_RE.match(bare(lines[i]))
                                                        or MD_HEADING_RE.match(lines[i])):
                    answer.append(lines[i])
                    i += 1
                if answer:
                    question = re.sub(r"^(?:Q\s*\d*\s*[.:)]|Question\s*\d*\s*[.:)])\s*", "", b, flags=re.I)
                    first = ANSWER_RE.sub("", bare(answer[0]), count=1)
                    text = "\n".join([first] + [bare(a) for a in answer[1:]]).strip()
                    add(Fact("qa", question.strip(), text), line, page)
                continue
            # Markdown tables (Gemini output)
            if b.startswith("|") or line.startswith("|"):
                cells = _markdown_cells(line)
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                    continue
                if not table_header or (i >= 2 and not lines[i - 2].startswith("|")):
                    table_header = cells
                    continue
                for f in _table_row_facts(list(zip(table_header, cells))):
                    add(f, line, page)
                continue
            table_header = []
            # Numbered item "9. Title": remember it; the title itself may hold a value
            m = re.match(r"^(\d{1,3})\.\s+(.+)$", b)
            if m and ITEM_RE.match(b):
                item_no, item_title = m.group(1), m.group(2).strip()
                add(parse_value_line(item_title) if "=" in item_title else None, line, page)
                continue
            if MD_HEADING_RE.match(line):
                item_no, item_title = "", b
                continue
            # Sub-item "(1) Push-up distance, 10.6 mm"
            m = SUBITEM_RE.match(b)
            if m:
                sub = m.group(0).strip("() ")
                add(parse_value_line(b[m.end():]), line, page, sub)
                continue
            # Local table rows "Header: value; Header: value"
            if ";" in b and b.count(":") >= 2:
                pairs = [p.split(":", 1) for p in b.split(";") if ":" in p]
                for f in _table_row_facts([(h.strip(), v.strip()) for h, v in pairs]):
                    add(f, line, page)
                continue
            add(_variable_fact(b) or parse_value_line(b), line, page)
    return facts[:MAX_FACTS_PER_DOCUMENT]


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def _words(text: str) -> set[str]:
    words = set()
    for w in re.findall(r"[\w²³°%]+", text.lower()):
        if w in QUESTION_STOPWORDS:
            continue
        words.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return words


def question_covered(question: str, fact: Fact) -> bool:
    """True if every content word of the question appears in the fact (name, symbol, unit, item title).

    Guards against near misses such as "push-up force" matching "Push-up distance".
    """
    parts = [fact.name, fact.symbol, fact.unit, fact.item_title, fact.item]
    if fact.kind == "qa":
        parts.append(fact.value)  # "How many years ...?" is answered by "... 10 years ..."
    vocabulary = _words(" ".join(parts))
    return _words(question) <= vocabulary


def symbol_in_question(question: str, symbol: str) -> bool:
    """'What is σra?' names the symbol σra exactly (symbols of 2+ characters only)."""
    return len(symbol) >= 2 and re.search(rf"(?<![\w]){re.escape(symbol)}(?![\w])", question) is not None


def format_fact_answer(fact: Fact) -> str:
    if fact.kind == "qa":
        return fact.value
    return f"{fact.value} {fact.unit}".strip()
