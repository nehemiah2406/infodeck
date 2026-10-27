"""Split pages into chunks that keep file name, page and section.

Structure first, size second:
1. Each page is cut into blocks: a heading or numbered item ("9. ...") with
   everything under it (sub-items "(1)", formulas, variable lines), or a
   question with its answer. An item continued on the next page keeps its title.
2. Blocks that fit are packed together into chunks of at most `size` characters.
3. Only a block that is too long is split (by line, then sentence, then word).
   Every part starts with the block's title, rows of a Markdown table repeat
   the table's header, and plain text parts overlap a little.

Chunks never cross a page boundary, so every chunk has exactly one page number to cite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loaders import Page

HEADING_RE = re.compile(r"^(\d{1,2})\.\s+[A-Z][^\n]{2,80}$")  # section heading, e.g. "1. Purpose and Scope"
ITEM_RE = re.compile(r"^\d{1,3}\.\s+\S")  # numbered item, e.g. "9. Our proposal of push-up distance"
MD_HEADING_RE = re.compile(r"^#{1,6}\s+(\S.*)$")
SUBITEM_RE = re.compile(r"^\(\s*(?:\d{1,2}|[a-z])\s*\)")  # "(1)", "(a)"
QUESTION_RE = re.compile(r"^(?:Q\s*\d*\s*[.:)]|Question\s*\d*\s*[.:)])\s*\S", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}")
MAX_TITLE_PREFIX = 160


@dataclass
class Chunk:
    text: str
    file_name: str
    page: int  # 0 = no page number
    section: str
    index: int  # position of the chunk within its document


@dataclass
class _Block:
    kind: str  # "item" | "qa" | "text"
    title: str  # first line, repeated at the top of every part if the block is split
    section: str
    lines: list[str] = field(default_factory=list)


def bare(line: str) -> str:
    """Line without Markdown decoration (#, **, >), for pattern matching only."""
    return re.sub(r"^[#>*_\s]+|[*_\s]+$", "", line)


def is_question(line: str) -> bool:
    b = bare(line)
    return bool(QUESTION_RE.match(b)) or (b.endswith("?") and 10 <= len(b) <= 200 and not b.startswith("|"))


def _blocks(lines: list[str], section: str, open_title: str) -> tuple[list[_Block], str]:
    blocks: list[_Block] = []
    current: _Block | None = None
    for line in lines:
        b = bare(line)
        md = MD_HEADING_RE.match(line)
        if md or ITEM_RE.match(b):
            if md or HEADING_RE.match(b):
                section = b
            current = _Block("item", line, section, [line])
            blocks.append(current)
        elif is_question(line):
            current = _Block("qa", line, section, [line])
            blocks.append(current)
        elif current is None:
            # An item continued from the previous page keeps its title.
            continued = open_title and SUBITEM_RE.match(b)
            title = f"{open_title} (continued)" if continued else ""
            current = _Block("item" if continued else "text", title, section, [title, line] if continued else [line])
            blocks.append(current)
        else:
            current.lines.append(line)
    return blocks, section


def _split_long(unit: str, size: int) -> list[str]:
    """Split a single over-long line into pieces <= size (by sentence, then by word)."""
    if len(unit) <= size:
        return [unit]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?;])\s+", unit):
        while len(sentence) > size:  # a single huge "sentence": cut at a word boundary
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap characters of text, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _split_block(block: _Block, size: int, overlap: int) -> list[str]:
    """Split a too-long block; every part starts with the block title (and table header)."""
    title = block.title[:MAX_TITLE_PREFIX] if block.kind in ("item", "qa") else ""
    lines = block.lines[1:] if title and block.lines and block.lines[0] == block.title else block.lines
    units: list[tuple[str, str]] = []  # (text, context lines repeated at the top of a new part)
    header: list[str] = []
    previous_was_table = False
    for line in lines:
        is_table = line.lstrip().startswith("|")
        if is_table and not previous_was_table:
            header = [line]
        elif is_table and len(header) == 1 and TABLE_SEPARATOR_RE.match(line.strip()):
            header.append(line)
        previous_was_table = is_table
        in_body = is_table and line not in header
        context = _join(title, *(header if in_body else []))
        for piece in _split_long(line, max(size - overlap - len(context) - 1, size // 4)):
            units.append((piece, context))

    parts: list[str] = []
    body, current = "", title
    for unit, context in units:
        if body and len(current) + 1 + len(unit) > size:
            parts.append(current)
            tail = "" if context != title else _tail(body, overlap)  # table rows get the header instead
            while tail and len(_join(context, tail, unit)) > size:
                tail = _tail(tail, len(tail) // 2)
            body = _join(tail, unit)
            current = _join(context, body)
        else:
            body = _join(body, unit)
            current = _join(current, unit)
    if body:
        parts.append(current)
    return parts


class Chunker:
    """Chunks pages one at a time (in page order), remembering the section and any
    numbered item that continues onto the next page. Used while a big PDF is still loading."""

    def __init__(self, file_name: str, size: int = 800, overlap: int = 150):
        if overlap >= size:
            raise ValueError("overlap must be smaller than chunk size")
        self.file_name, self.size, self.overlap = file_name, size, overlap
        self.section, self.open_title = "", ""
        self.count = 0  # chunks made so far (chunk index within the document)

    def add_page(self, page: Page) -> list[Chunk]:
        chunks: list[Chunk] = []

        def emit(text: str, chunk_section: str) -> None:
            chunks.append(Chunk(text, self.file_name, page.number, chunk_section, self.count))
            self.count += 1

        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in page.text.splitlines()]
        blocks, self.section = _blocks([line for line in lines if line], self.section, self.open_title)
        if blocks:
            last = blocks[-1]
            self.open_title = last.title.removesuffix(" (continued)") if last.kind == "item" else ""

        size = self.size
        current, current_section = "", ""
        for block in blocks:
            text = "\n".join(block.lines)
            if len(text) <= size and current and len(current) + 1 + len(text) <= size:
                current += "\n" + text
            elif len(text) <= size:
                if current:
                    emit(current, current_section)
                current, current_section = text, block.section
            else:
                if current:
                    emit(current, current_section)
                current = ""
                for part in _split_block(block, size, self.overlap):
                    emit(part, block.section)
        if current:
            emit(current, current_section)
        return chunks


def chunk_pages(pages: list[Page], file_name: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunker = Chunker(file_name, size, overlap)
    return [chunk for page in pages for chunk in chunker.add_page(page)]
