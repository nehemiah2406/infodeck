"""Split pages into overlapping chunks that keep file name, page and section.

Chunks never cross a page boundary, so every chunk has exactly one page number
to cite. Text is split at line boundaries where possible (then sentences, then
words) so chunks don't cut sentences in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Page

HEADING_RE = re.compile(r"^(\d{1,2})\.\s+[A-Z][^\n]{2,80}$")


@dataclass
class Chunk:
    text: str
    file_name: str
    page: int  # 0 = no page number
    section: str
    index: int  # position of the chunk within its document


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


def chunk_pages(pages: list[Page], file_name: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    section = ""
    for page in pages:
        units: list[tuple[str, str]] = []  # (line, section the line belongs to)
        for line in page.text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line:
                continue
            if HEADING_RE.match(line):
                section = line
            for piece in _split_long(line, size - overlap):
                units.append((piece, section))

        current, current_section = "", units[0][1] if units else section
        for unit, unit_section in units:
            if current and len(current) + 1 + len(unit) > size:
                chunks.append(Chunk(current, file_name, page.number, current_section, len(chunks)))
                current = f"{_tail(current, overlap)}\n{unit}".strip()
                current_section = unit_section
            else:
                current = f"{current}\n{unit}".strip()
                if not current_section:
                    current_section = unit_section
        if current:
            chunks.append(Chunk(current, file_name, page.number, current_section, len(chunks)))
    return chunks
