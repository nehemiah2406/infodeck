"""Read documents into pages of plain text.

Supported: .pdf (page by page, tables kept as readable rows), .docx and .txt.
Word and text files have no real pages, so they are returned as page 0
("no page number") unless a .txt file contains form-feed page breaks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MIN_CHARS_PER_PAGE = 20  # fewer characters than this = "no text" (probably a scanned image)


@dataclass
class Page:
    number: int  # 1-based page number; 0 means "no page numbers" (docx/txt)
    text: str


@dataclass
class LoadedDocument:
    file_name: str
    pages: list[Page]
    page_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return any(p.text.strip() for p in self.pages)


class UnsupportedFileError(ValueError):
    pass


def load_document(path: Path, display_name: str | None = None) -> LoadedDocument:
    path = Path(path)
    name = display_name or path.name
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path, name)
    if ext == ".docx":
        return _load_docx(path, name)
    if ext == ".txt":
        return _load_txt(path, name)
    raise UnsupportedFileError(f"{name}: file type '{ext}' is not supported (use PDF, DOCX or TXT).")


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip()


def table_to_text(rows: list[list]) -> str:
    """Turn a table into one line per row: 'Header: value; Header: value'."""
    rows = [[_clean_cell(c) for c in row] for row in rows if row and any(row)]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if not body:
        return " | ".join(header)
    lines = []
    for row in body:
        parts = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
        lines.append("; ".join(parts))
    return "\n".join(lines)


def _pdf_page_text(page) -> str:
    """Extract text in reading order, replacing each table with readable rows."""
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    if not tables:
        return page.extract_text() or ""
    parts, top = [], 0.0
    for t in tables:
        x0, t_top, x1, t_bottom = t.bbox
        if t_top > top:
            above = page.within_bbox((0, top, page.width, t_top)).extract_text() or ""
            parts.append(above)
        parts.append(table_to_text(t.extract()))
        top = max(top, t_bottom)
    if top < page.height:
        parts.append(page.within_bbox((0, top, page.width, page.height)).extract_text() or "")
    return "\n".join(p for p in parts if p.strip())


def _strip_repeated_lines(pages: list[Page]) -> None:
    """Remove header/footer lines that repeat on most pages (e.g. 'Company | Policy', 'Page 3')."""
    if len(pages) < 3:
        return

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts = Counter()
    for p in pages:
        counts.update({norm(l) for l in p.text.splitlines() if l.strip()})
    repeated = {line for line, n in counts.items() if n >= max(3, 0.6 * len(pages))}
    for p in pages:
        p.text = "\n".join(l for l in p.text.splitlines() if norm(l) not in repeated)


def _load_pdf(path: Path, name: str) -> LoadedDocument:
    import pdfplumber

    pages: list[Page] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = _pdf_page_text(page)
                except Exception:  # fall back to plain text if table detection fails
                    log.warning("Table extraction failed on %s page %d; using plain text", name, i)
                    text = page.extract_text() or ""
                # Glyphs without a text mapping (usually bullet symbols) come out as "(cid:N)".
                text = re.sub(r"\(cid:\d+\)", "•", text)
                pages.append(Page(i, text))
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as a PDF (is it damaged or password-protected?)") from exc

    _strip_repeated_lines(pages)
    doc = LoadedDocument(name, pages, page_count=len(pages))
    empty = [p.number for p in pages if len(p.text.strip()) < MIN_CHARS_PER_PAGE]
    if pages and len(empty) == len(pages):
        doc.warnings.append(
            f"{name} has no selectable text. It is probably a scanned image, so it cannot be "
            "searched. Please upload a text-based PDF (or run OCR on it first)."
        )
    elif empty:
        doc.warnings.append(
            f"{name}: page(s) {', '.join(map(str, empty))} have no selectable text "
            "(probably scanned images) and were skipped."
        )
    doc.pages = [p for p in pages if len(p.text.strip()) >= MIN_CHARS_PER_PAGE]
    return doc


# --------------------------------------------------------------------------- #
# DOCX / TXT
# --------------------------------------------------------------------------- #
def _load_docx(path: Path, name: str) -> LoadedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as a Word document.") from exc

    blocks = []
    for child in d.element.body.iterchildren():  # paragraphs and tables in document order
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, d).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            rows = [[c.text for c in r.cells] for r in Table(child, d).rows]
            blocks.append(table_to_text(rows))
    text = "\n".join(b for b in blocks if b)
    doc = LoadedDocument(name, [Page(0, text)] if text.strip() else [], page_count=1)
    if not doc.has_text:
        doc.warnings.append(f"{name} contains no text.")
    return doc


def _load_txt(path: Path, name: str) -> LoadedDocument:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    if "\f" in text:  # form feeds mark page breaks
        parts = text.split("\f")
        pages = [Page(i, t) for i, t in enumerate(parts, start=1) if t.strip()]
        count = len(parts)
    else:
        pages = [Page(0, text)] if text.strip() else []
        count = 1
    doc = LoadedDocument(name, pages, page_count=count)
    if not doc.has_text:
        doc.warnings.append(f"{name} contains no text.")
    return doc
