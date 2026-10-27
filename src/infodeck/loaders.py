"""Read documents into pages of plain text (the "local" parser).

Supported: .pdf (page by page, tables kept as readable rows), .docx, .txt,
images (.png/.jpg), Excel (.xlsx/.csv) and PowerPoint (.pptx).
- Scanned PDF pages (a full-page image, or almost no text) are read with local
  OCR (rapidocr). A scanned page's own text layer is ignored, because the OCR
  that made it is often poor ("I I I" instead of 111).
- Excel: one sheet = one page, each row as "Header: value; Header: value".
- PowerPoint: one slide = one page.
Word, text, CSV and image files have no real pages, so they are returned as
page 0 ("no page number") unless a .txt file contains form-feed page breaks.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".csv", ".pptx"} | IMAGE_EXTENSIONS
MIN_CHARS_PER_PAGE = 20  # fewer characters than this = "no text" (probably a scanned image)
SCANNED_IMAGE_COVERAGE = 0.8  # a page mostly covered by one image is treated as a scan
OCR_DPI = 200

# progress(done, total, note): called after each page is read
Progress = Callable[[int, int, str], None]


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


def load_document(path: Path, display_name: str | None = None, progress: Progress | None = None) -> LoadedDocument:
    path = Path(path)
    name = display_name or path.name
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path, name, progress)
    if ext == ".docx":
        return _load_docx(path, name)
    if ext == ".txt":
        return _load_txt(path, name)
    if ext in IMAGE_EXTENSIONS:
        return _load_image(path, name)
    if ext == ".xlsx":
        return _load_xlsx(path, name)
    if ext == ".csv":
        return _load_csv(path, name)
    if ext == ".pptx":
        return _load_pptx(path, name)
    raise UnsupportedFileError(
        f"{name}: file type '{ext}' is not supported (use PDF, Word, Excel, PowerPoint, text or image files).")


# --------------------------------------------------------------------------- #
# OCR (rapidocr + onnxruntime, runs on this computer)
# --------------------------------------------------------------------------- #
_ocr_engine = None
_ocr_lock = threading.Lock()


def ocr_image(image) -> str:
    """Read the text in a PIL image, one output line per printed line."""
    global _ocr_engine
    import numpy as np

    with _ocr_lock:  # the OCR engine is not thread-safe; also loads its models only once
        if _ocr_engine is None:
            from rapidocr import RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            _ocr_engine = RapidOCR()
        result = _ocr_engine(np.array(image.convert("RGB")))
    if result.txts is None or result.boxes is None:
        return ""
    return _join_ocr_lines(
        [(min(p[1] for p in box), max(p[1] for p in box), min(p[0] for p in box), max(p[0] for p in box), text)
         for box, text in zip(result.boxes, result.txts) if text.strip()]
    )


def _join_ocr_lines(boxes: list[tuple[float, float, float, float, str]]) -> str:
    """Group OCR boxes (top, bottom, left, right, text) into lines, left to right."""
    lines: list[list[tuple]] = []
    for box in sorted(boxes, key=lambda b: (b[0] + b[1]) / 2):
        middle, height = (box[0] + box[1]) / 2, box[1] - box[0]
        if lines:
            last = lines[-1]
            last_middle = sum((b[0] + b[1]) / 2 for b in last) / len(last)
            if abs(middle - last_middle) < 0.5 * max(height, 1):
                last.append(box)
                continue
        lines.append([box])
    out = []
    for line in lines:
        line.sort(key=lambda b: b[2])
        text = line[0][4]
        for prev, box in zip(line, line[1:]):
            gap = box[2] - prev[3]
            text += ("   " if gap > 2 * (box[1] - box[0]) else " ") + box[4]
        out.append(text.strip())
    return "\n".join(out)


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


def is_scanned_page(page) -> bool:
    """True if one image covers most of the page (a scan, maybe with a hidden OCR text layer)."""
    area = float(page.width * page.height) or 1.0
    for im in page.images:
        w, h = min(im["x1"], page.width) - max(im["x0"], 0), min(im["bottom"], page.height) - max(im["top"], 0)
        if w > 0 and h > 0 and w * h / area >= SCANNED_IMAGE_COVERAGE:
            return True
    return False


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


def _load_pdf(path: Path, name: str, progress: Progress | None = None) -> LoadedDocument:
    import pdfplumber

    pages: list[Page] = []
    ocr_pages: list[int] = []
    try:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = _pdf_page_text(page)
                except Exception:  # fall back to plain text if table detection fails
                    log.warning("Table extraction failed on %s page %d; using plain text", name, i)
                    text = page.extract_text() or ""
                # Glyphs without a text mapping (usually bullet symbols) come out as "(cid:N)".
                text = re.sub(r"\(cid:\d+\)", "•", text)
                if is_scanned_page(page) or len(text.strip()) < MIN_CHARS_PER_PAGE:
                    ocr_text = ocr_image(page.to_image(resolution=OCR_DPI).original)
                    if len(ocr_text.strip()) >= MIN_CHARS_PER_PAGE or is_scanned_page(page):
                        text = ocr_text
                        ocr_pages.append(i)
                pages.append(Page(i, text))
                if progress:
                    progress(i, total, "")
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as a PDF (is it damaged or password-protected?)") from exc

    _strip_repeated_lines(pages)
    doc = LoadedDocument(name, pages, page_count=len(pages))
    empty = [p.number for p in pages if len(p.text.strip()) < MIN_CHARS_PER_PAGE]
    if pages and len(empty) == len(pages):
        doc.warnings.append(
            f"{name} has no selectable text and OCR found none. It is probably a blank or "
            "unreadable scanned image, so it cannot be searched."
        )
    elif empty:
        doc.warnings.append(
            f"{name}: page(s) {', '.join(map(str, empty))} have no selectable text "
            "(probably scanned images that OCR could not read) and were skipped."
        )
    read_by_ocr = [n for n in ocr_pages if n not in empty]
    if read_by_ocr:
        doc.warnings.append(
            f"{name}: page(s) {_ranges(read_by_ocr)} are scanned and were read with OCR. "
            "Check important values against the original."
        )
    doc.pages = [p for p in pages if len(p.text.strip()) >= MIN_CHARS_PER_PAGE]
    return doc


def _ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'"""
    out, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)


# --------------------------------------------------------------------------- #
# Images (photos, screenshots)
# --------------------------------------------------------------------------- #
def _load_image(path: Path, name: str) -> LoadedDocument:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()
            text = ocr_image(im)
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as an image.") from exc
    doc = LoadedDocument(name, [Page(0, text)] if len(text.strip()) >= MIN_CHARS_PER_PAGE else [], page_count=1)
    if doc.has_text:
        doc.warnings.append(f"{name} was read with OCR. Check important values against the original.")
    else:
        doc.warnings.append(f"{name}: no readable text was found in this image.")
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


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _load_txt(path: Path, name: str) -> LoadedDocument:
    text = _decode(path.read_bytes())
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


# --------------------------------------------------------------------------- #
# Excel / CSV
# --------------------------------------------------------------------------- #
def cell_text(value) -> str:
    """A cell value as text, without rounding (10.0 -> '10', 10.6 -> '10.6')."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat(" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _rows_text(rows: list[list]) -> str:
    rows = [[cell_text(v) for v in row] for row in rows]
    return table_to_text([r for r in rows if any(c.strip() for c in r)])


def _load_xlsx(path: Path, name: str) -> LoadedDocument:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as an Excel workbook.") from exc
    pages = []
    try:
        for i, ws in enumerate(wb.worksheets, start=1):
            body = _rows_text([list(r) for r in ws.iter_rows(values_only=True)])
            if body.strip():
                pages.append(Page(i, f"Sheet: {ws.title}\n{body}"))
        count = len(wb.worksheets)
    finally:
        wb.close()
    doc = LoadedDocument(name, pages, page_count=count)
    if not doc.has_text:
        doc.warnings.append(f"{name} contains no data.")
    return doc


def _load_csv(path: Path, name: str) -> LoadedDocument:
    text = _decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    body = _rows_text(list(csv.reader(io.StringIO(text), dialect)))
    doc = LoadedDocument(name, [Page(0, body)] if body.strip() else [], page_count=1)
    if not doc.has_text:
        doc.warnings.append(f"{name} contains no data.")
    return doc


# --------------------------------------------------------------------------- #
# PowerPoint
# --------------------------------------------------------------------------- #
def _shape_texts(shapes, ocr_pictures: list) -> list[str]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out = []
    for shape in sorted(shapes, key=lambda s: ((s.top or 0), (s.left or 0))):
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            out += _shape_texts(shape.shapes, ocr_pictures)
        elif getattr(shape, "has_table", False) and shape.has_table:
            out.append(table_to_text([[c.text for c in row.cells] for row in shape.table.rows]))
        elif getattr(shape, "has_text_frame", False) and shape.has_text_frame and shape.text_frame.text.strip():
            out.append(shape.text_frame.text.strip())
        elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            ocr_pictures.append(shape)
    return out


def _load_pptx(path: Path, name: str) -> LoadedDocument:
    from PIL import Image
    from pptx import Presentation

    try:
        prs = Presentation(str(path))
    except Exception as exc:
        raise ValueError(f"{name}: could not be opened as a PowerPoint file.") from exc
    pages = []
    for i, slide in enumerate(prs.slides, start=1):
        pictures: list = []
        parts = _shape_texts(slide.shapes, pictures)
        if len("".join(parts)) < MIN_CHARS_PER_PAGE:  # mostly pictures (e.g. a screenshot): OCR them
            for pic in pictures:
                try:
                    with Image.open(io.BytesIO(pic.image.blob)) as im:
                        parts.append(ocr_image(im))
                except Exception:
                    log.warning("Could not read a picture on %s slide %d", name, i)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
        text = "\n".join(p for p in parts if p.strip())
        if text.strip():
            pages.append(Page(i, text))
    doc = LoadedDocument(name, pages, page_count=len(prs.slides))
    if not doc.has_text:
        doc.warnings.append(f"{name} contains no text.")
    return doc
