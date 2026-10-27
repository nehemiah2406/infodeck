"""Page previews for answer cards: a crop of the real source page with the matched line boxed.

A preview is made only when an answer card asks for one (never during upload) and is
cached next to the parsed pages: storage/parsed/<doc_id>/preview_p<N>_<hash>.png (crop)
and ..._full.png (whole page, for "Open page"). The line is found with pdfplumber's text
positions, or, on scanned pages and photos, with one OCR pass of that page. If the line
cannot be found, the crop is shown without a box.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .loaders import IMAGE_EXTENSIONS, MIN_CHARS_PER_PAGE, is_scanned_page
from .ocr import ocr_line_boxes

PREVIEW_DPI = 200  # same as OCR_DPI, so OCR finds the line as well as it did at upload
CROP_WIDTH = 720  # pixels of the saved crop (shown about 300 px wide)
FULL_WIDTH = 1100
MIN_LINE_MATCH = 0.5  # share of the target's words (numbers count 3x) a page line must contain
BOX_COLOR = (52, 211, 153)
PREVIEW_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS
TOKEN_RE = re.compile(r"\w+(?:[.,]\w+)*")

Box = tuple[float, float, float, float]  # left, top, right, bottom in image pixels


@dataclass
class PagePreview:
    crop: bytes  # PNG
    full: bytes  # PNG
    boxed: bool  # True if the matched line was found and boxed


def can_preview(source_path: str | Path, page: int) -> bool:
    path = Path(source_path)
    if path.suffix.lower() == ".pdf":
        return page >= 1 and path.exists()
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.exists()


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def find_line(target: str, lines: list[tuple[str, Box]]) -> Box | None:
    """Box of the page line that best contains the target text, or None if none is close enough.

    Words with digits count three times, so a table row with the right value beats a header
    row that only repeats the column names.
    """
    wanted = {t: 3 if any(c.isdigit() for c in t) else 1 for t in _tokens(target)}
    total = sum(wanted.values())
    if not total:
        return None
    best, best_score = None, 0.0
    for text, box in lines:
        found = set(_tokens(text))
        score = sum(w for t, w in wanted.items() if t in found) / total
        if score > best_score:
            best, best_score = box, score
    return best if best_score >= MIN_LINE_MATCH else None


def _render(path: Path, page: int):
    """(page image, its text lines with boxes) at PREVIEW_DPI."""
    from PIL import Image

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(path) as im:
            image = im.convert("RGB")
        return image, ocr_line_boxes(image)
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        p = pdf.pages[page - 1]
        image = p.to_image(resolution=PREVIEW_DPI).original.convert("RGB")
        text = p.extract_text() or ""
        if is_scanned_page(p) or len(text.strip()) < MIN_CHARS_PER_PAGE:  # read with OCR at upload too
            return image, ocr_line_boxes(image)
        s = PREVIEW_DPI / 72
        lines = [(ln["text"], (ln["x0"] * s, ln["top"] * s, ln["x1"] * s, ln["bottom"] * s))
                 for ln in p.extract_text_lines()]
    return image, lines


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _resized(image, width: int):
    if image.width <= width:
        return image
    return image.resize((width, round(image.height * width / image.width)))


def _draw(image, box: Box | None):
    """(crop, full page) with the box drawn: soft green fill plus a rounded green outline."""
    from PIL import Image, ImageDraw

    w, h = image.size
    page = image.convert("RGBA")
    if box:
        pad = max(6, round(w * 0.006))
        rect = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(rect, radius=pad, fill=BOX_COLOR + (46,), outline=BOX_COLOR + (255,),
                               width=max(3, round(w * 0.0028)))
        page = Image.alpha_composite(page, overlay)
    page = page.convert("RGB")

    cw = round(w * 0.8)
    ch = min(h, round(cw * 0.62))
    if box:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    else:  # no box: the top of the page, where the title usually is
        cx, cy = w / 2, ch / 2
    left = int(min(max(cx - cw / 2, 0), w - cw))
    top = int(min(max(cy - ch / 2, 0), h - ch))
    crop = page.crop((left, top, left + cw, top + ch))
    return _resized(crop, CROP_WIDTH), _resized(page, FULL_WIDTH)


def page_preview(source_path: str | Path, page: int, target: str, cache_dir: Path,
                 cached_only: bool = False) -> PagePreview | None:
    """Preview of the page with the target line boxed (cached), or None for files without pages.

    cached_only=True never renders: it returns None unless the preview is already cached.
    """
    path = Path(source_path)
    if not can_preview(path, page):
        return None
    key = hashlib.sha1(f"{page}|{target}".encode()).hexdigest()[:12]
    crop_file = cache_dir / f"preview_p{page}_{key}.png"
    full_file = cache_dir / f"preview_p{page}_{key}_full.png"
    box_file = cache_dir / f"preview_p{page}_{key}.boxed"
    if crop_file.exists() and full_file.exists():
        return PagePreview(crop_file.read_bytes(), full_file.read_bytes(), box_file.exists())
    if cached_only:
        return None

    image, lines = _render(path, page)
    box = find_line(target, lines)
    crop, full = _draw(image, box)
    result = PagePreview(_png(crop), _png(full), box is not None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if box:
        box_file.touch()
    for file, data in ((full_file, result.full), (crop_file, result.crop)):  # crop last: it marks "done"
        tmp = file.with_name(f".{os.getpid()}_{file.name}")
        tmp.write_bytes(data)
        os.replace(tmp, file)
    return result
