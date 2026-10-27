"""Turn a file into pages of text with the configured parser (PARSER=local|gemini).

Every parsed page is saved as storage/parsed/<doc_id>/page_N.md (page_0.md for
files without page numbers) so you can check exactly what was read, plus a
meta.json naming the file and parser. These files are also the parse cache:
the doc_id is the file's content hash, so re-uploading the same file reuses
them instead of calling Gemini (or OCR) again. Gemini pages are saved one by
one, so an interrupted 200-page upload continues where it stopped.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .config import Settings
from .gemini_parser import GeminiPageReader, ParseError
from .loaders import IMAGE_EXTENSIONS, MIN_CHARS_PER_PAGE, LoadedDocument, Page, Progress, load_document

log = logging.getLogger(__name__)

PARSE_DPI = 200
GEMINI_WORKERS = 4  # pages read in parallel
GEMINI_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS  # file types read as page images in gemini mode
META_FILE = "meta.json"


def parser_name(settings: Settings, path: Path) -> str:
    if settings.parser == "gemini" and path.suffix.lower() in GEMINI_EXTENSIONS:
        return f"gemini:{settings.gemini_model}"
    return "local"


def parse_document(path: Path, name: str, doc_id: str, settings: Settings,
                   progress: Progress | None = None, gemini_client=None) -> LoadedDocument:
    """Parse a file into pages (from the cache when possible). Raises ParseError/ValueError."""
    path = Path(path)
    if settings.parser not in ("local", "gemini"):
        raise ParseError(f"Unknown PARSER '{settings.parser}' in .env (use local or gemini).")
    parser = parser_name(settings, path)
    out = settings.parsed_dir / doc_id
    meta = read_meta(out)
    if meta.get("parser") != parser:  # parsed before with another parser/model: start again
        shutil.rmtree(out, ignore_errors=True)
        meta = {}
    if meta.get("complete"):
        log.info("Using cached parse of %s (%s)", name, parser)
        return _load_cached(out, name, meta)
    out.mkdir(parents=True, exist_ok=True)
    if parser == "local":
        doc = load_document(path, name, progress, settings.ocr_dpi, settings.ocr_workers)
        for p in doc.pages:
            (out / f"page_{p.number}.md").write_text(p.text, encoding="utf-8")
    else:
        doc = _parse_with_gemini(path, name, out, settings, progress, gemini_client)
    write_meta(out, {"file_name": name, "parser": parser, "complete": True, "page_count": doc.page_count,
                      "pages": [p.number for p in doc.pages], "warnings": doc.warnings})
    return doc


def read_meta(folder: Path) -> dict:
    try:
        return json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_partial(settings: Settings, doc_id: str) -> bool:
    """True if reading this document started but did not finish (e.g. the app was closed)."""
    meta = read_meta(settings.parsed_dir / doc_id)
    return bool(meta) and not meta.get("complete")


def write_meta(folder: Path, meta: dict) -> None:
    (folder / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cached(folder: Path, name: str, meta: dict) -> LoadedDocument:
    pages = [Page(n, (folder / f"page_{n}.md").read_text(encoding="utf-8")) for n in meta["pages"]]
    return LoadedDocument(name, pages, meta["page_count"], list(meta.get("warnings", [])))


# --------------------------------------------------------------------------- #
# Gemini: render every page at 200 dpi and send only the image
# --------------------------------------------------------------------------- #
def _png(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class _PageImages:
    """Page count and 200-dpi PNG renders of a PDF or image file (pages are 1-based)."""

    def __init__(self, path: Path, name: str):
        self.path, self.name = path, name
        self._pdf = None
        try:
            if path.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                self._pdf = pdfium.PdfDocument(str(path))
                self.count = len(self._pdf)
            else:
                from PIL import Image

                with Image.open(path) as im:
                    im.verify()
                self.count = 1
        except Exception as exc:
            kind = "a PDF (is it damaged or password-protected?)" if path.suffix.lower() == ".pdf" else "an image"
            raise ValueError(f"{name}: could not be opened as {kind}") from exc

    def render(self, number: int) -> bytes:
        if self._pdf is None:
            from PIL import Image

            with Image.open(self.path) as im:
                return _png(im)
        page = self._pdf[number - 1]
        try:
            return _png(page.render(scale=PARSE_DPI / 72).to_pil())
        finally:
            page.close()

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()


def _parse_with_gemini(path: Path, name: str, out: Path, settings: Settings,
                       progress: Progress | None, client=None) -> LoadedDocument:
    if not settings.gemini_api_key.strip() and client is None:
        raise ParseError("PARSER=gemini needs a GEMINI_API_KEY in .env.")
    reader = GeminiPageReader(settings.gemini_api_key, settings.gemini_model, client=client)
    images = _PageImages(path, name)
    total = images.count
    is_image = path.suffix.lower() in IMAGE_EXTENSIONS

    def page_label(n: int) -> int:  # images have no page numbers (page 0)
        return 0 if is_image else n

    write_meta(out, {"file_name": name, "parser": parser_name(settings, path), "complete": False,
                      "page_count": total})
    texts: dict[int, str] = {}
    note = {"text": ""}
    lock = threading.Lock()

    def on_wait(delay: float, reason: str) -> None:  # called from worker threads
        with lock:
            note["text"] = f"{reason}, waiting {delay:.0f} s and retrying..."

    def report() -> None:
        if progress:
            with lock:
                text = note["text"]
            progress(len(texts), total, text)

    def save(number: int, text: str) -> None:
        texts[number] = text
        (out / f"page_{page_label(number)}.md").write_text(text, encoding="utf-8")
        with lock:
            note["text"] = ""

    def collect(pending: dict, until_below: int) -> None:
        while len(pending) > until_below:
            done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                save(pending.pop(future), future.result())  # re-raises ParseError
            report()

    pool = ThreadPoolExecutor(GEMINI_WORKERS)
    pending: dict = {}
    try:
        for number in range(1, total + 1):
            cached = out / f"page_{page_label(number)}.md"
            if cached.exists():
                texts[number] = cached.read_text(encoding="utf-8")
                report()
                continue
            pending[pool.submit(reader.read, images.render(number), on_wait)] = number
            collect(pending, until_below=2 * GEMINI_WORKERS - 1)  # bounded queue: big PDFs don't fill memory
        collect(pending, until_below=0)
    except ParseError as e:
        raise ParseError(f"{name}: stopped at page {len(texts) + 1} of {total}. {e}") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        images.close()

    pages = [Page(page_label(n), texts[n]) for n in sorted(texts)
             if len(texts[n].strip()) >= MIN_CHARS_PER_PAGE]
    doc = LoadedDocument(name, pages, page_count=total)
    empty = [n for n in sorted(texts) if len(texts[n].strip()) < MIN_CHARS_PER_PAGE]
    if empty and len(empty) < total:
        doc.warnings.append(f"{name}: page(s) {', '.join(map(str, empty))} are blank and were skipped.")
    elif empty:
        doc.warnings.append(f"{name}: Gemini found no text on any page.")
    return doc
