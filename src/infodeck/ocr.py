"""Local OCR (rapidocr + onnxruntime), in this process or in parallel worker processes.

OCR is ~95% of the time it takes to upload a scanned PDF, so pages are OCR'd by
a pool of worker processes (OCR_WORKERS). Each worker loads its own OCR models
once and gets a fair share of the CPU threads, so workers don't fight over cores.

Windows notes: workers are started with the "spawn" method (the only one Windows
has). A spawned worker imports this module fresh, and Python also re-imports the
main script, so every script that uploads files must keep its code under
`if __name__ == "__main__":`. The pool is shared by the whole server process and
reused for every upload, so models are loaded only once per worker.
"""

from __future__ import annotations

import atexit
import io
import logging
import multiprocessing
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor

log = logging.getLogger(__name__)

_engine = None  # the OCR engine of this process (main process, or one per worker)
_engine_lock = threading.Lock()


def _load_engine(threads: int = -1):
    from rapidocr import RapidOCR

    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    params = {"EngineConfig.onnxruntime.intra_op_num_threads": threads} if threads > 0 else None
    return RapidOCR(params=params)


def ocr_image(image) -> str:
    """Read the text in a PIL image (in this process), one output line per printed line."""
    with _engine_lock:  # the OCR engine is not thread-safe; also loads its models only once
        return "\n".join(text for text, _ in _read_lines(_get_engine(), image))


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _load_engine()
    return _engine


def _result_boxes(result) -> list[tuple[float, float, float, float, str]]:
    if result.txts is None or result.boxes is None:
        return []
    return [(min(p[1] for p in box), max(p[1] for p in box), min(p[0] for p in box), max(p[0] for p in box), text)
            for box, text in zip(result.boxes, result.txts) if text.strip()]


def ocr_line_boxes(image) -> list[tuple[str, tuple[float, float, float, float]]]:
    """OCR a PIL image (in this process) into (line text, (left, top, right, bottom)) in pixels.

    Used for page previews; the lines are read exactly like the text read at upload.
    """
    with _engine_lock:
        return _read_lines(_get_engine(), image)


def _read_lines(engine, image) -> list[tuple[str, tuple[float, float, float, float]]]:
    """OCR an image into lines; lines with broken numbers are re-read (see recheck_numbers)."""
    import numpy as np

    image = image.convert("RGB")
    # The engine remembers use_det/use_cls/use_rec from its last call, so always pass them.
    result = engine(np.array(image), use_det=True, use_cls=True, use_rec=True)
    out = []
    for line in group_ocr_lines(_result_boxes(result)):
        box = (min(b[2] for b in line), min(b[0] for b in line), max(b[3] for b in line), max(b[1] for b in line))
        text = _line_text(line)
        if broken_numbers(text):
            text = recheck_numbers(text, lambda scale, line=line: _reread_line(engine, image, line, scale))
        out.append((text, box))
    return out


# --------------------------------------------------------------------------- #
# Broken numbers ("10).6", "(0).534", "5,940) kN")
# --------------------------------------------------------------------------- #
# OCR sometimes reads a speck or a thin line next to a digit as a bracket. Such a number
# is never guessed or "cleaned up": each text box of the line is cut out of the page image
# and read again on its own, as it is and enlarged 2x (a single box is read more reliably
# than a whole page). Only if both reads show exactly the same numbers as the line without
# those brackets is the line changed.
_NUMBER_RUN = re.compile(r"[\d.,()]*\d[\d.,()]*")
_INNER_BRACKET = re.compile(r"\d[()][.,]?\d")  # a bracket between digits: "10).6", "(0).534"
_LIST_NUMBER = re.compile(r"\(?\d{1,2}\)")  # "1)" or "(1)" at the start of a line
RECHECK_SCALES = (1.0, 2.0)  # every re-read must agree


def _unmatched_brackets(line: str) -> set[int]:
    """Positions of brackets in the line that have no partner."""
    opened, unmatched = [], set()
    for i, ch in enumerate(line):
        if ch == "(":
            opened.append(i)
        elif ch == ")":
            if opened:
                opened.pop()
            else:
                unmatched.add(i)
    return unmatched | set(opened)


def _broken_runs(line: str) -> list[re.Match]:
    unmatched = _unmatched_brackets(line)
    found = []
    for m in _NUMBER_RUN.finditer(line):
        token = m.group()
        if "(" not in token and ")" not in token:
            continue
        if m.start() == 0 and _LIST_NUMBER.fullmatch(token):
            continue
        if _INNER_BRACKET.search(token) or any(m.start() <= i < m.end() for i in unmatched):
            found.append(m)
    return found


def broken_numbers(line: str) -> list[str]:
    """Numbers in an OCR line with a stray bracket inside or next to them, e.g. '10).6'."""
    return [m.group() for m in _broken_runs(line)]


def _numbers(text: str) -> list[str]:
    return [m.group().strip(".,") for m in _NUMBER_RUN.finditer(text)]


def recheck_numbers(line: str, reread: Callable[[float], str]) -> str:
    """Fix the broken numbers of a line only if a second read of the image confirms them.

    reread(scale) reads the line again from the image, enlarged by scale. The brackets of the
    broken numbers are dropped ('10).6' -> '10.6') only if EVERY re-read (one per scale in
    RECHECK_SCALES) has exactly the same numbers, in the same order, as the line without those
    brackets. Nothing else is ever changed: if a re-read disagrees (e.g. '5.940' for '5,940)'),
    the line is kept exactly as it was read (and gives no exact-value card).
    """
    runs = _broken_runs(line)
    if not runs:
        return line
    fixed = line
    for m in reversed(runs):
        fixed = fixed[:m.start()] + m.group().replace("(", "").replace(")", "") + fixed[m.end():]
    expected = _numbers(fixed)
    try:
        confirmed = all(_numbers(reread(scale)) == expected for scale in RECHECK_SCALES)
    except Exception:
        log.warning("Re-reading an OCR line failed", exc_info=True)
        confirmed = False
    if not confirmed:
        log.warning("OCR numbers kept as read (a second read did not confirm them): %s",
                    ", ".join(m.group() for m in runs))
        return line
    log.info("OCR numbers confirmed by a second read: %s",
             ", ".join(f"{m.group()} -> {m.group().replace('(', '').replace(')', '')}" for m in runs))
    return fixed


def _reread_line(engine, image, line: list[tuple], scale: float) -> str:
    """Read one OCR line of a page image again, box by box (recognition only, no detection).

    Each box is cut out with a small margin and enlarged by `scale`. Recognition alone takes
    ~0.03 s per box; detection would first blow a thin strip up to a huge image (~3 s).
    """
    import numpy as np
    from PIL import Image

    boxes = []
    for top, bottom, left, right, _ in line:
        pad = 0.3 * (bottom - top)
        crop = image.crop((max(0, int(left - pad)), max(0, int(top - pad)),
                           min(image.width, int(right + pad) + 1), min(image.height, int(bottom + pad) + 1)))
        if scale != 1.0:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        result = engine(np.array(crop), use_det=False, use_cls=False, use_rec=True)
        boxes.append((top, bottom, left, right, " ".join(result.txts or ())))
    return _line_text(boxes)


def join_ocr_lines(boxes: list[tuple[float, float, float, float, str]]) -> str:
    """Group OCR boxes (top, bottom, left, right, text) into lines, left to right."""
    return "\n".join(_line_text(line) for line in group_ocr_lines(boxes))


def _line_text(line: list[tuple]) -> str:
    text = line[0][4]
    for prev, box in zip(line, line[1:]):
        gap = box[2] - prev[3]
        text += ("   " if gap > 2 * (box[1] - box[0]) else " ") + box[4]
    return text.strip()


def group_ocr_lines(boxes: list[tuple[float, float, float, float, str]]) -> list[list[tuple]]:
    """OCR boxes grouped into lines (top to bottom), each line sorted left to right."""
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
    for line in lines:
        line.sort(key=lambda b: b[2])
    return lines


# --------------------------------------------------------------------------- #
# Worker processes
# --------------------------------------------------------------------------- #
def _init_worker(threads: int) -> None:
    global _engine
    _engine = _load_engine(threads)


def _ocr_png(png: bytes) -> str:
    """Runs in a worker process."""
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        return "\n".join(text for text, _ in _read_lines(_engine, im))


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", compress_level=1)  # fast; lossless like the original
    return buf.getvalue()


class OcrPool:
    """OCR for many pages at once. With workers <= 1 everything runs in this process."""

    def __init__(self, workers: int):
        self.workers = max(1, workers)
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def submit(self, image) -> Future:
        if self.workers <= 1:
            future: Future = Future()
            future.set_result(ocr_image(image))
            return future
        png = png_bytes(image)
        with self._lock:
            if self._pool is None:
                threads = max(1, (os.cpu_count() or 2) // self.workers)
                log.info("Starting %d OCR worker processes (%d threads each)", self.workers, threads)
                self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=_init_worker, initargs=(threads,))
            return self._pool.submit(_ocr_png, png)

    def reset(self) -> None:
        """Drop a broken pool (e.g. a worker crashed); the next submit starts a new one."""
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


_pools: dict[int, OcrPool] = {}
_pools_lock = threading.Lock()


def get_pool(workers: int) -> OcrPool:
    """The shared pool for this worker count (created once, reused for every upload)."""
    with _pools_lock:
        if workers not in _pools:
            _pools[workers] = OcrPool(workers)
        return _pools[workers]


@atexit.register
def _shutdown_pools() -> None:
    for pool in _pools.values():
        pool.reset()
