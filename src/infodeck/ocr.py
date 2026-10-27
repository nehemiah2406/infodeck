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
import threading
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
    global _engine
    import numpy as np

    with _engine_lock:  # the OCR engine is not thread-safe; also loads its models only once
        if _engine is None:
            _engine = _load_engine()
        result = _engine(np.array(image.convert("RGB")))
    return _result_text(result)


def _result_text(result) -> str:
    if result.txts is None or result.boxes is None:
        return ""
    return join_ocr_lines(
        [(min(p[1] for p in box), max(p[1] for p in box), min(p[0] for p in box), max(p[0] for p in box), text)
         for box, text in zip(result.boxes, result.txts) if text.strip()]
    )


def join_ocr_lines(boxes: list[tuple[float, float, float, float, str]]) -> str:
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
# Worker processes
# --------------------------------------------------------------------------- #
def _init_worker(threads: int) -> None:
    global _engine
    _engine = _load_engine(threads)


def _ocr_png(png: bytes) -> str:
    """Runs in a worker process."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        return _result_text(_engine(np.array(im.convert("RGB"))))


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
