"""Background reading of uploaded files: one file at a time, in upload order.

The queue belongs to the engine, which Streamlit creates once per server process,
so every browser tab and every rerun sees the same jobs. A job is identified by
the file's content hash (doc_id): uploading the same file again, rerunning the
script or refreshing the browser while it is being read never starts a second job.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised inside a running job when its document was removed."""


@dataclass
class IngestJob:
    doc_id: str
    file_name: str
    path: Path
    status: str = "queued"  # queued | running | done | cancelled
    done: int = 0  # pages read so far
    total: int = 0  # pages in the document (0 = not known yet)
    note: str = ""
    result: object = None  # the IngestResult, when finished
    cancel_requested: bool = False

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def update(self, done: int, total: int, note: str = "") -> None:
        self.done, self.total, self.note = done, total, note

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelled()


class IngestQueue:
    def __init__(self, run: Callable[[IngestJob], object]):
        self._run = run  # reads one file; returns its IngestResult (never raises)
        self._jobs: dict[str, IngestJob] = {}
        self._queue: queue.Queue[IngestJob] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.finished_count = 0  # lets the UI notice that a job has finished

    def submit(self, doc_id: str, file_name: str, path: Path) -> tuple[IngestJob, bool]:
        """Queue a file. Returns (job, True), or (the existing job, False) if it is already queued/running."""
        with self._lock:
            job = self._jobs.get(doc_id)
            if job and job.active:
                return job, False
            job = IngestJob(doc_id, file_name, Path(path))
            self._jobs.pop(doc_id, None)
            self._jobs[doc_id] = job  # newest last
            self._queue.put(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, name="infodeck-ingest", daemon=True)
                self._thread.start()
            return job, True

    def get(self, doc_id: str) -> IngestJob | None:
        return self._jobs.get(doc_id)

    def active(self) -> list[IngestJob]:
        """Jobs waiting or being read, in upload order."""
        with self._lock:
            return [j for j in self._jobs.values() if j.active]

    def cancel(self, doc_id: str) -> IngestJob | None:
        """Stop a queued or running job. A running job stops before its next page."""
        with self._lock:
            job = self._jobs.get(doc_id)
            if job is None or not job.active:
                return None
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
            return job

    def wait(self, timeout: float = 600) -> bool:
        """Wait until no job is waiting or running (scripts and tests). False on timeout."""
        deadline = time.monotonic() + timeout
        while self.active():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                with self._lock:
                    if job.cancel_requested:  # removed while waiting
                        job.status = "cancelled"
                        continue
                    job.status = "running"
                job.result = self._run(job)
                job.status = "cancelled" if getattr(job.result, "status", "") == "cancelled" else "done"
            except Exception:  # _run handles errors itself; this is only a safety net
                log.exception("Background reading of %s failed", job.file_name)
                job.status = "done"
            finally:
                with self._lock:
                    self.finished_count += 1
                self._queue.task_done()
