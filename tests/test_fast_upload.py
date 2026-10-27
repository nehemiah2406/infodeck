"""Faster uploads: parallel OCR, background reading one file at a time, questions while a big
PDF is still loading, stopping, and resuming. No network; fictional data only."""

from __future__ import annotations

import threading
import time

from infodeck import config
from infodeck.chunking import Chunk
from infodeck.engine import COMMIT_EVERY_PAGES
from infodeck.jobs import IngestQueue
from infodeck.loaders import HEADER_SAMPLE_PAGES, PdfPage, iter_pdf_pages
from infodeck.parsing import write_meta
from scripts.make_eval_docs import make_scanned_pdf
from tests.conftest import SOURCES_STATUS

PAGES = 20
# Pages become searchable in batches: first after the header sample, then every few pages.
SEARCHABLE_AFTER = [HEADER_SAMPLE_PAGES + k * COMMIT_EVERY_PAGES for k in range(PAGES)]
PAUSE = SEARCHABLE_AFTER[2]  # pause the reader right after the third batch became searchable
PARTS = ("shaft hub key bolt flange gasket bearing seal coupling rotor stator valve pump nozzle liner "
         "piston crank spring damper casing").split()


def page_text(n: int) -> str:
    # Every page uses different words: lines that differ only in numbers count as repeated headers.
    part = PARTS[n - 1]
    return f"{n}. {part.title()} data\n(1) {part.title()} clearance, {n}.5 mm\nThe {part} topic{n} is explained here."


def wait_until(condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


class FakePdfReader:
    """Stands in for iter_pdf_pages: yields fictional pages and can pause after a given page."""

    def __init__(self, pause_after: int | None = None, fail_at: int | None = None):
        self.pause_after, self.fail_at = pause_after, fail_at
        self.paused = threading.Event()
        self.resume = threading.Event()
        self.known: dict[int, str] | None = None

    def __call__(self, path, name, ocr_dpi, ocr_workers, known=None):
        self.known = dict(known or {})
        for n in range(1, PAGES + 1):
            if n == self.fail_at:
                raise RuntimeError("the laptop went to sleep")
            yield PdfPage(n, PAGES, self.known.get(n, page_text(n)), ocr=n not in self.known)
            if n == self.pause_after:
                self.paused.set()
                self.resume.wait(20)


def test_ocr_settings_have_safe_defaults_and_can_be_changed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("OCR_WORKERS", raising=False)
    monkeypatch.delenv("OCR_DPI", raising=False)
    s = config.get_settings()
    assert (s.ocr_workers, s.ocr_dpi) == (6, 200)
    env.write_text("OCR_WORKERS=3\nOCR_DPI=300\n")
    s = config.get_settings()
    assert (s.ocr_workers, s.ocr_dpi) == (3, 300)


def test_parallel_ocr_gives_the_same_pages_in_the_same_order(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_scanned_pdf(pdf, [["(1) Push-up distance, 10.6 mm"], ["(2) Grip stress, 111 N/mm²"],
                           ["(3) Contact length, 210 mm"]], bad_text_layer=False)
    one = [(p.number, p.text, p.ocr) for p in iter_pdf_pages(pdf, "scan.pdf", ocr_workers=1)]
    two = [(p.number, p.text, p.ocr) for p in iter_pdf_pages(pdf, "scan.pdf", ocr_workers=2)]  # spawned workers
    assert [n for n, _, _ in two] == [1, 2, 3]
    assert two == one and "10.6 mm" in two[0][1]


def test_queue_reads_files_one_at_a_time_in_upload_order(tmp_path):
    running, most, order = [0], [0], []
    lock = threading.Lock()

    def run(job):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
            order.append(job.doc_id)
        time.sleep(0.05)
        with lock:
            running[0] -= 1

    q = IngestQueue(run)
    for d in ("big1", "big2", "small"):
        q.submit(d, f"{d}.pdf", tmp_path / d)
    assert q.wait(10)
    assert most[0] == 1 and order == ["big1", "big2", "small"]


def test_same_file_is_never_queued_twice_while_it_is_read(engine, monkeypatch):
    reader = FakePdfReader(pause_after=3)
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", reader)
    data = b"%PDF fictional calc sheet"
    first = engine.queue_upload("calc.pdf", data)
    assert first.status == "queued"
    reader.paused.wait(10)
    again = engine.queue_upload("calc.pdf", data)  # a rerun, a refresh or a second upload
    assert again.status == "loading" and len(engine.jobs.active()) == 1
    reader.resume.set()
    assert engine.jobs.wait(20)
    assert engine.jobs.get(first.doc_id).result.status == "added"
    assert engine.queue_upload("calc.pdf", data).status == "duplicate"


def test_questions_can_be_asked_about_pages_already_read(engine, monkeypatch):
    reader = FakePdfReader(pause_after=PAUSE)
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", reader)
    r = engine.queue_upload("big_scan.pdf", b"%PDF fictional 20 pages")
    assert reader.paused.wait(10)
    job = engine.jobs.get(r.doc_id)
    assert job.status == "running" and (job.done, job.total) == (PAUSE, PAGES)

    ans = engine.answer("key topic3", SOURCES_STATUS)
    assert ans.passages and ans.passages[0].page == 3
    assert f"big_scan.pdf is still loading (page {PAUSE} of 20)" in ans.loading
    exact = engine.answer("What is the valve clearance?", SOURCES_STATUS)
    assert exact.mode == "exact" and exact.text == "12.5 mm"
    assert not any(p.page > PAUSE for p in engine.store.search("spring topic18", 10))  # not read yet

    reader.resume.set()
    assert engine.jobs.wait(20)
    assert job.result.status == "added" and job.result.pages == PAGES
    assert engine.store.search("spring topic18", 1)[0].page == 18
    assert engine.loading_note() == ""


def test_removing_a_document_while_it_loads_stops_the_job(engine, monkeypatch):
    reader = FakePdfReader(pause_after=SEARCHABLE_AFTER[0])
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", reader)
    r = engine.queue_upload("big_scan.pdf", b"%PDF fictional")
    assert reader.paused.wait(10)
    assert engine.store.has_document(r.doc_id)  # first pages are already searchable
    engine.remove_document(r.doc_id)
    reader.resume.set()
    assert engine.jobs.wait(20)
    assert engine.jobs.get(r.doc_id).status == "cancelled"
    assert not engine.store.has_document(r.doc_id)
    assert not list(engine.settings.upload_dir.glob("*big_scan.pdf"))


def test_an_interrupted_upload_continues_from_the_pages_already_read(engine, monkeypatch):
    broken = FakePdfReader(fail_at=SEARCHABLE_AFTER[1] + 2)
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", broken)
    data = b"%PDF fictional"
    r = engine.queue_upload("big_scan.pdf", data)
    assert engine.jobs.wait(20)
    assert engine.jobs.get(r.doc_id).result.status == "error"
    assert not engine.store.has_document(r.doc_id)

    working = FakePdfReader()
    monkeypatch.setattr("infodeck.engine.iter_pdf_pages", working)
    assert engine.queue_upload("big_scan.pdf", data).status == "queued"
    assert engine.jobs.wait(20)
    # pages that were already searchable before the failure come from the cache, not from OCR again
    assert sorted(working.known) == list(range(1, SEARCHABLE_AFTER[1] + 1))
    assert engine.jobs.get(r.doc_id).result.status == "added"


def test_leftovers_of_a_closed_app_are_read_again_not_reported_as_duplicate(engine):
    doc_id = "0123456789abcdef"
    engine.store.add_document(doc_id, [Chunk("Page 1 text about topic1.", "half.pdf", 1, "", 0)], 20, "x")
    folder = engine.settings.parsed_dir / doc_id
    folder.mkdir(parents=True)
    write_meta(folder, {"file_name": "half.pdf", "parser": "local", "complete": False})
    assert not engine.is_loaded(doc_id)
    assert not engine.store.has_document(doc_id)  # half-indexed chunks removed


def test_small_files_are_also_read_in_the_background_queue(engine):
    r = engine.queue_upload("notes.txt", b"The dial gauge resolution is 0.01 mm.")
    assert r.status == "queued" and engine.jobs.wait(20)
    assert engine.jobs.get(r.doc_id).result.status == "added"
    assert engine.store.search("dial gauge", 1)[0].file_name == "notes.txt"
