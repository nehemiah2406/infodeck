"""A fresh install (no storage folder) must read an uploaded scan with parallel OCR.

Bug: Streamlit makes app.py the main module, and on Windows every new OCR worker process
first re-runs the main module. Each worker started a whole second InfoDeck (AI models,
database, sample documents) before reading a page: a 1-page scan took ~2 minutes and the
"waiting to be read" note stayed. app.py now only starts the app when it is not run as
"__mp_main__" (the name a worker uses).
"""

from __future__ import annotations

import shutil
import sys
import types
from dataclasses import replace

from infodeck import ocr
from infodeck.config import ROOT, Settings
from infodeck.engine import InfoDeckEngine
from infodeck.jobs import IngestJob
from infodeck.store import VectorStore
from scripts.make_eval_docs import make_scanned_pdf
from tests.conftest import FakeEmbedder


def copy_of_the_app(folder):
    """app.py + src, without .env or storage: a worker that wrongly starts this copy can only
    create folder/storage, never touch the real one."""
    folder.mkdir()
    for name in ("app.py", "app.css"):
        shutil.copy(ROOT / name, folder / name)
    shutil.copytree(ROOT / "src", folder / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return folder


def test_fresh_install_reads_a_scan_with_ocr_workers_without_starting_the_app_again(tmp_path, monkeypatch):
    app = copy_of_the_app(tmp_path / "install")
    storage = tmp_path / "storage"  # does not exist yet: a fresh install
    settings = replace(Settings(), chroma_dir=storage / "chroma", parsed_dir=storage / "parsed",
                       upload_dir=tmp_path / "uploads", sample_dir=ROOT / "data" / "sample_docs",
                       ocr_workers=2, parser="local", rerank_model="")
    engine = InfoDeckEngine(VectorStore(settings.chroma_dir, FakeEmbedder()), settings)
    assert len(engine.load_samples_once()) == 3  # first start: the sample documents are loaded

    # Like Streamlit: app.py is the main module while the upload is read (and OCR workers start).
    streamlit_main = types.ModuleType("__main__")
    streamlit_main.__file__, streamlit_main.__spec__ = str(app / "app.py"), None
    monkeypatch.setitem(sys.modules, "__main__", streamlit_main)
    monkeypatch.setattr(ocr, "_pools", {})  # new worker processes, started under that main module
    scan = tmp_path / "letter.pdf"
    make_scanned_pdf(scan, [["Experience letter", "Valve clearance is 12.5 mm"]])
    try:
        result = engine.queue_upload("letter.pdf", scan.read_bytes())
        assert result.status == "queued"
        assert engine.jobs.wait(300)
    finally:
        for pool in ocr._pools.values():
            pool.reset()

    job = engine.jobs.get(result.doc_id)
    assert job.result.status == "added" and job.result.pages == 1
    assert engine.loading_note() == ""  # nothing is "waiting to be read" any more
    assert engine.store.search("valve clearance", 1)[0].file_name == "letter.pdf"  # searchable
    assert not (app / "storage").exists()  # the OCR worker did not start a second InfoDeck


def test_a_started_job_is_being_read_not_waiting(engine, tmp_path):
    job = IngestJob("d1", "letter.pdf", tmp_path / "letter.pdf", status="running")  # page count not known yet
    engine.jobs._jobs["d1"] = job
    assert engine.loading_note() == "letter.pdf is being read. Pages not read yet are not searched."
    job.update(0, 4)
    assert engine.loading_note() == "letter.pdf is still loading (page 0 of 4). Pages not read yet are not searched."
    job.status = "queued"
    assert engine.loading_note() == "letter.pdf is waiting to be read. Pages not read yet are not searched."
    job.status = "done"
    assert engine.loading_note() == ""


def test_app_starts_only_when_it_is_not_an_ocr_worker():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert source.rstrip().endswith('if __name__ != "__mp_main__":\n    main()')
    top_level = [line for line in source.splitlines() if line.startswith("st.")]
    assert top_level == []  # no Streamlit call runs at import time (a worker imports this file)
