"""The real app in a real browser (headless Chrome): empty screen, click send, upload with +,
refresh the page and restart the server. Documents read once stay ready; an unchanged file is
never read again (checked by its hash), a changed one is.

The app runs from a copy in a temp folder without .env: AI is off (no API is called) and the real
storage folder is never touched. Skipped when Playwright or Chrome is not installed.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from infodeck.config import ROOT

sync_api = pytest.importorskip("playwright.sync_api")

QUESTION = "What is the valve clearance?"
DOC = "Pump maintenance notes\nThe valve clearance is 12.5 mm.\nThe pump speed is 1450 rpm.\n"
CHANGED = "Pump maintenance notes\nThe valve clearance is 14.0 mm.\nThe pump speed is 1450 rpm.\n"


class App:
    """streamlit run app.py on a copy of the app; the server output goes to a log file per start."""

    def __init__(self, folder: Path):
        self.folder, self.starts, self.proc = folder, 0, None
        folder.mkdir()
        for name in ("app.py", "app.css"):
            shutil.copy(ROOT / name, folder / name)
        shutil.copytree(ROOT / "src", folder / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / ".streamlit", folder / ".streamlit", ignore=shutil.ignore_patterns("secrets.toml"))
        (folder / "no_samples").mkdir()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://localhost:{self.port}"

    def start(self) -> None:
        self.starts += 1
        env = {**os.environ, "LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "",
               "GROQ_API_KEY": "", "RERANK_MODEL": "none", "SAMPLE_DIR": "no_samples", "OCR_WORKERS": "1",
               "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
        self.log = open(self.folder / f"server_{self.starts}.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port",
                                      str(self.port), "--server.address", "127.0.0.1", "--server.headless", "true"],
                                     cwd=self.folder, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"{self.url}/_stcore/health", timeout=2)
                return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("the app did not start")

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait(30)
            self.log.close()
            self.proc = None

    def reads(self, file_name: str) -> int:
        """How often the file was read (indexed), over all server starts."""
        return sum(p.read_text(encoding="utf-8").count(f"Added {file_name}:") +
                   p.read_text(encoding="utf-8").count(f"Updated {file_name}:")
                   for p in self.folder.glob("server_*.log"))


@pytest.fixture
def app(tmp_path):
    a = App(tmp_path / "install")
    a.start()
    yield a
    a.stop()


@pytest.fixture
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:  # no Chrome on this computer
            pytest.skip(f"Chrome is not available: {e}")
        yield b
        b.close()


def open_app(browser, app):
    page = browser.new_page(viewport={"width": 1300, "height": 900})
    page.goto(app.url)
    page.wait_for_selector("textarea", timeout=180_000)  # the engine has started
    page.wait_for_timeout(1500)
    return page


def click_send(page) -> None:
    page.click("[data-testid='stChatInputSubmitButton']")


def ask(page, question: str) -> str:
    """Type the question and click the send arrow; returns the text of the new answer card."""
    before = page.locator("[class*='st-key-idcard_']").count()
    page.fill("textarea", question)
    click_send(page)
    page.wait_for_function(f"document.querySelectorAll(\"[class*='st-key-idcard_']\").length > {before}",
                           timeout=60_000)
    return page.locator("[class*='st-key-idcard_']").nth(before).inner_text()


def upload_with_plus(page, path: Path, expected_note: str) -> None:
    page.locator("[data-testid='stChatInput'] input[type=file]").set_input_files(str(path))
    page.wait_for_timeout(500)
    click_send(page)
    page.get_by_text(expected_note).wait_for(timeout=60_000)


def sidebar_documents(page) -> list[str]:
    """The lines of the documents list in the sidebar (name, plus "Reading..." while being read)."""
    rows = page.locator("[data-testid='stSidebar'] .id-doc")
    return [" ".join(rows.nth(i).inner_text().split()) for i in range(rows.count())]


def test_empty_screen_send_upload_refresh_and_restart(app, browser, tmp_path):
    doc = tmp_path / "valve.txt"
    doc.write_text(DOC, encoding="utf-8")
    page = open_app(browser, app)

    # 1. The empty screen: only the InfoDeck title and the input box
    main = page.locator("[data-testid='stMainBlockContainer']").inner_text()
    assert main.split() == ["InfoDeck"]
    assert page.locator("[data-testid='stMainBlockContainer'] .stButton").count() == 0
    assert sidebar_documents(page) == []  # no documents yet (no samples in this test)

    # 3. Upload with + : a clear note when it is ready, and the file in the sidebar list (name only)
    upload_with_plus(page, doc, "valve.txt is ready.")
    assert app.reads("valve.txt") == 1
    assert sidebar_documents(page) == ["valve.txt"]

    # 2. Click the send arrow: the question is answered from the uploaded file
    assert "valve.txt" in ask(page, QUESTION)

    # 4a. Refresh the page: the document is still ready, nothing is read again
    page.reload()
    page.wait_for_selector("textarea", timeout=60_000)
    page.wait_for_timeout(1000)
    assert sidebar_documents(page) == ["valve.txt"]  # documents loaded before the refresh are listed
    assert "valve.txt" in ask(page, QUESTION)
    upload_with_plus(page, doc, "valve.txt is already loaded and unchanged, so it was not read again.")
    assert app.reads("valve.txt") == 1
    assert sidebar_documents(page) == ["valve.txt"]  # still one line

    # 4b. Restart the app: still ready from storage, the unchanged file is not read again
    app.stop()
    app.start()
    page = open_app(browser, app)
    assert sidebar_documents(page) == ["valve.txt"]  # listed from storage after a restart
    assert "valve.txt" in ask(page, QUESTION)
    upload_with_plus(page, doc, "valve.txt is already loaded and unchanged, so it was not read again.")
    assert app.reads("valve.txt") == 1

    # A changed file (same name, new content) is read again
    doc.write_text(CHANGED, encoding="utf-8")
    upload_with_plus(page, doc, "valve.txt has new content: it was read again and is ready.")
    assert app.reads("valve.txt") == 2
    assert sidebar_documents(page) == ["valve.txt"]  # replaced, not listed twice
    assert "14.0" in ask(page, QUESTION)


def test_sidebar_shows_reading_while_a_scan_is_read_then_just_the_name(app, browser, tmp_path):
    from scripts.make_eval_docs import make_scanned_pdf

    scan = tmp_path / "scan.pdf"  # a scan is read with OCR: slow enough to see "Reading..."
    topics = ["Pump speed is 1450 rpm", "Valve clearance is 12.5 mm", "Bolt torque is 540 N m",
              "Shaft diameter is 180 mm", "Oil pressure is 4.2 bar", "Coolant flow is 35 litres",
              "Bearing gap is 0.15 mm", "Filter change every 500 hours"]
    # every page different: lines repeated on all pages would be removed as headers/footers
    make_scanned_pdf(scan, [[topic, f"Inspection record {chr(65 + n)} for unit {n * 7 + 3}"]
                            for n, topic in enumerate(topics)])
    page = open_app(browser, app)
    page.locator("[data-testid='stSidebar'] input[type=file]").set_input_files(str(scan))
    page.wait_for_function("[...document.querySelectorAll(\"[data-testid='stSidebar'] .id-doc\")]"
                           ".some(e => e.innerText.includes('scan.pdf') && e.innerText.includes('Reading...'))",
                           timeout=60_000)
    page.get_by_text("scan.pdf is ready.").wait_for(timeout=180_000)
    page.wait_for_function("[...document.querySelectorAll(\"[data-testid='stSidebar'] .id-doc\")]"
                           ".some(e => e.innerText.trim() === 'scan.pdf')", timeout=30_000)
    assert sidebar_documents(page) == ["scan.pdf"]
