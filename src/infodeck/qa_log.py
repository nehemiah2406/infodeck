"""Q&A log: one JSON line per answered question in logs/qa_log.jsonl (QA_LOG_FILE in .env).

Each line has the time, the question, the answer shown, the file and page, and the card type.
The log holds document text, so logs/ is git-ignored. API keys never reach it: any key that
appears in the text is replaced with '***' before the line is written.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from .config import _api_keys, get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()  # several browser sessions can answer at the same time


def _mask(value, secrets: list[str]):
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, "***")
    return value


def log_answer(path: Path, question: str, view, seconds: float = 0.0, secrets: list[str] | None = None) -> dict:
    """Append one line for an answered question (view: ui.CardView). Never raises."""
    if secrets is None:
        secrets = _api_keys(get_settings())
    secrets = [s for s in secrets if s and len(s) >= 8]
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": question,
        "kind": view.kind,
        "answer": view.answer_text or view.message,
        "file": view.file_name,
        "page": view.page or None,
        "quote": view.quote_text,
        "seconds": round(seconds, 1),
    }
    record = {k: _mask(v, secrets) for k, v in record.items()}
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("Could not write the Q&A log %s", path, exc_info=True)
    return record
