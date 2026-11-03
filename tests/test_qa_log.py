"""The Q&A log: one JSON line per question with the shown answer, file, page and time, never an API key."""

import json
from datetime import datetime

from infodeck.engine import Answer
from infodeck.facts import Fact
from infodeck.prompts import NOT_FOUND
from infodeck.qa_log import log_answer
from infodeck.ui import card_view

GEMINI_KEY = "AIza-THIS-IS-A-FAKE-GEMINI-KEY-0123456789"
GROQ_KEY = "gsk_THIS-IS-A-FAKE-GROQ-KEY-0123456789"
FACT = Fact("value", "Push-up distance", "10.6", unit="mm", line="(1) Push-up distance, 10.6 mm",
            file_name="calc.pdf", page=21, doc_id="d1")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_line_is_written_per_question_with_answer_file_page_and_time(tmp_path):
    path = tmp_path / "logs" / "qa_log.jsonl"  # the folder is created when needed
    view = card_view(Answer("10.6 mm", "exact", fact=FACT), "What is the push-up distance?", 0.55)
    log_answer(path, "What is the push-up distance?", view, 0.42, secrets=[])
    log_answer(path, "Who is the CEO?", card_view(Answer(NOT_FOUND, "not_found"), "q", 0.55), secrets=[])
    first, second = read(path)
    assert (first["question"], first["kind"], first["answer"], first["file"], first["page"]) == \
        ("What is the push-up distance?", "exact", "10.6 mm", "calc.pdf", 21)
    assert first["quote"] == "(1) Push-up distance, 10.6 mm" and first["seconds"] == 0.4
    assert datetime.fromisoformat(first["time"]).tzinfo is not None
    assert (second["kind"], second["answer"], second["file"], second["page"]) == ("not_found", NOT_FOUND, "", None)


def test_api_keys_never_reach_the_log(tmp_path):
    path = tmp_path / "qa_log.jsonl"
    ans = Answer(f"The key is {GEMINI_KEY}", "groq")  # e.g. pasted into a document or a question
    log_answer(path, f"is {GROQ_KEY} valid?", card_view(ans, "q", 0.55), secrets=[GEMINI_KEY, GROQ_KEY, ""])
    text = path.read_text(encoding="utf-8")
    assert GEMINI_KEY not in text and GROQ_KEY not in text and text.count("***") == 2


def test_keys_from_the_settings_are_masked_by_default(tmp_path, monkeypatch):
    import infodeck.config as config

    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={GROQ_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    path = tmp_path / "qa_log.jsonl"
    log_answer(path, f"what about {GROQ_KEY}?", card_view(Answer(NOT_FOUND, "not_found"), "q", 0.55))
    assert GROQ_KEY not in path.read_text(encoding="utf-8")


def test_a_log_that_cannot_be_written_never_breaks_the_answer(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    record = log_answer(blocked / "qa_log.jsonl", "q", card_view(Answer(NOT_FOUND, "not_found"), "q", 0.55), secrets=[])
    assert record["question"] == "q"


def test_logs_folder_is_git_ignored():
    from infodeck.config import ROOT

    assert "logs/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
