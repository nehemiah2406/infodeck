"""Exact-value cards must only show the item the question is about. Small fictional snippets only;
the full evaluation over real documents is tests/exact_eval.py."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from infodeck.engine import InfoDeckEngine
from infodeck.facts import Fact, extract_facts, match_quality, question_subject
from infodeck.loaders import Page
from infodeck.store import VectorStore
from tests.conftest import FakeEmbedder

# A calculation-sheet item whose value is on the NEXT line (fictional values, OCR typo kept)
SHEET = """9. Our proposal of push-up distance, (Pd), mm
(1) Push-up distance, Pd = 10.6 mm
(2) Grip stress coresponding to Pd, N/mm²
σra = Pd / Pr x σr = 111 N/mm²
10. Max. equivalent von Mises stress of tiller boss, (σe), N/mm²
σe = { σra / (1 − f² ) } x ( 3 + f4 )^0.5 = 272.4 N/mm²
where, f = (dm / δGH) = 0.534
∴ σe = 272.4 N/mm² ≤ σyt = 345 N/mm²
"""


@pytest.fixture
def sheet_engine(settings, tmp_path) -> InfoDeckEngine:
    # A low similarity gate lets many facts in as candidates: the safety check must do the work.
    engine = InfoDeckEngine(VectorStore(settings.chroma_dir, FakeEmbedder()), replace(settings, fact_match_score=0.2))
    (tmp_path / "sheet.txt").write_text(SHEET, encoding="utf-8")
    engine.ingest_file(tmp_path / "sheet.txt")
    return engine


def facts_by_item() -> dict[str, Fact]:
    return {f.item + (f.symbol if not f.name else ""): f for f in extract_facts([Page(21, SHEET)], "sheet.pdf")}


# ---- reading the facts ---------------------------------------------------------- #
def test_value_on_the_next_line_belongs_to_its_item():
    grip = facts_by_item()["9.(2)"]
    assert (grip.name, grip.value, grip.unit, grip.symbol) == ("Grip stress coresponding to Pd", "111", "N/mm²", "σra")
    assert grip.line == "(2) Grip stress coresponding to Pd, N/mm²\nσra = Pd / Pr x σr = 111 N/mm²"  # word for word
    von_mises = facts_by_item()["10."]
    assert (von_mises.name, von_mises.value) == ("Max. equivalent von Mises stress of tiller boss", "272.4")


def test_next_line_is_only_taken_when_it_clearly_belongs_to_the_item():
    text = "(3) Section modulus, (Z), cm³\nI = b x h3 / 12 = 9,800 cm⁴\n(4) Material factor, (Mf)\nk = 1.2"
    facts = extract_facts([Page(1, text)], "x.pdf")
    # other unit / other symbol: the lines stay facts of their own symbols, never of the headers
    assert [(f.name, f.symbol, f.value) for f in facts] == [("", "I", "9,800"), ("", "k", "1.2")]


def test_comparison_helper_and_converted_values_are_not_given_to_the_wrong_symbol():
    facts = facts_by_item()
    assert "345" not in {f.value for f in facts.values()}  # "≤ σyt = 345" is not σe
    assert facts["10.f"].value == "0.534" and facts["10.f"].name == ""  # "where, f" has no name "where"
    assert extract_facts([Page(1, "(1) Yield point, (σyt)   345 N/mm² (= 35 kgf/mm²)")], "x.pdf") == []
    [one] = extract_facts([Page(1, "t = { ( k2 + 1 ) / ( k2 − 1 ) } + 1   ① = 2.8")], "x.pdf")
    assert one.symbol == "①"  # two equations side by side: 2.8 is ①, not t
    variable = extract_facts([Page(1, "Q : Shear force = (2 x M x10^6) / dm, (N)   Q= 7.728.592")], "x.pdf")
    assert [(f.kind, f.name, f.value) for f in variable] == [("value", "Shear force", "7.728.592")]  # not a Q&A


# ---- the safety check (every exact card) --------------------------------------------- #
def test_main_subject_must_be_in_the_items_own_name():
    facts = facts_by_item()
    question = "What is the grip stress corresponding to Pd?"
    assert question_subject(question) == ["grip", "stress"]
    assert match_quality(question, facts["9.(1)"]) is None  # Pd is only a reference
    assert match_quality(question, facts["9.(2)"]) == "tight"  # despite the OCR typo "coresponding"
    assert match_quality("What is the proposed push-up distance Pd?", facts["9.(1)"]) == "tight"
    assert match_quality("What is the push-up force based on Pd?", facts["9.(1)"]) is None
    # a helper line never answers its item's title
    assert match_quality("What is the max. equivalent von Mises stress of the tiller boss?", facts["10.f"]) is None


def test_grip_stress_question_never_shows_the_push_up_distance(sheet_engine):
    ans = sheet_engine.exact_answer("What is the grip stress corresponding to Pd?")
    assert ans is None or (ans.fact.item == "9.(2)" and ans.text == "111 N/mm²")
    assert ans is not None  # here it is found: the value on the next line is now read
    for q in ["What is the grip stress for Pd?", "What is the stress corresponding to Pd?"]:
        ans = sheet_engine.exact_answer(q)
        assert ans is None or ans.text == "111 N/mm²", q


def test_proposed_push_up_distance_is_still_10_6_mm(sheet_engine):
    ans = sheet_engine.exact_answer("What is the proposed push-up distance Pd?")
    assert ans.text == "10.6 mm" and ans.fact.item == "9.(1)"
    assert sheet_engine.exact_answer("What is Pd?").text == "10.6 mm"


def test_no_card_when_nothing_matches_safely(sheet_engine):
    for q in ["What is the push-up force based on Pd?", "What is the hub diameter corresponding to Pd?",
              "What is σyt?"]:
        assert sheet_engine.exact_answer(q) is None, q


def test_two_items_with_the_same_name_and_different_values_give_no_card(sheet_engine, tmp_path):
    (tmp_path / "other.txt").write_text("8. Push-up distance, (Pr), mm\nPr = Id x taper = 10.5 mm\n",
                                        encoding="utf-8")
    sheet_engine.ingest_file(tmp_path / "other.txt")
    assert sheet_engine.exact_answer("What is the push-up distance?") is None  # 10.5 or 10.6?
    assert sheet_engine.exact_answer("What is the push-up distance Pr?").text == "10.5 mm"


# ---- the upload self-test -------------------------------------------------------------- #
def test_upload_self_test_checks_every_fact_and_logs_the_result(settings, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="infodeck.engine")
    engine = InfoDeckEngine(VectorStore(settings.chroma_dir, FakeEmbedder()), settings)
    (tmp_path / "sheet.txt").write_text(SHEET, encoding="utf-8")
    engine.ingest_file(tmp_path / "sheet.txt")
    [doc_id] = engine.store.unchecked_documents() or [d.doc_id for d in engine.store.list_documents()]
    facts = [f for _, f in engine.store.document_facts(doc_id)]
    assert facts and all(f.checked for f in facts) and not any(f.unsafe for f in facts)
    assert "Exact-value self-test for sheet.txt: 0 of" in caplog.text
    assert engine.store.unchecked_documents() == []


def test_self_test_marks_confused_facts_unsafe_and_they_are_never_shown(sheet_engine, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="infodeck.engine")
    [doc_id] = [d.doc_id for d in sheet_engine.store.list_documents()]
    pd = next(f for _, f in sheet_engine.store.document_facts(doc_id) if f.symbol == "Pd")
    grip = next(f for _, f in sheet_engine.store.document_facts(doc_id) if f.symbol == "σra")
    real_match = sheet_engine.match_fact

    def confused(question):  # pretend the matcher answers the grip stress name with Pd's value
        return (pd, 1.0) if "Grip stress" in question else real_match(question)

    monkeypatch.setattr(sheet_engine, "match_fact", confused)
    assert sheet_engine.self_test_facts(doc_id, "sheet.txt") == 2
    assert "2 of" in caplog.text and "marked unsafe" in caplog.text
    monkeypatch.undo()
    unsafe = {f.symbol for _, f in sheet_engine.store.document_facts(doc_id) if f.unsafe}
    assert unsafe == {"Pd", "σra"}  # both: the asked fact and the one that answered for it
    assert sheet_engine.exact_answer("What is the proposed push-up distance Pd?") is None  # never shown
    assert sheet_engine.exact_answer(f"What is the {grip.name}?") is None


def test_documents_indexed_before_the_self_test_are_checked_at_start(sheet_engine):
    [doc_id] = [d.doc_id for d in sheet_engine.store.list_documents()]
    ids = [i for i, _ in sheet_engine.store.document_facts(doc_id)]
    sheet_engine.store.update_facts(ids, checked=False)  # like facts stored by an older version
    assert sheet_engine.store.unchecked_documents() == [doc_id]
    sheet_engine.self_test_unchecked()
    assert sheet_engine.store.unchecked_documents() == []
