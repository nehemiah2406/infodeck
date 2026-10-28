"""Exact-value evaluation: does the exact-value card ever show the WRONG item?

Usage:
    .venv\\Scripts\\python tests\\exact_eval.py draft [--doc NAME]   list numbered items with a value
    .venv\\Scripts\\python tests\\exact_eval.py run                  ask every question, print the table
    .venv\\Scripts\\python tests\\exact_eval.py why "question" ...   every candidate and the safety check

Questions live in data/eval/exact_questions.json (git-ignored: it holds document text). Each entry:
    {"doc": file name, "page": n, "item": "9.(2)", "question": "...", "value": "114", "unit": "N/mm²",
     "lines": ["parts of the item's line(s), word for word"], "pages": [other pages with the same item],
     "also": [the same item with the same value in another document], "note": "optional"}
"value" is copied word for word from the page; "" means no exact card is expected (any card is wrong).

"run" indexes every document in storage/parsed (the parse cache, no OCR and no Gemini), the KRT sample
policies and three made-up calculation sheets (tests/exact_sheets.py) into a temporary folder with
the local embedding model, asks every question through InfoDeckEngine.exact_answer and counts:
    correct     the card shows the expected value (and unit) from the expected item and page
    wrong item  a card is shown with anything else  -> must be 0
    no card     no exact card (the AI / passages answer is used instead) -> acceptable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "data" / "eval" / "exact_questions.json"
ITEM_START = re.compile(r"^\s*(\d{1,3}\.\s|\d{1,2}\.\d{1,2}\s|\(\d{1,2}\)|\d{1,2}\)\s)")
HAS_NUMBER = re.compile(r"\d")
VALUE_AFTER_EQ = re.compile(r"=\s*[-+−]?\d[\d,.]*\s*\S*\s*$")
TABLE_ROW = re.compile(r"^[^;:]{1,40}:\s*[^;]+(;\s*[^;:]{1,60}:\s*[^;]+)+$")


def cached_documents() -> list[tuple[str, list[tuple[int, str]]]]:
    """(file name, [(page, text)]) for every complete parse in storage/parsed + the sample policies."""
    from infodeck.config import get_settings
    from infodeck.loaders import load_document

    docs = []
    for meta_file in sorted(get_settings().parsed_dir.glob("*/meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if not meta.get("complete"):
            continue
        pages = [(n, (meta_file.parent / f"page_{n}.md").read_text(encoding="utf-8")) for n in meta["pages"]]
        docs.append((meta["file_name"], pages))
    for pdf in sorted((ROOT / "data" / "sample_docs").glob("*.pdf")):
        doc = load_document(pdf)
        docs.append((pdf.name, [(p.number, p.text) for p in doc.pages]))
    return docs


def draft(only: str | None) -> None:
    """Print numbered item lines that hold a number (plus the next line when the value is there),
    and table rows. Found with simple rules of its own, NOT with facts.py, so the eval stays honest."""
    for name, pages in cached_documents():
        if only and only.lower() not in name.lower():
            continue
        print(f"\n######## {name}")
        for number, text in pages:
            lines = [line.strip() for line in text.splitlines()]
            for i, line in enumerate(lines):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if TABLE_ROW.match(line) and HAS_NUMBER.search(line):
                    print(f"p{number} ROW   {line}")
                elif ITEM_START.match(line):
                    if HAS_NUMBER.search(ITEM_START.sub("", line, count=1)):
                        print(f"p{number} ITEM  {line}")
                    elif VALUE_AFTER_EQ.search(nxt) and not ITEM_START.match(nxt):
                        print(f"p{number} ITEM  {line}\n        NEXT  {nxt}")


def load_questions() -> list[dict]:
    from tests.exact_sheets import QUESTIONS as SHEET_QUESTIONS

    real = json.loads(QUESTIONS.read_text(encoding="utf-8")) if QUESTIONS.exists() else []
    return real + SHEET_QUESTIONS


def build_engine(folder: Path):
    """A temporary index of every document (cached pages, sample policies, made-up sheets)."""
    import hashlib

    from infodeck.config import get_settings
    from infodeck.embeddings import SentenceTransformerEmbedder
    from infodeck.engine import InfoDeckEngine
    from infodeck.loaders import Page
    from infodeck.store import VectorStore
    from tests.exact_sheets import SHEETS

    s = replace(get_settings(), chroma_dir=folder / "chroma", parsed_dir=folder / "parsed",
                upload_dir=folder / "uploads")
    engine = InfoDeckEngine(VectorStore(s.chroma_dir, SentenceTransformerEmbedder(s.embedding_model)), s)
    docs = cached_documents() + [(name, [(0, text)]) for name, text in SHEETS.items()]
    for name, pages in docs:
        doc_id = hashlib.sha1(name.encode()).hexdigest()[:16]
        engine.index_pages(doc_id, name, [Page(n, t) for n, t in pages], max(n for n, _ in pages) or 1, "eval")
    return engine


def _key(question: str) -> str:
    return " ".join(question.lower().split())


def is_expected(card, entry: dict) -> bool:
    """The card shows this entry's value, from this entry's document, page and item line (or one of
    the entry's "also" answers: the same item, with the same value, in another document)."""
    if any(is_expected(card, {**entry, "also": [], "pages": [a["page"]], **a}) for a in entry.get("also", [])):
        return True
    f = card.fact
    return (bool(entry["value"]) and f.value == entry["value"] and f.file_name == entry["doc"]
            and f.page in entry.get("pages", [entry["page"]])
            and (not entry.get("unit") or f.unit == entry["unit"])
            and (not entry.get("lines") or any(part in f.line for part in entry["lines"])))


def run() -> int:
    questions = load_questions()
    # The same question may be asked about two documents (e.g. "grip stress corresponding to Pd" in two
    # calculation sheets): a card from either one, showing that document's own value, is correct.
    by_question: dict[str, list[dict]] = {}
    for q in questions:
        by_question.setdefault(_key(q["question"]), []).append(q)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = build_engine(Path(tmp))
        results = []
        for q in questions:
            card = engine.exact_answer(q["question"])
            if card is None:
                outcome, shown = "no card", ""
            else:
                ok = any(is_expected(card, e) for e in by_question[_key(q["question"])])
                outcome = "correct" if ok else "wrong item"
                f = card.fact
                shown = f"{f.value} {f.unit}".strip() + f" | {f.file_name} p.{f.page} item {f.item} | {f.line}"
            results.append({**q, "outcome": outcome, "shown": shown})

    docs = list(dict.fromkeys(r["doc"] for r in results))
    print(f"\n{'Document':42} {'questions':>9} {'correct':>8} {'wrong item':>11} {'no card':>8}")
    for doc in docs + ["TOTAL"]:
        rows = [r for r in results if doc in ("TOTAL", r["doc"])]
        count = {o: sum(r["outcome"] == o for r in rows) for o in ("correct", "wrong item", "no card")}
        print(f"{doc[:42]:42} {len(rows):>9} {count['correct']:>8} {count['wrong item']:>11} {count['no card']:>8}")
    wrong = [r for r in results if r["outcome"] == "wrong item"]
    if wrong:
        print("\nWRONG ITEM:")
        for r in wrong:
            expected = f"{r['value']} (item {r['item']}, p.{r['page']})" if r["value"] else "no card"
            print(f"  {r['question']}\n      expected: {expected}\n      shown:    {r['shown']}")
    print("\nNO CARD (the AI / passages answer is used):")
    for r in results:
        if r["outcome"] == "no card":
            print(f"  [{'trap, ok' if not r['value'] else r['item']}] {r['question']}")
    out = QUESTIONS.parent / "exact_results.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetails saved to {out.relative_to(ROOT)}")
    return 1 if wrong else 0


def why(questions: list[str]) -> None:
    """Every exact-value candidate for each question, and what the safety check says about it."""
    from infodeck.engine import FACT_CANDIDATES
    from infodeck.facts import match_quality, question_subject, symbol_in_question

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = build_engine(Path(tmp))
        for q in questions:
            print(f"\n{q}\n  subject: {question_subject(q)}")
            tokens = [t for t in re.findall(r"[^\s?!,;:()]+", q) if len(t) >= 2]
            found = [(f, 1.0) for f in engine.store.facts_with_symbol(tokens) if symbol_in_question(q, f.symbol)]
            found += engine.store.search_facts(q, FACT_CANDIDATES)
            for f, score in found:
                print(f"  {score:.3f} {match_quality(q, f)!s:5} unsafe={f.unsafe!s:5} | {f.value} {f.unit} | "
                      f"{f.file_name[:20]} p.{f.page} {f.item} | {f.embed_text[:60]}")
            card = engine.exact_answer(q)
            print(f"  CARD: {card.text if card else None}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["draft", "run", "why"])
    ap.add_argument("questions", nargs="*", help="why: the questions to explain")
    ap.add_argument("--doc", help="draft: only documents whose name contains this")
    args = ap.parse_args()
    if args.command == "draft":
        draft(args.doc)
    elif args.command == "why":
        why(args.questions)
    else:
        import logging

        logging.basicConfig(level=logging.WARNING)
        sys.exit(run())
