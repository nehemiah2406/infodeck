"""Answer evaluation on the fictional engineering documents in data/eval_docs.

Usage:  .venv\\Scripts\\python scripts\\eval.py [--parser local|gemini] [--no-llm] [--top-k N]

Loads data/eval_docs into a temporary index, asks every question in
eval/questions.csv and prints how many answers contain the expected value
exactly as written (same digits, separators and units). Exact-value answers
need no LLM; the other questions use the LLM from .env (LLM_PROVIDER), or
sources mode with --no-llm (then the shown passages are checked instead).

Parsed pages are cached in storage/parsed, so a second run with --parser gemini
does not call Gemini again for the same files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401  (makes src/ importable)

from infodeck.config import get_settings, setup_logging
from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from infodeck.engine import InfoDeckEngine
from infodeck.llm import LLMStatus, resolve_llm
from infodeck.store import VectorStore

EVAL_DOCS = ROOT / "data" / "eval_docs"
QUESTIONS = ROOT / "eval" / "questions.csv"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parser", choices=["local", "gemini"], help="override PARSER from .env")
    ap.add_argument("--no-llm", action="store_true", help="do not call the LLM (sources mode)")
    ap.add_argument("--top-k", type=int, help="override TOP_K from .env")
    ap.add_argument("--no-exact", action="store_true",
                    help="switch off exact-value answers by name (to measure retrieval + LLM alone)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # σ, μ, ² in a Windows console

    setup_logging()
    settings = get_settings()
    settings = replace(settings, parser=args.parser or settings.parser, top_k=args.top_k or settings.top_k,
                       fact_match_score=2.0 if args.no_exact else settings.fact_match_score)
    rows = list(csv.DictReader(open(QUESTIONS, encoding="utf-8")))
    status = LLMStatus(None, "sources", "Sources mode", "--no-llm") if args.no_llm else resolve_llm(settings)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        s = replace(settings, chroma_dir=Path(tmp))
        store = VectorStore(s.chroma_dir, SentenceTransformerEmbedder(s.embedding_model))
        reranker = CrossEncoderReranker(s.rerank_model) if s.rerank_model else None
        engine = InfoDeckEngine(store, s, reranker)
        for r in engine.ingest_folder(EVAL_DOCS):
            print(f"[{r.status}] {r.message}")
        print(f"\nParser: {s.parser} | LLM: {status.label} | top-k {s.top_k} | "
              f"FACT_MATCH_SCORE {s.fact_match_score:.2f}\n")

        correct, by_mode = 0, {}
        for i, row in enumerate(rows, 1):
            ans = engine.answer(row["question"], status)
            shown = ans.text if ans.mode != "sources" else "\n".join(p.text for p in ans.passages)
            ok = norm(row["expected_answer"]) in norm(shown)
            correct += ok
            hits, total = by_mode.get(ans.mode, (0, 0))
            by_mode[ans.mode] = (hits + ok, total + 1)
            answer = re.sub(r"\s+", " ", shown)[:110]
            flag = f"  [check: {', '.join(ans.unverified)}]" if ans.unverified else ""
            print(f"{i:2}. {'OK  ' if ok else 'MISS'} [{ans.mode}] {row['question']}\n"
                  f"     expected: {row['expected_answer']}  |  answer: {answer}{flag}")

    print(f"\nScore: {correct}/{len(rows)} answers contain the expected value "
          f"({100 * correct / len(rows):.0f}%)")
    print("By answer mode: " + ", ".join(f"{m} {h}/{t}" for m, (h, t) in sorted(by_mode.items())))


if __name__ == "__main__":
    main()
