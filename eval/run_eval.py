"""Evaluate InfoDeck retrieval on the fictional sample PDFs.

Usage:  .venv\\Scripts\\python eval\\run_eval.py
Writes: eval/results.md

Metrics (no LLM / API key needed):
- Retrieval hit rate @5: answerable questions whose expected file + page is in the top 5 passages.
- Hit rate @1: same, but only the first passage counts.
- "Not found" accuracy: questions NOT in the documents that are correctly answered
  "I could not find this in the documents." (best vector score below MIN_SCORE).
- False "not found": answerable questions wrongly rejected by the threshold.
A threshold sweep shows how MIN_SCORE trades these off.
"""

from __future__ import annotations

import csv
import re
import sys
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infodeck.config import get_settings  # noqa: E402
from infodeck.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder  # noqa: E402
from infodeck.engine import InfoDeckEngine  # noqa: E402
from infodeck.loaders import load_document  # noqa: E402
from infodeck.store import VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_gold(rows, sample_dir: Path) -> None:
    """Make sure each evidence phrase really is on the expected page (catches wrong answer keys)."""
    for r in rows:
        if r["type"] == "not_in_docs":
            continue
        doc = load_document(sample_dir / r["expected_file"])
        page = next((p for p in doc.pages if p.number == int(r["expected_page"])), None)
        if page is None or norm(r["evidence"]) not in norm(page.text):
            raise SystemExit(f"Gold data error in question {r['id']}: '{r['evidence']}' "
                             f"not found on {r['expected_file']} p.{r['expected_page']}")


def main() -> None:
    settings = get_settings()
    rows = list(csv.DictReader(open(EVAL_DIR / "retrieval_questions.csv", encoding="utf-8")))
    check_gold(rows, settings.sample_dir)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        s = replace(settings, chroma_dir=Path(tmp), min_score=0.0)  # gate applied below, per threshold
        store = VectorStore(s.chroma_dir, SentenceTransformerEmbedder(s.embedding_model))
        reranker = CrossEncoderReranker(s.rerank_model) if s.rerank_model else None
        engine = InfoDeckEngine(store, s, reranker)
        engine.ingest_folder(s.sample_dir)

        results = []
        for r in rows:
            best = store.search(r["question"], 1)[0].score  # what the "not found" rule looks at
            top5 = engine.retrieve(r["question"])  # re-ranked top 5
            rank = None
            if r["type"] != "not_in_docs":
                for i, p in enumerate(top5, 1):
                    if p.file_name == r["expected_file"] and p.page == int(r["expected_page"]):
                        rank = i
                        break
            results.append({**r, "best": best, "rank": rank, "top": top5[0].label if top5 else "-"})

    answerable = [x for x in results if x["type"] != "not_in_docs"]
    missing = [x for x in results if x["type"] == "not_in_docs"]

    def metrics(t: float) -> dict:
        found = lambda x: x["best"] >= t  # noqa: E731
        hit5 = sum(1 for x in answerable if found(x) and x["rank"] is not None)
        hit1 = sum(1 for x in answerable if found(x) and x["rank"] == 1)
        nf_ok = sum(1 for x in missing if not found(x))
        false_nf = sum(1 for x in answerable if not found(x))
        return dict(t=t, hit5=hit5, hit1=hit1, nf_ok=nf_ok, false_nf=false_nf)

    current = settings.min_score
    sweep = [metrics(t) for t in sorted(set(THRESHOLDS + [current]))]
    cur = metrics(current)
    n_a, n_m = len(answerable), len(missing)
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)"  # noqa: E731

    lines = [
        "# InfoDeck evaluation results",
        "",
        f"Run on {date.today().isoformat()} with the 3 fictional sample PDFs "
        f"({len(rows)} questions in `eval/retrieval_questions.csv`).  ",
        f"Embedding model `{settings.embedding_model}`, re-ranker `{settings.rerank_model or 'off'}`, "
        f"top-k {settings.top_k}, MIN_SCORE {current:.2f}.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit rate @5 (right file + page in top 5) | {pct(cur['hit5'], n_a)} |",
        f"| Hit rate @1 (right page ranked first) | {pct(cur['hit1'], n_a)} |",
        f"| \"Not found\" accuracy (questions not in the documents) | {pct(cur['nf_ok'], n_m)} |",
        f"| Answerable questions wrongly rejected | {cur['false_nf']}/{n_a} |",
        "",
        "## Per question",
        "",
        "| # | Type | Question | Expected | Best score | Rank of expected page | Top passage |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in results:
        expected = f"{x['expected_file']}, p.{x['expected_page']}" if x["expected_file"] else "not found"
        if x["type"] == "not_in_docs":
            outcome = "correctly not found" if x["best"] < current else "**passed the threshold**"
        elif x["best"] < current:
            outcome = "**wrongly rejected**"
        else:
            outcome = str(x["rank"]) if x["rank"] else "**not in top 5**"
        top = x["top"] if x["best"] >= current else "-"
        lines.append(f"| {x['id']} | {x['type']} | {x['question']} | {expected} | {x['best']:.3f} | {outcome} | {top} |")
    lines += [
        "",
        "## Threshold sweep (MIN_SCORE)",
        "",
        "| MIN_SCORE | Hit rate @5 | Hit rate @1 | \"Not found\" accuracy | Wrongly rejected |",
        "|---|---|---|---|---|",
    ]
    for m in sweep:
        mark = " (current)" if abs(m["t"] - current) < 1e-9 else ""
        lines.append(f"| {m['t']:.2f}{mark} | {pct(m['hit5'], n_a)} | {pct(m['hit1'], n_a)} | "
                     f"{pct(m['nf_ok'], n_m)} | {m['false_nf']} |")
    lines += [
        "",
        "Notes: these numbers measure retrieval only (no API key needed). With Claude enabled, the",
        "answer prompt adds a second \"not found\" check: the model must reply \"I could not find this in",
        "the documents.\" when the retrieved passages do not contain the answer.",
        "",
    ]
    (EVAL_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
