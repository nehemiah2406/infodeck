"""Ask InfoDeck a question from the command line (same pipeline as the app).

Usage:  .venv\\Scripts\\python scripts\\ask.py "How many sick days do I get?" ["another question" ...]
"""

import sys

from _bootstrap import build_engine
from infodeck.config import get_settings
from infodeck.llm import resolve_llm


def main() -> None:
    engine = build_engine()
    status = resolve_llm(get_settings())
    print(f"AI mode: {status.label}" + (f" - {status.reason}" if status.reason else ""))

    for question in sys.argv[1:]:
        ans = engine.answer(question, status)
        print("\n" + "=" * 78 + f"\nQ: {question}\n" + "-" * 78)
        if ans.mode == "exact":
            f = ans.fact
            print(f"Exact value: {ans.text}" + (f"\nFormula: {f.formula}" if f.formula else ""))
        else:
            print(ans.text)
        if ans.unverified:
            print(f"(Check this value in the source: {', '.join(ans.unverified)})")
        if ans.note:
            print(f"({ans.note})")
        for i, p in enumerate(ans.passages, 1):
            print(f"\n  [{i}] {p.label}  |  {p.section}  |  relevance {p.score:.2f}")
            print("      " + p.text.replace("\n", "\n      "))


# Required: OCR worker processes re-import this file on Windows ("spawn"), so the code must not
# run on import.
if __name__ == "__main__":
    main()
