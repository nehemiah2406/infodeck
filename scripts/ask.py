"""Ask InfoDeck a question from the command line (same pipeline as the app).

Usage:  .venv\\Scripts\\python scripts\\ask.py "How many sick days do I get?" ["another question" ...]
"""

import sys

from _bootstrap import build_engine
from infodeck.config import get_settings
from infodeck.llm import resolve_llm

engine = build_engine()
status = resolve_llm(get_settings())
print(f"AI mode: {status.label}" + (f" - {status.reason}" if status.reason else ""))

for question in sys.argv[1:]:
    ans = engine.answer(question, status)
    print("\n" + "=" * 78 + f"\nQ: {question}\n" + "-" * 78)
    print(ans.text)
    if ans.note:
        print(f"({ans.note})")
    for i, p in enumerate(ans.passages, 1):
        print(f"\n  [{i}] {p.label}  |  {p.section}  |  relevance {p.score:.2f}")
        print("      " + p.text.replace("\n", "\n      "))
