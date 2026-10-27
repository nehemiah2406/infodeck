"""Index all supported files (PDF, Word, Excel, PowerPoint, text, images) in a folder.

Usage:  .venv\\Scripts\\python scripts\\ingest_docs.py [folder]   (default: data/sample_docs)
"""

import sys
from pathlib import Path

from _bootstrap import build_engine

engine = build_engine()
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else engine.settings.sample_dir
for r in engine.ingest_folder(folder):
    print(f"[{r.status}] {r.message}")
    for w in r.warnings:
        print(f"   warning: {w}")
print(f"Total chunks in store: {engine.store.count()}")
