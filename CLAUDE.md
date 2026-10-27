# InfoDeck - project memory for Claude Code

## What this is
InfoDeck is a document Q&A chatbot (RAG). Users upload files and ask questions; answers come ONLY from the documents, with file name and page. Built by Hari (data science intern) as a practice task; demo for sir. Maritime context: engineering calculation sheets, class rules, policies.

## Stack
- Python 3, Streamlit UI (app.py), code in src/infodeck/, tests in tests/, scripts in scripts/
- Loaders: PDF (pdfplumber), DOCX, TXT, images, XLSX/CSV, PPTX
- PARSER=local|gemini (default local). local = file text + tables, local OCR (rapidocr) for scanned pages/images, ignores bad hidden OCR layers. gemini = page images sent to Gemini (free key has a daily limit; big PDFs need a paid key)
- Parsed pages cached in storage/parsed/<doc_id>/page_N.md
- Local OCR runs in parallel worker processes (src/infodeck/ocr.py, OCR_WORKERS=6, OCR_DPI=200, spawn-safe: scripts need `if __name__ == "__main__":`)
- Uploads are read in the background, one file at a time (src/infodeck/jobs.py); pages become searchable in batches while a big PDF loads
- Embeddings: BAAI/bge-small-en-v1.5 (local); re-ranker: cross-encoder/ms-marco-MiniLM-L6-v2 (local)
- Vector store: ChromaDB in storage/
- Exact values: src/infodeck/facts.py, FACT_MATCH_SCORE=0.80
- LLM providers (src/infodeck/llm.py, resolve_llm): gemini (gemini-3.8-flash), anthropic, ollama; no key or quota used up -> sources mode
- Settings: .env (template .env.example), read in src/infodeck/config.py
- Evaluation: eval/questions.csv + scripts/eval.py (last score 14/15)

## Product rules (never break)
1. Answers only from the documents. If not found, say "I could not find this in the documents." No general-knowledge fallback (decided 27 Oct).
2. Numbers, units, formulas and quotes must be EXACT as in the document. Never calculate, round or invent.
3. Always show the source file and page.

## Safety rules
- Never print, log, show or commit .env or any API key.
- Never add real company files to the repo, tests or samples. Fictional data only.
- data/uploads/, storage/ and .env stay in .gitignore.
- Do NOT git push. Commit only when I ask; I push myself.
- Save Gemini quota: do not run the full eval or big Gemini jobs without asking me first.

## How to work
- Before a big change: show a short plan and wait for my OK.
- One feature per task. Keep existing features working.
- Every change: add/update tests, run the full pytest suite (.venv\Scripts\python -m pytest), all must pass.
- Explain changes in simple English (I am a beginner), and end with: what changed, test result, how I can try it in the app.
- Windows laptop, PowerShell, VS Code. Use .venv.

## History
- Part 1-4: core app, README, New Era UI
- Part 5: "+" upload in chat box, sidebar toggle
- Part 6: Gemini provider (gemini-3.8-flash)
- Part 7 (51c8c7b): complex documents - OCR, PARSER=local|gemini, structure chunking, exact-value facts index, number check, eval 14/15, 71 tests
- Part 8: faster big scanned PDFs - parallel OCR (6 workers), background reading queue with "page X of N" in the sidebar, questions while loading, stop/resume; 20-page scan 87-91 s -> 52-54 s, 204 pages ~15 -> ~9 min; eval still 14/15 (--no-llm); 80 tests
- Keep this list updated after each part.