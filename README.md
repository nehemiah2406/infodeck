# InfoDeck

**Ask questions about your company documents and get short answers with sources.**

Upload policy PDFs (or Word and text files) and ask questions in plain English, such as
*"What is the hotel limit per night in Mumbai?"*. InfoDeck finds the relevant paragraphs and
answers **only from your documents**. Every answer cites the file and page it came from. If
the documents don't contain the answer, InfoDeck says *"I could not find this in the
documents."* instead of guessing.

![InfoDeck screenshot](docs/screenshot.png)

> The documents in `data/sample_docs/` belong to a **fictional company** ("Kestrel Ridge
> Technologies Ltd.") and exist only for testing and demos. They are not real policies.

---

## How it works (RAG)

InfoDeck uses **Retrieval-Augmented Generation (RAG)**. It does not send whole documents to
an AI. Instead, it first finds the few paragraphs that matter and sends only those.

```
 INDEXING (once per document)
 PDF/DOCX/TXT ─► read page by page ─► ~800-char chunks ─► embedding model ─► ChromaDB
                 (tables kept as rows)  (file + page kept)   (local, free)    (on disk)

 ANSWERING (every question)
 Question ─► 20 most similar chunks ─► best match too weak? ─► "I could not find this in the documents."
                                            │ no
                                            ▼
                        re-ranker picks the best 5 chunks
                                            │
                       ┌────────────────────┴────────────────────┐
              AI available (Claude / Ollama)               no API key yet
                       │                                         │
       short answer with citations                  "sources mode": shows the
       like [KRT_Leave_Policy.pdf, p.1]             matching passages + file/page
```

## Features

- Upload **PDF, DOCX and TXT** files with the **"+" button** in the chat box or from the sidebar. Tables are read row by row, and scanned PDFs with no text get a clear warning.
- Uploading the same file again is detected by its fingerprint (SHA-256 hash), so nothing is duplicated.
- Answers cite **file + page**. A **Sources** panel shows the exact paragraphs that were used.
- **Three AI modes**, chosen in `.env`:
  - Claude API (default)
  - Ollama (free and local)
  - **Sources mode** (no AI): shows the best matching passages. This is used automatically
    when there is no API key, so the app never crashes because a key is missing.
- Works on the office network: colleagues open `http://<laptop-ip>:8501` in their browser.

## Tech stack

| Part | Library | Why |
|---|---|---|
| PDF reading | `pdfplumber` | Text **and tables** page by page, with page numbers for citations |
| Word files | `python-docx` | Paragraphs and tables from `.docx` |
| Embeddings | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | Small, fast, runs locally on CPU |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | Re-orders candidates by how well they answer the question |
| Vector store | `ChromaDB` (persistent folder) | Local, no server needed |
| LLM | `anthropic` (Claude, default `claude-sonnet-5-5`) or `ollama` | Answers streamed live; only 5 short passages are sent |
| UI | `Streamlit` | Chat web app; other people on the network can open it |
| Tests | `pytest` | 41 fast tests with the LLM mocked: no API key or internet needed |

## Run it on Windows

**First time only** (needs Python 3.11+ and Git):

```powershell
git clone https://github.com/nehemiah2406/infodeck.git
cd infodeck
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

**Start:** double-click **`run.bat`**. The first start takes about a minute while the models
download and load. The window prints two addresses:

- On this laptop: `http://localhost:8501`
- For colleagues on the same network: `http://<this-laptop-ip>:8501`

To let colleagues connect, allow the port in Windows Firewall once. Open PowerShell
**as Administrator** and run:

```powershell
New-NetFirewallRule -DisplayName "InfoDeck (port 8501)" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow -Profile Domain,Private
```

The three sample PDFs load automatically the first time the app starts.

## Add your Claude API key

1. Create a key at <https://console.anthropic.com> (API Keys).
2. Open the file **`.env`** in the project folder and paste the key after the equals sign:
   `ANTHROPIC_API_KEY=your-key-here` (no quotes, no spaces).
3. Save the file and **refresh the browser**. You don't need to restart. The sidebar pill
   changes from *Sources mode* to *Claude*.

`.env` is listed in `.gitignore`, so it is never committed. The app never prints or logs the key.

**Free local alternative:** install [Ollama](https://ollama.com), run `ollama pull llama3.2:3b`,
then set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2:3b` in `.env`.

**Google Gemini:** create a key at <https://aistudio.google.com/apikey>, then set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY=your-key-here` in `.env`.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `gemini` or `ollama` |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Claude model used for answers |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used when `LLM_PROVIDER=gemini` |
| `TOP_K` | `5` | Passages used per question |
| `MIN_SCORE` | `0.55` | If the best passage scores lower than this, the answer is "not found" and no AI call is made |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Set to `none` to switch re-ranking off |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunk length and overlap in characters |

## Evaluation

`eval/questions.csv` contains 10 questions about the sample PDFs: 6 answerable, 2 worded
differently from the documents, and 2 whose answer is not in the documents. Run the
evaluation with:

```powershell
.venv\Scripts\python eval\run_eval.py
```

| Metric | Result |
|---|---|
| Retrieval hit rate @5 (right file + page in the top 5) | **8/8 (100%)** |
| Right page ranked first (@1) | 6/8 (75%) |
| "Not found" accuracy (questions not in the documents) | 1/2 (50%) |
| Answerable questions wrongly rejected | 0/8 |

The full per-question table and a threshold sweep are in [eval/results.md](eval/results.md).
A higher `MIN_SCORE` would reject the off-topic "guest WiFi password" question, but it would
also reject real questions, so `0.55` was kept. With Claude enabled, the prompt adds a second
"not found" check.

## Tests

```powershell
.venv\Scripts\python -m pytest            # 41 fast tests, LLM mocked
.venv\Scripts\python -m pytest -m slow    # 6 tests with the real embedding + re-ranking models
```

The tests cover chunking, duplicate uploads, the "not found" rule, sources mode, error
handling for Claude and Ollama, and API-key safety.

## Project structure

```
app.py                  Streamlit UI (InfoDeck)
run.bat                 Double-click launcher (LAN mode)
src/infodeck/           loaders, chunking, embeddings, store, llm, prompts, engine
scripts/                ingest_docs.py, ask.py (command line), make_sample_docs.py
eval/                   questions.csv, run_eval.py, results.md
tests/                  pytest suite
data/sample_docs/       fictional sample PDFs (safe to share)
```

## Known limits

- **Off-topic questions that sound similar can pass the first check.** For example, "guest
  WiFi password" scores close to real IT-policy questions. In sources mode they show
  unrelated passages. With Claude, the prompt answers "not found".
- **The small model can miss unusual wording.** Questions phrased very differently from
  the documents may need rephrasing.
- **Each question is answered on its own.** Follow-up questions don't use earlier chat messages.
- **One shared library.** Everyone on the network sees the same documents and can upload or
  remove them. There is no login.
- **Scanned PDFs need OCR first.** InfoDeck warns about them but cannot read images.
- Answers have only been tested in sources mode and with a mocked Claude client. Check
  important answers against the cited page.

## Privacy

- Embedding and search run **locally**. With Claude, only the question and the 5 retrieved
  passages are sent to the API, never whole documents. With Ollama, nothing leaves the laptop.
- `.env`, uploaded files (`data/uploads/`), the vector store (`storage/`) and every PDF/Office
  file outside `data/sample_docs/` are git-ignored.
