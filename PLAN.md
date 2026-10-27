# InfoDeck: Project Plan

## 1. Goal

**InfoDeck** is an internal chatbot that answers questions about company documents.
A user uploads PDFs (for example company policies) and asks questions in plain English,
such as *"How many sick days do I get?"*. InfoDeck:

- answers **only** from the uploaded documents,
- **cites its source** for every answer (file name and page number),
- says **"I could not find this in the documents"** instead of guessing,
- runs on one laptop and is reachable by colleagues on the **same office network** at
  `http://<laptop-ip>:8501`. It is **not** exposed to the public internet.

Constraints: deliver everything today, keep the code in a public GitHub repository with
**no real company documents and no API keys**, and keep everything testable without an
API key (the tests use a fake LLM).

---

## 2. How RAG works (in simple words)

An LLM like Claude doesn't know your company's policies. We could paste every document into
each question, but that is slow, expensive and sends whole documents to an outside service.
**RAG (Retrieval-Augmented Generation)** does this instead:

1. **Prepare once (indexing).** Read each PDF page by page and cut the text into small pieces
   ("chunks") of about one or two paragraphs. Turn each chunk into a list of numbers (an
   *embedding*) that represents its meaning. Chunks with similar meaning get similar numbers.
   Store the chunks, their numbers and their *file + page* labels in a vector database.
2. **For every question (retrieval).** Turn the question into numbers the same way, and ask
   the database for the 4 chunks whose meaning is closest to the question.
3. **Answer (generation).** Send **only those few chunks** and the question to the LLM, with
   strict instructions: *"Answer only from these excerpts, cite them, and if the answer isn't
   there say 'I could not find this in the documents'."*

```
                 INDEXING (once per document)
  PDF ──► pdfplumber ──► pages ──► chunks ──► embedding model ──► ChromaDB
                          (text + tables)      (local, free)      (vectors + file/page)

                 ANSWERING (every question)
  Question ──► embedding model ──► ChromaDB: top 4 similar chunks
                                        │
                     relevance too low? ├──► "I could not find this in the documents"
                                        ▼
               Prompt = rules + 4 chunks [S1..S4] + question
                                        │
                                        ▼
                    LLM (Claude API, or Ollama as backup)
                                        │
                                        ▼
        Answer + sources, e.g. "18 days [S1]"  →  KRT_Leave_Policy.pdf, page 1
```

There are two layers of protection against made-up answers:
- **Before the LLM:** if no chunk is similar enough to the question (score below a
  threshold), we answer "not found" without calling the LLM at all.
- **Inside the LLM:** the prompt requires the model to cite sources and to use the exact
  "not found" sentence when the excerpts don't contain the answer.

---

## 3. Tech stack and why each library is needed

| Library | Role | Why this choice |
|---|---|---|
| **Python 3.11+** (laptop has 3.13) | Language | You already know it, and it is the standard for data science and ML work. |
| **pdfplumber** | Read PDFs page by page, including tables | Keeps page numbers (needed for citations) and extracts tables much better than plain text extraction. |
| **sentence-transformers** with `BAAI/bge-small-en-v1.5` | Embeddings (text to vectors) | Small (~130 MB), runs on a laptop CPU, free, and strong at retrieval. Documents never leave the laptop for embedding. |
| **ChromaDB** | Vector database | Runs locally and saves to a folder (`storage/chroma/`). No server to install. Stores the file and page labels next to each chunk. |
| **anthropic** (official SDK) | Main LLM: Claude (`claude-sonnet-5-5`) | High-quality, well-grounded answers. Only the 4 retrieved chunks are sent, never whole documents. |
| **ollama** (Python client) | Backup LLM, free and local | Works without internet or an API key. Optional: Ollama is installed only when needed. |
| **python-dotenv** | Load settings from `.env` | Keeps secrets out of the code. `.env` is git-ignored. |
| **Streamlit** | Web UI | Builds a chat web app in pure Python, and other people on the network can open it in their browser. |
| **pytest** | Tests | The LLM is replaced by a fake in tests, so they run with no API key, no internet and no cost. |
| **reportlab** | Generates the sample PDFs | Creates realistic fictional policy documents for development and evaluation. |

**Switching the LLM:** set `LLM_PROVIDER=anthropic`, `LLM_PROVIDER=gemini` or `LLM_PROVIDER=ollama` in `.env`.
All are wrapped behind one small interface (`generate(system, prompt) -> text`), so the
rest of the code doesn't change.

---

## 4. Folder structure

```
Chatbot/
├── app.py                     # Streamlit UI: the "InfoDeck" app (Step 2)
├── src/infodeck/              # (Step 2)
│   ├── config.py              # reads settings from .env (never prints the API key)
│   ├── ingest.py              # PDF -> pages (text + tables) -> chunks with file/page metadata
│   ├── embeddings.py          # loads the sentence-transformers model once
│   ├── store.py               # ChromaDB: add / search / delete documents
│   ├── llm.py                 # AnthropicLLM, OllamaLLM, FakeLLM (for tests)
│   ├── prompts.py             # system prompt + "not found" sentence
│   └── rag.py                 # answer(question) -> answer text + sources
├── scripts/
│   └── make_sample_docs.py    # generates the 3 fictional policy PDFs (Step 1)
├── tests/                     # pytest; LLM is mocked (Step 2)
├── eval/                      # (Step 3)
│   ├── questions.jsonl        # 15 questions, expected answers, expected file + page
│   └── run_eval.py            # retrieval hit rate + answer correctness report
├── data/
│   ├── sample_docs/           # FICTIONAL PDFs, safe to commit
│   └── uploads/               # files uploaded in the app (git-ignored)
├── storage/chroma/            # vector database (git-ignored, rebuilt from PDFs)
├── .streamlit/config.toml     # network + branding settings
├── .env.example               # template settings (committed, no secrets)
├── .env                       # your real settings + API key (git-ignored)
├── requirements.txt
├── PLAN.md
└── README.md                  # (Step 3)
```

---

## 5. Steps

### Step 1: Research, plan and set up (this step)

- Check the laptop (Python, RAM) and install the tools.
- Write this plan.
- Generate 3 fictional sample policy PDFs in `data/sample_docs/`.
- Set up Git, `.venv`, `requirements.txt`, `.gitignore`, `.env.example` and `.streamlit/config.toml`.
- Download the embedding model once to confirm it works offline.

**Done when:**
- [x] `.venv\Scripts\python -c "import anthropic, chromadb, sentence_transformers, streamlit, pdfplumber"` runs without errors.
- [x] `data/sample_docs/` contains 3 PDFs of 3–6 pages each (4 pages each, 5–6 tables each), with numbered sections and tables, and pdfplumber extracts their text and tables.
- [x] The embedding model (`BAAI/bge-small-en-v1.5`) loads and turns a sentence into a 384-number vector.
- [x] `git status` shows that `.env`, `.venv/` and `storage/` are **not** tracked, and `git check-ignore` confirms a test `secret.pdf` would be ignored.
- [x] First commit created (not pushed).

### Step 2: Full pipeline and the InfoDeck app

1. **Ingestion** (`ingest.py`): read each page with pdfplumber. Convert tables into
   readable rows (`Leave type: Sick leave | Entitlement: 10 working days | ...`) so the
   numbers in tables can be found. Split pages into chunks of about 800 characters with
   about 150 characters of overlap, preferring to split at section headings and
   paragraphs. Each chunk keeps `{source file, page, section, chunk_id}`. A file hash
   prevents indexing the same file twice.
2. **Store** (`store.py`): a ChromaDB persistent client in `storage/chroma/` using cosine
   distance. Supports add, search and delete-by-file.
3. **Retrieval** (`rag.py`): embed the question (with bge's query prefix), get the top 4
   chunks, and drop chunks below a similarity threshold (tuned in Step 3). If nothing is
   left, return the "not found" sentence **without calling the LLM**.
4. **Generation** (`llm.py`, `prompts.py`): the system prompt says to answer only from the
   numbered excerpts `[S1]..[S4]`, cite them, treat document text as data and never as
   instructions, and reply exactly `I could not find this in the documents.` when the
   excerpts lack the answer. The app turns `[S1]` into "KRT_Leave_Policy.pdf, page 2".
   Supports Claude (default) and Ollama. `FakeLLM` is used in tests.
5. **App** (`app.py`): page title and header "InfoDeck", no Streamlit branding visible.
   A sidebar to upload PDFs (saved to `data/uploads/`) and list or remove indexed
   documents. A chat box with answers and an expandable "Sources" section (file, page,
   excerpt). A clear error message if the API key is missing. Optional: an `APP_PASSCODE`
   in `.env` so only staff who know it can upload files.
6. **Tests** (`tests/`): chunking keeps page numbers, tables are extracted, the
   not-found path skips the LLM, citations map to the correct file and page, the provider
   switch works, and the API key never appears in logs or error messages.

**Done when:**
- [ ] `pytest` passes with **no** API key and no internet (the LLM is mocked).
- [ ] `python -m infodeck.ingest data/sample_docs` indexes the 3 sample PDFs.
- [ ] `streamlit run app.py` opens InfoDeck. Asking "How many earned leave days do I get?" gives an answer that cites `KRT_Leave_Policy.pdf, page N` (once the key is added, or with Ollama).
- [ ] An off-topic question ("What is the dental insurance limit?") returns "I could not find this in the documents."
- [ ] A colleague's computer on the same Wi-Fi can open `http://<laptop-ip>:8501`.

### Step 3: Evaluation, README and demo

1. **Test set** (`eval/questions.jsonl`): 15 questions. About 11 can be answered from the
   documents and are spread across all 3 PDFs, including answers that only appear in
   tables. About 4 have answers that are **not** in the documents (for example dental
   insurance, pets in the office, stock options, the WiFi password). Each record has
   `question`, `expected_answer`, `expected_source` (file name), `expected_page`, and
   `answerable` (true/false).
2. **Metrics** (`eval/run_eval.py`):
   - **Retrieval hit rate @4:** the share of answerable questions where the expected
     file and page appear in the top 4 retrieved chunks. This measures retrieval only,
     with no LLM, so it runs for free.
   - **Answer correctness:** answerable questions must contain the key facts (for example
     "18" and "days") and cite the correct source. Unanswerable questions must return the
     "not found" sentence. This is scored with simple rules (key-fact matching), with
     manual review of any failures.
   - **Refusal accuracy:** 4 out of 4 unanswerable questions correctly refused.
   - The script saves a results table (`eval_results/`, git-ignored) and prints a summary.
3. **Tune:** adjust the chunk size, top-k and similarity threshold based on the results.
4. **README.md:** what InfoDeck is, setup steps, how to add the API key, how to run the
   app and tests, how to share it on the LAN, privacy notes, the evaluation results table
   and known limitations.
5. **Demo:** a 5-minute script covering upload, 3 questions answered with citations,
   1 "not found" question, and the evaluation results.

**Done when:**
- [ ] `python eval/run_eval.py` runs and prints the hit rate, correctness and refusal accuracy.
- [ ] Targets met (or the gap is explained): hit rate ≥ 90%, correctness ≥ 80%, refusals 4/4.
- [ ] A new person can install and run InfoDeck by following only the README.
- [ ] The demo works end to end on the office network.

---

## 6. Security and privacy rules

- `.env` (API key) and `data/uploads/` are git-ignored. **All** `*.pdf` / `*.docx` files
  are ignored except `data/sample_docs/`, which contains only fictional documents.
- The API key is read from `.env` only. It is never printed, logged or shown in the UI.
- Only the retrieved chunks are sent to the LLM, never whole documents. With
  `LLM_PROVIDER=ollama`, nothing leaves the laptop.
- The app is reachable only on the local network. The Windows Firewall rule is limited to
  **Private** networks and port 8501.
- Before every `git push`, run `git status` and check that only code and sample files are
  listed.
