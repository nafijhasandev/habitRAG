# Setup Ask Atomic Habits

These commands use Windows PowerShell from the project folder. Python 3.11 or newer, the book PDF, an API key for at least one hosted model provider, and an internet connection are required. The commands below use Python 3.12.

## Install

```powershell
cd path\to\RAGSystem
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first ingest and first server start download the embedding and reranker models from Hugging Face into `data/model_cache`. Their names are in `config.py`.

## Configure hosted models

Open the local `.env` file in the project folder and paste your keys after the equals signs:

```text
GROQ_API_KEY=
OPENROUTER_API_KEY=
AI_GATEWAY_API_KEY=
```

The app uses Groq first, OpenRouter second, and Vercel AI Gateway third. It skips providers without a key and tries the next provider if a request fails or hits a rate limit. The defaults are `openai/gpt-oss-120b` on Groq, `openrouter/free` on OpenRouter, and `nvidia/nemotron-3.5-lightning-free` on Vercel. OpenRouter's `openrouter/auto` can select paid models, so the default uses its free router. To change a model, add `GROQ_MODEL=`, `OPENROUTER_MODEL=`, or `VERCEL_AI_MODEL=` to `.env` with the desired model ID. Restart the server after editing `.env`. The file is excluded by `.gitignore` and must not be shared.

Vercel's free Nemotron provider may retain prompts and responses for model improvement. Each model request includes the question and retrieved book passages.

## Build the indexes

Place a legally obtained copy of `Atomic Habits by James Clear.pdf.pdf` in the project folder, or update `PDF_PATH` in `config.py`. The book PDF is excluded from GitHub. Then run:

```powershell
.\venv\Scripts\python.exe main.py ingest
```

This creates ChromaDB and BM25 data under `data/`. Ingestion skips a populated ChromaDB collection. To rebuild both indexes after changing the PDF or chunk settings, run:

```powershell
.\venv\Scripts\python.exe main.py ingest --force
```

## Ask a question

```powershell
.\venv\Scripts\python.exe main.py query "What is habit stacking?"
.\venv\Scripts\python.exe main.py serve
```

Open `http://localhost:8000` in a browser. The server stays active in that terminal until you press Ctrl+C. `http://localhost:8000/api/health` reports whether the index is loaded and at least one provider key is configured. The CLI prints the answer, source passages, and pipeline timings. The web page streams the answer and shows sources in its source panel.

## Tests and lint

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe -m pip install ruff
.\venv\Scripts\ruff.exe check core ingestion web tests main.py config.py
$jsTests = (Get-ChildItem .\tests -Filter '*.js').FullName
node --test $jsTests
```

The Python tests mock hosted model requests and use temporary local PDF and index data. They do not need API keys or a model download. The JavaScript tests need Node.js. Install Ruff once if it is not already in the virtual environment.

## Common problems

- **Model download fails:** Restore network access to Hugging Face, then retry. An offline machine needs the model files cached from a prior successful run.
- **No model provider configured:** Put at least one API key in `.env`, then restart the server.
- **All providers fail:** Check each account's key, rate limit, and model availability. The app tries only providers with configured keys.
- **No ingested data:** Run the ingest command before starting the web server.
- **PDF not found:** Check the filename and `PDF_PATH` in `config.py`.
- **Wrong or stale search results:** Rebuild both indexes with `ingest --force` after confirming the PDF and chunk settings.
- **Old L2 vector index:** Rebuild with `ingest --force` to create a cosine collection and normalized vectors.
- **Port 8000 already in use:** Stop the other process using that port, or change `WEB_PORT` in `config.py`.
