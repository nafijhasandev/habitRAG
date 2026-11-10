# How Ask Atomic Habits Works

Ask Atomic Habits answers questions using passages from the local PDF. The book, search indexes, embedding model, and reranker run locally. Hosted providers generate the answer. See [SETUP.md](SETUP.md) for commands to install and run it.

## Ingestion

Run `python main.py ingest` once after setup. `ingestion/pipeline.py` coordinates these steps:

1. `ingestion/pdf_processor.py` extracts PDF text blocks and detects chapter and topic headings, including headings in the middle of a page.
2. `ingestion/chunker.py` assigns each paragraph to its most recent topic. It never mixes different topics in one chunk. Long topics become smaller overlapping chunks with the same topic label. The target is 800 characters with 100 characters of overlap so most passages fit the embedding model's input window. Sentence boundaries stay intact, including unusually long sentences.
3. `core/embeddings.py` converts each chunk into a normalized vector using `sentence-transformers/all-MiniLM-L6-v2`, in batches of 32.
4. `core/vector_store.py` writes vectors, text, and metadata to a cosine-distance ChromaDB collection. `core/bm25_index.py` builds a keyword index from the same chunks.

Both indexes are stored under `data/`. `data/chunks.json` is an inspection copy and is not read during search. A normal ingest skips work when ChromaDB already contains chunks. Use `python main.py ingest --force` after changing the PDF or chunking rules. This rebuilds both indexes.

## Answering a question

`core/query_engine.py` handles CLI and web questions:

1. A rule-based scope check returns a book-only message for clearly unrelated questions. These questions do not search the indexes.
2. The query classifier selects a factual, analytical, comparison, or summary prompt. By default, factual questions use three passages, analytical and comparison questions use five, and summaries use eight. Comparison questions search each named subject separately.
3. If a provider key is configured, query expansion suggests alternate wording. An expansion failure falls back to the original question.
4. `core/retriever.py` searches ChromaDB and BM25. Reciprocal Rank Fusion merges each search's ranked results. Results from expanded queries are scored together and capped at 15 unique candidates before reranking.
5. `core/reranker.py` scores the candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the highest scoring chunks.
6. `core/context_builder.py` orders the chosen passages by page, removes near duplicates, adds chapter, page, and topic labels, and limits context length.
7. `core/llm_client.py` sends the question and context to Groq, then tries OpenRouter and Vercel AI Gateway if an earlier provider fails. Streaming switches providers only before any answer text has been shown, avoiding duplicate partial answers. `core/citations.py` checks chapter and page citations against retrieved passages, normalizes supported citations, and adds a citation from the highest-ranked passage with chapter metadata when the model omits one. The answer and source metadata are returned to the CLI or streamed to the browser.

The web API is `POST /api/query`. It sends Server-Sent Events in this order: `metadata` with query type, zero or more answer `chunk` events, `sources`, then `done` with the final formatted answer; failures use an `error` event. `GET /api/health` reports index status and whether a provider key is configured. The browser streams draft answer text, then replaces it with the citation-checked answer when `done` arrives. The source panel lists the retrieved passages.

Source locations come from PDF metadata. Citation validation checks locations, not whether every claim is supported by its cited passage. Check the source panel for the actual retrieved text. If retrieval finds nothing, the system reports that no relevant passages were found. If every configured provider fails, the CLI query can still return retrieved sources.

## Performance and tuning

`config.py` sets chunk size, retrieval counts, context limit, and model names. `QueryEngine.query()` records time spent in classification, expansion, retrieval, reranking, context building, and generation. The CLI prints the main timings.

On November 5, 2026, an offline local probe used the 576-chunk index, a stored ChromaDB embedding, and 30 warmed runs of a five-chunk context. Median context assembly fell from 2.51 ms to 1.36 ms after reusing each chunk's text bigrams during duplicate detection. ChromaDB search measured 2.65 ms before and 4.16 ms after in separate runs, so those figures should not be treated as a change caused by this patch. BM25 and rank fusion were below 1 ms in both runs.

A historical CPU CLI question using Ollama before the candidate cap took 102.45 seconds: 24.52 seconds for query expansion, 3.44 seconds to rerank 44 candidates, and 74.38 seconds for answer generation. A later same-process probe on that question compared 44 and 15 candidates with the cached reranker. Two warmed runs gave median reranking times of 3.115 and 1.325 seconds respectively, with the same top three passages. These Ollama generation timings do not describe the current hosted-provider path.

The quality tests use fixed ranked results to verify fusion behavior, a small indexed corpus to check BM25 scoring, and a ten-page PDF excerpt to check matching ChromaDB and BM25 chunk IDs. A manual full-book probe found the expected chapter within the top three passages for habit stacking, the two-minute rule, and the four laws of behavior change. A broader retrieval quality evaluation still needs more labeled book questions and expected passages.
