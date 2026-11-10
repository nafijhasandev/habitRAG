"""
Query engine that orchestrates the full RAG pipeline.

WHY THIS EXISTS:
The RAG pipeline has many steps: classification, expansion, retrieval,
reranking, context building, and generation. Each step is handled by a
separate component. The QueryEngine ties them all together into a single
coherent workflow, managing the data flow between components and handling
failures gracefully.

HOW THE PIPELINE WORKS:
1. CLASSIFY: Determine the query type (factual/analytical/comparison/summary)
   to select the prompt and retrieval depth.
2. EXPAND: Use the LLM to generate alternative query phrasings for broader
   retrieval coverage. Skipped if the LLM is unavailable.
3. RETRIEVE: Run each expanded query through the retriever (dense + sparse
   hybrid search), combine scores by chunk_id, and keep 15 candidates.
4. RERANK: Score all candidates with the cross-encoder reranker and keep
   the top_k most relevant.
5. BUILD CONTEXT: Format the top chunks into a context string, ordered by
   book position, with chapter/page headers.
6. GENERATE: Send the context + question to the LLM for answer generation.
   If the LLM is unavailable, return the retrieved chunks without an answer.

Each step is timed for performance monitoring. The pipeline degrades
gracefully: if expansion fails, we search with just the original query;
if the LLM is down, we return sources without a generated answer.
"""

import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import config
from core.citations import format_cited_answer
from core.context_builder import build_context, build_prompt
from core.query_classifier import classify_query
from core.scope_checker import OUT_OF_SCOPE_ANSWER, is_in_scope

logger = logging.getLogger(__name__)


class QueryEngine:
    """Orchestrates the full RAG query pipeline.

    Connects the retriever, reranker, LLM client, and query expander into
    a single pipeline that takes a user question and returns a grounded
    answer with sources.

    Attributes:
        retriever: The hybrid retriever (dense + sparse) component.
        reranker: The cross-encoder reranker component.
        llm_client: The hosted LLM client for generation and expansion.
        query_expander: The query expansion component.
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        llm_client: Any,
        query_expander: Any,
    ) -> None:
        """Store references to each pipeline component.

        Args:
            retriever: Object with a `search(query, top_k) -> list[dict]` method.
            reranker: Object with a `rerank(query, chunks, top_n) -> list[dict]` method.
            llm_client: Object with async `generate(messages, stream)`, `is_available()`,
                and `expand_query(query)` methods.
            query_expander: Object with an async `expand(query) -> list[str]` method.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.query_expander = query_expander

    async def query(
        self, question: str, top_k: int | None = None
    ) -> dict[str, Any]:
        """Run the full RAG pipeline and return an answer with sources.

        Steps: classify -> expand -> retrieve -> rerank -> build context -> generate.

        If any component fails, the pipeline degrades gracefully:
        - Expansion failure: searches with original query only
        - Retrieval returns nothing: returns empty answer with a message
        - LLM unavailable: returns sources without a generated answer

        Args:
            question: The user's natural language question.
            top_k: Number of final chunks to use for context. By default,
                the query type selects the SRS retrieval depth.

        Returns:
            A dict with:
            - answer (str): The LLM-generated answer, or a fallback message.
            - sources (list[dict]): The top chunks with metadata.
            - query_type (str): The classified query type.
            - timing (dict[str, float]): Seconds taken for each pipeline step.
        """
        timing: dict[str, float] = {}
        total_start = time.monotonic()

        if not is_in_scope(question):
            return {
                "answer": OUT_OF_SCOPE_ANSWER,
                "sources": [],
                "query_type": "out_of_scope",
                "timing": {"total": time.monotonic() - total_start},
            }

        # Step 1: Classify the query
        step_start = time.monotonic()
        query_type, classification_reason = classify_query(question)
        top_k = top_k if top_k is not None else config.QUERY_TOP_K[query_type]
        timing["classify"] = time.monotonic() - step_start
        logger.info("Query type: %s (%s)", query_type, classification_reason)

        # Step 2: Expand the query (skip if LLM is unavailable)
        step_start = time.monotonic()
        expanded_queries = await self._try_expand(question)
        timing["expand"] = time.monotonic() - step_start

        # Step 3: Retrieve chunks for each expanded query
        step_start = time.monotonic()
        retrieval_queries = _retrieval_queries(question, query_type, expanded_queries)
        all_chunks = self._retrieve_for_queries(retrieval_queries)
        timing["retrieve"] = time.monotonic() - step_start

        if not all_chunks:
            timing["total"] = time.monotonic() - total_start
            logger.warning("No chunks retrieved for question: '%.80s'", question)
            return {
                "answer": "I could not find any relevant passages for your question.",
                "sources": [],
                "query_type": query_type,
                "timing": timing,
            }

        # Step 4: Rerank combined results
        step_start = time.monotonic()
        reranked_chunks = self.reranker.rerank(question, all_chunks, top_n=top_k)
        timing["rerank"] = time.monotonic() - step_start

        # Step 5: Build context
        step_start = time.monotonic()
        context_str = build_context(reranked_chunks)
        messages = build_prompt(question, context_str, query_type)
        timing["build_context"] = time.monotonic() - step_start

        # Prepare source metadata for the response
        sources = _extract_sources(reranked_chunks)

        # Step 6: Generate answer via LLM
        step_start = time.monotonic()
        answer = await self._try_generate(messages, sources)
        timing["generate"] = time.monotonic() - step_start

        timing["total"] = time.monotonic() - total_start

        logger.info(
            "Pipeline completed in %.2fs (classify=%.3fs, expand=%.3fs, "
            "retrieve=%.3fs, rerank=%.3fs, context=%.3fs, generate=%.3fs).",
            timing["total"],
            timing["classify"],
            timing["expand"],
            timing["retrieve"],
            timing["rerank"],
            timing["build_context"],
            timing["generate"],
        )

        return {
            "answer": answer,
            "sources": sources,
            "query_type": query_type,
            "timing": timing,
        }

    async def query_stream(
        self, question: str, top_k: int | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the pipeline and stream the answer token-by-token.

        Yields query metadata, answer chunks, source passages, and a final
        citation-checked answer.

        Args:
            question: The user's natural language question.
            top_k: Number of final chunks to use. Defaults by query type.

        Yields:
            Dicts with either:
            - {"type": "metadata", "query_type": str}
            - {"type": "chunk", "content": str}
            - {"type": "sources", "sources": [...]}
            - {"type": "done", "answer": str, "timing": dict}
            - {"type": "error", "message": str}
        """
        timing: dict[str, float] = {}
        total_start = time.monotonic()

        if not is_in_scope(question):
            yield {"type": "metadata", "query_type": "out_of_scope"}
            yield {"type": "chunk", "content": OUT_OF_SCOPE_ANSWER}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "answer": OUT_OF_SCOPE_ANSWER,
                   "timing": {"total": time.monotonic() - total_start}}
            return

        # Steps 1-5: same as non-streaming query
        step_start = time.monotonic()
        query_type, _ = classify_query(question)
        top_k = top_k if top_k is not None else config.QUERY_TOP_K[query_type]
        timing["classify"] = time.monotonic() - step_start

        step_start = time.monotonic()
        expanded_queries = await self._try_expand(question)
        timing["expand"] = time.monotonic() - step_start

        step_start = time.monotonic()
        retrieval_queries = _retrieval_queries(question, query_type, expanded_queries)
        all_chunks = self._retrieve_for_queries(retrieval_queries)
        timing["retrieve"] = time.monotonic() - step_start

        if not all_chunks:
            yield {
                "type": "error",
                "message": "No relevant passages found for your question.",
            }
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "timing": {"total": time.monotonic() - total_start}}
            return

        step_start = time.monotonic()
        reranked_chunks = self.reranker.rerank(question, all_chunks, top_n=top_k)
        timing["rerank"] = time.monotonic() - step_start

        step_start = time.monotonic()
        context_str = build_context(reranked_chunks)
        messages = build_prompt(question, context_str, query_type)
        timing["build_context"] = time.monotonic() - step_start

        sources = _extract_sources(reranked_chunks)

        yield {"type": "metadata", "query_type": query_type}

        # Step 6: Stream LLM response
        step_start = time.monotonic()
        answer_parts: list[str] = []
        try:
            llm_available = await self.llm_client.is_available()
            if not llm_available:
                yield {
                    "type": "error",
                    "message": "LLM is not available. Sources were retrieved but no answer can be generated.",
                }
            else:
                stream = await self.llm_client.generate(messages, stream=True)
                async for chunk_text in stream:
                    answer_parts.append(chunk_text)
                    yield {"type": "chunk", "content": chunk_text}

        except Exception:
            logger.exception("Streaming generation failed")
            yield {"type": "error", "message": "Answer generation failed. Please try again."}

        timing["generate"] = time.monotonic() - step_start
        timing["total"] = time.monotonic() - total_start
        yield {"type": "sources", "sources": sources}
        answer = format_cited_answer("".join(answer_parts), sources) if answer_parts else ""
        yield {"type": "done", "answer": answer, "timing": timing}

    async def _try_expand(self, question: str) -> list[str]:
        """Attempt query expansion, falling back to the original query.

        Args:
            question: The original user query.

        Returns:
            A list of queries (original + expansions, or just original).
        """
        try:
            llm_available = await self.llm_client.is_available()
            if not llm_available:
                logger.info("LLM unavailable, skipping query expansion.")
                return [question]

            return await self.query_expander.expand(question)

        except Exception:
            logger.warning("Query expansion failed, using original query.", exc_info=True)
            return [question]

    def _retrieve_for_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        """Run retrieval for each query and merge results.

        Deduplicates by chunk_id, combines scores from expanded queries,
        and caps the reranker input at the configured candidate count.

        Args:
            queries: One or more query strings to retrieve for.

        Returns:
            A deduplicated list of chunk dicts.
        """
        merged: dict[str, dict[str, Any]] = {}

        for q in queries:
            try:
                results = self.retriever.retrieve(q, top_k=config.RERANK_CANDIDATES)
                for rank, chunk in enumerate(results, start=1):
                    chunk_id = chunk.get("chunk_id", "")
                    if not chunk_id:
                        continue
                    if chunk_id not in merged:
                        merged[chunk_id] = dict(chunk, score=0.0)
                    merged[chunk_id]["score"] += float(
                        chunk.get("score", 1.0 / (config.RRF_K + rank))
                    )
            except Exception:
                logger.warning(
                    "Retrieval failed for query '%.80s'.", q, exc_info=True
                )

        logger.info(
            "Retrieved %d unique chunks from %d queries.", len(merged), len(queries)
        )
        return sorted(
            merged.values(), key=lambda chunk: chunk["score"], reverse=True
        )[:config.RERANK_CANDIDATES]

    async def _try_generate(
        self, messages: list[dict[str, str]], sources: list[dict[str, Any]]
    ) -> str:
        """Attempt LLM generation, returning a fallback message on failure.

        Args:
            messages: The chat messages for the hosted model API.
            sources: Retrieved source metadata for citation validation.

        Returns:
            The generated answer text, or a fallback message if the LLM
            is unavailable or fails.
        """
        try:
            llm_available = await self.llm_client.is_available()
            if not llm_available:
                logger.warning("LLM unavailable for answer generation.")
                return (
                    "The LLM is currently unavailable. "
                    "Here are the relevant passages I found -- "
                    "please review the sources below."
                )

            answer = await self.llm_client.generate(messages, stream=False)
            return format_cited_answer(answer, sources)

        except Exception:
            logger.exception("LLM generation failed")
            return (
                "Answer generation failed due to an error. "
                "Please review the retrieved source passages below."
            )


def _retrieval_queries(
    question: str, query_type: str, expanded_queries: list[str]
) -> list[str]:
    """Search each comparison subject separately, then use expanded wording."""
    if query_type != "comparison":
        return expanded_queries

    pattern = re.search(
        r"(?:compare|difference between)\s+(.+?)\s+"
        r"(?:vs\.?|versus|and)\s+(.+?)[?.!]*$",
        question,
        re.IGNORECASE,
    )
    if pattern is None:
        pattern = re.search(
            r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]*$",
            question,
            re.IGNORECASE,
        )
    if pattern is None:
        return expanded_queries

    subjects = [part.strip(" ?.!") for part in pattern.groups()]
    return list(dict.fromkeys([*subjects, *expanded_queries]))


def _extract_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract source metadata from reranked chunks for the response.

    Includes full text for the API and a short preview for the frontend.

    Args:
        chunks: Reranked chunk dicts.

    Returns:
        A list of dicts with chapter, pages, full text, metadata, score,
        and a text preview (first 200 chars).
    """
    sources = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "chapter": metadata.get("chapter", ""),
            "pages": f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}",
            "text": text,
            "metadata": metadata,
            "rerank_score": chunk.get("rerank_score"),
            "text_preview": text[:200],
        })
    return sources
