"""
Cross-encoder reranker for improving retrieval precision.

WHY THIS EXISTS:
In a RAG pipeline, the initial retrieval step (dense + sparse) casts a wide net
to find candidate chunks. These retrievers use bi-encoders that encode the query
and each document independently into separate vectors, then compare them via
cosine similarity. This is fast but imprecise because the model never sees
the query and document together -- it cannot reason about their fine-grained
interaction (e.g., word-level matches, negation, paraphrases).

A cross-encoder, by contrast, takes the query and a candidate document as a
single input pair and processes them jointly through a transformer. This means
the model can attend across both sequences simultaneously, producing a much
more accurate relevance score. The cost is that cross-encoders are too slow to
run over an entire corpus (O(n) forward passes per query), so they are used
only as a second-stage reranker over a small candidate set (typically 10-30
chunks from the first stage).

HOW IT WORKS:
1. The bi-encoder retriever returns ~15 candidate chunks (config.RERANK_CANDIDATES).
2. The Reranker scores each (query, chunk.text) pair using the cross-encoder.
3. Chunks are sorted by the new score, and the top_n are returned.
4. These top_n chunks go into the context builder for the final LLM prompt.

The model used (ms-marco-MiniLM-L-6-v2) was trained on the MS MARCO passage
ranking dataset, making it well-suited for question-to-passage relevance.
"""

import logging
from typing import Any

from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieval candidates using a cross-encoder model.

    A cross-encoder sees query + document together in a single forward pass,
    enabling fine-grained relevance scoring that bi-encoders cannot achieve.
    This class wraps the sentence-transformers CrossEncoder for easy use in
    the RAG pipeline.

    Attributes:
        model: The loaded CrossEncoder model instance.
        model_name: Name of the model for logging.
    """

    def __init__(self) -> None:
        """Load the cross-encoder model specified in config.

        The model is loaded once and reused for all reranking calls.
        Loading can take a few seconds and ~200MB of memory.
        """
        self.model_name: str = config.RERANKER_MODEL
        logger.info("Loading cross-encoder model: %s", self.model_name)
        self.model = CrossEncoder(
            self.model_name, cache_folder=str(config.MODEL_CACHE_DIR)
        )
        logger.info("Cross-encoder model loaded successfully.")

    def rerank(
        self, query: str, chunks: list[dict[str, Any]], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """Re-score and re-sort chunks by relevance to the query.

        Each chunk is paired with the query and fed through the cross-encoder.
        The model outputs a single relevance score per pair. Chunks are then
        sorted by descending score and the top_n are returned.

        Args:
            query: The user's search query.
            chunks: Candidate chunks from the retriever. Each dict must have
                at least 'chunk_id', 'text', and 'metadata' keys.
            top_n: Number of top-scoring chunks to return. Defaults to
                config.FINAL_TOP_K if not specified.

        Returns:
            The top_n chunks sorted by relevance, each with an added
            'rerank_score' field containing the cross-encoder score.
        """
        if top_n is None:
            top_n = config.FINAL_TOP_K

        if not chunks:
            logger.warning("Reranker received empty chunk list.")
            return []

        if not query.strip():
            logger.warning("Reranker received empty query, returning chunks as-is.")
            return chunks[:top_n]

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        logger.debug(
            "Reranking %d chunks for query: '%.80s...'", len(chunks), query
        )

        scores = self.model.predict(pairs)

        # Log score distribution for debugging retrieval quality
        if len(scores) > 0:
            logger.info(
                "Rerank scores -- min: %.4f, max: %.4f, mean: %.4f, count: %d",
                float(scores.min()),
                float(scores.max()),
                float(scores.mean()),
                len(scores),
            )

        # Attach scores to chunks and sort by descending relevance
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            scored_chunk = dict(chunk)
            scored_chunk["rerank_score"] = float(score)
            scored_chunks.append(scored_chunk)

        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        top_chunks = scored_chunks[:top_n]

        logger.info(
            "Reranker selected top %d chunks (score range: %.4f to %.4f).",
            len(top_chunks),
            top_chunks[-1]["rerank_score"] if top_chunks else 0.0,
            top_chunks[0]["rerank_score"] if top_chunks else 0.0,
        )

        return top_chunks
