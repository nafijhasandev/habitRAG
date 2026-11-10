"""
Hybrid retriever with Reciprocal Rank Fusion (RRF).

This module combines dense (vector/semantic) retrieval and sparse (BM25/keyword)
retrieval into a single hybrid search. The results from both retrievers are
merged using Reciprocal Rank Fusion, a simple but effective technique for
combining ranked lists from different sources.

WHY HYBRID RETRIEVAL MATTERS:
    Neither dense nor sparse retrieval is universally better:

    - Dense retrieval (embeddings) excels at semantic understanding. It finds
      chunks that are *about* the same topic even when different words are used.
      "How to build good routines" matches "habit formation process" because
      their embeddings are close in vector space.

    - Sparse retrieval (BM25) excels at exact keyword matching. It reliably
      finds chunks containing specific terms, names, or phrases. Searching for
      "two-minute rule" directly finds chunks with that exact phrase, which
      dense retrieval might rank lower if the surrounding context is different.

    By combining both, we get robust retrieval that handles both semantic
    similarity and keyword precision. Research consistently shows hybrid
    retrieval outperforms either method alone.

HOW RECIPROCAL RANK FUSION (RRF) WORKS:
    RRF is a rank-based fusion method. It does NOT try to compare raw scores
    between retrievers (which would be meaningless since BM25 scores and
    cosine distances are on completely different scales). Instead, it uses
    only the *rank position* of each result.

    For each document d that appears in any result list:
        RRF_score(d) = sum(1 / (k + rank_i)) for each list i where d appears

    Where:
        - k is a constant (config.RRF_K, typically 60) that dampens the
          impact of rank differences. Higher k means ranks matter less.
        - rank_i is the 1-based position of document d in list i.

    Example: if a chunk is rank 1 in dense and rank 3 in sparse (k=60):
        score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

    A chunk found by both retrievers gets a higher fused score than one
    found by only one, which is exactly the behavior we want.
"""

import logging

import config
from core.bm25_index import BM25Index
from core.embeddings import EmbeddingEngine
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion.

    This is the main retrieval interface for the RAG system. It delegates
    to VectorStore (dense) and BM25Index (sparse), then fuses their results
    into a single ranked list.

    Attributes:
        vector_store: The ChromaDB vector store for dense retrieval.
        bm25_index: The BM25 index for sparse keyword retrieval.
        embedding_engine: Used to embed the query for dense search.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        """Store references to the retrieval components.

        Args:
            vector_store: Initialized VectorStore with ingested chunks.
            bm25_index: Built BM25Index with indexed chunks.
            embedding_engine: Loaded EmbeddingEngine for query embedding.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_engine = embedding_engine

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform hybrid retrieval: dense + sparse search merged with RRF.

        Steps:
            1. Embed the query using the embedding engine.
            2. Run dense search via the vector store.
            3. Run sparse search via the BM25 index.
            4. Merge results using Reciprocal Rank Fusion.
            5. Return the top_k results from the fused ranking.

        Each result includes a 'sources' field indicating which retriever(s)
        found it ('dense', 'sparse', or 'both'). This is useful for debugging
        and understanding retrieval behavior.

        Args:
            query: The user's search query as raw text.
            top_k: Number of final results to return after fusion.

        Returns:
            List of dicts sorted by RRF score (descending), each with:
                - chunk_id (str): Chunk identifier.
                - text (str): The chunk's full text.
                - metadata (dict): Chapter, section, and page info.
                - score (float): The fused RRF score.
                - sources (str): Which retriever(s) found this chunk -
                  one of 'dense', 'sparse', or 'both'.
        """
        logger.info("Hybrid retrieval for: '%.80s...'", query)

        # Step 1: Embed the query
        query_embedding = self.embedding_engine.embed_query(query)

        # Step 2: Dense retrieval via vector store
        dense_results = self.vector_store.search(
            query_embedding, top_k=config.RETRIEVAL_TOP_K
        )
        logger.debug("Dense retrieval returned %d results", len(dense_results))

        # Step 3: Sparse retrieval via BM25
        sparse_results = self.bm25_index.search(
            query, top_k=config.RETRIEVAL_TOP_K
        )
        logger.debug("Sparse retrieval returned %d results", len(sparse_results))

        # Step 4: Merge with Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            dense_results, sparse_results, k=config.RRF_K
        )

        # Step 5: Take top_k and log sources for debugging
        top_results = fused[:top_k]

        source_counts = {"dense": 0, "sparse": 0, "both": 0}
        for result in top_results:
            source_counts[result["sources"]] += 1

        logger.info(
            "Hybrid retrieval returned %d results "
            "(from dense: %d, sparse: %d, both: %d)",
            len(top_results),
            source_counts["dense"],
            source_counts["sparse"],
            source_counts["both"],
        )

        return top_results


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    For each document, the RRF score is the sum of 1/(k + rank) across
    all lists the document appears in. Documents found by both retrievers
    naturally get higher scores.

    Args:
        dense_results: Ranked results from dense (vector) retrieval.
        sparse_results: Ranked results from sparse (BM25) retrieval.
        k: The RRF constant. Higher values reduce the impact of rank
            position differences. Standard value is 60.

    Returns:
        Merged list of results sorted by RRF score (descending).
        Each result includes a 'sources' field.
    """
    # Track which chunk_id appears in which retriever(s)
    # Key: chunk_id, Value: dict with accumulated data
    fused_scores: dict[str, dict] = {}

    # Process dense results (rank is 1-based)
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_score = 1.0 / (k + rank)

        fused_scores[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "score": rrf_score,
            "in_dense": True,
            "in_sparse": False,
        }

    # Process sparse results, accumulating scores for overlapping results
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_score = 1.0 / (k + rank)

        if chunk_id in fused_scores:
            # Found by both retrievers - add scores
            fused_scores[chunk_id]["score"] += rrf_score
            fused_scores[chunk_id]["in_sparse"] = True
        else:
            fused_scores[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "score": rrf_score,
                "in_dense": False,
                "in_sparse": True,
            }

    # Build the final results with a human-readable 'sources' field
    results = []
    for entry in fused_scores.values():
        if entry["in_dense"] and entry["in_sparse"]:
            sources = "both"
        elif entry["in_dense"]:
            sources = "dense"
        else:
            sources = "sparse"

        results.append({
            "chunk_id": entry["chunk_id"],
            "text": entry["text"],
            "metadata": entry["metadata"],
            "score": entry["score"],
            "sources": sources,
        })

    # Sort by fused score, highest first
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
