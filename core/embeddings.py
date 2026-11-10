"""
Embedding engine for the RAG system.

This module wraps a sentence-transformers model to convert text into dense
vector representations (embeddings). Embeddings capture semantic meaning:
texts with similar meaning produce vectors that are close together in the
embedding space. This is the foundation of dense (semantic) retrieval.

HOW IT WORKS:
    1. On initialization, the SentenceTransformer model is loaded into memory.
       The first time you run this, the model files (~90 MB for all-MiniLM-L6-v2)
       are automatically downloaded from HuggingFace and cached locally.
    2. `embed_texts` converts a batch of text strings into a list of float vectors.
       Batching is more efficient than embedding one text at a time because the
       model can process multiple inputs through the GPU/CPU in parallel.
    3. `embed_query` is a convenience wrapper for embedding a single search query.

WHY THIS EXISTS:
    The vector store needs embeddings to perform similarity search. We wrap the
    model loading and inference in a dedicated class so the rest of the system
    does not need to know about sentence-transformers internals.
"""

import logging
import time

from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Loads a sentence-transformers model and produces embeddings from text.

    The model is loaded once on construction and reused for all subsequent
    embed calls. This avoids the overhead of reloading weights every time
    we need an embedding.

    Attributes:
        model: The loaded SentenceTransformer model instance.
        model_name: Name/path of the model (for logging and debugging).
    """

    def __init__(self) -> None:
        """Load the sentence-transformers model specified in config.

        The model name comes from config.EMBEDDING_MODEL. On the very first run,
        this triggers an automatic download from HuggingFace Hub. Subsequent runs
        use the locally cached copy (typically stored in ~/.cache/huggingface/).

        Logs the time taken to load the model so you can see startup cost.
        """
        self.model_name: str = config.EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", self.model_name)

        start = time.perf_counter()
        self.model = SentenceTransformer(
            self.model_name, cache_folder=str(config.MODEL_CACHE_DIR)
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Embedding model loaded in %.2f seconds (dimension=%d)",
            elapsed,
            config.EMBEDDING_DIMENSION,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        This is the primary method for embedding document chunks during
        ingestion. It processes all texts in one call for efficiency and
        uses batches of 32 and normalizes the resulting vectors.

        Args:
            texts: List of text strings to embed. Each string is typically
                a document chunk of ~1500 characters.

        Returns:
            List of embedding vectors. Each vector is a list of floats with
            length equal to config.EMBEDDING_DIMENSION (384 for MiniLM).

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        total = len(texts)
        logger.info("Embedding %d texts...", total)

        start = time.perf_counter()
        embeddings = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Embedded all %d texts in %.2f seconds (%.1f texts/sec)",
            total,
            elapsed,
            total / elapsed if elapsed > 0 else float("inf"),
        )

        # Convert numpy arrays to plain Python lists for JSON serialization
        # and ChromaDB compatibility.
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a dense vector.

        This is a thin wrapper around embed_texts for the common case of
        embedding one search query. Separated from embed_texts for clarity
        and because query embedding is on the hot path during search.

        Args:
            query: The user's search query text.

        Returns:
            A single embedding vector as a list of floats.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed an empty query.")

        logger.debug("Embedding query: %.80s...", query)
        result = self.model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        return result.tolist()
