"""
BM25 sparse retrieval index.

This module builds a BM25 (Best Matching 25) keyword index over document
chunks. BM25 is a classic information retrieval algorithm that ranks documents
based on term frequency and inverse document frequency - essentially, it
finds documents that contain the query's keywords, weighted by how rare and
prominent those keywords are.

HOW IT WORKS:
    1. Each chunk's text is tokenized: lowercased, split on word boundaries,
       and filtered to remove common stopwords (defined in config.BM25_STOPWORDS).
    2. The BM25Okapi algorithm builds a statistical model of term frequencies
       across the entire corpus.
    3. At query time, the query is tokenized the same way, and BM25 scores
       each document based on how well its terms match the query terms.

WHY THIS EXISTS (BM25 vs. Dense Retrieval):
    Dense retrieval (embeddings) is great at finding semantically similar text,
    but it can miss exact keyword matches. For example, if a user searches for
    "cue craving response reward", dense retrieval might return chunks about
    general habit concepts, while BM25 will precisely find chunks containing
    those exact terms. By combining both approaches (hybrid retrieval), we
    get the best of both worlds.

PERSISTENCE:
    The BM25 index can be saved to disk via pickle and reloaded, avoiding the
    need to re-tokenize all chunks on every startup. The save path is
    config.BM25_INDEX_PATH.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Self

from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Applies a simple but effective tokenization pipeline:
    1. Lowercase the text (case-insensitive matching).
    2. Split on non-alphanumeric characters (whitespace, punctuation).
    3. Remove stopwords that add noise to keyword search.
    4. Filter out empty strings and single-character tokens.

    This function is used both at index time and query time to ensure
    consistent tokenization. If you tokenize differently at query time,
    the BM25 scores will be meaningless.

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of cleaned, lowercased tokens with stopwords removed.
    """
    # Split on anything that is not a letter or digit
    tokens = re.split(r"[^a-z0-9]+", text.lower())

    # Filter out stopwords and very short tokens
    return [
        token
        for token in tokens
        if token and len(token) > 1 and token not in config.BM25_STOPWORDS
    ]


class BM25Index:
    """BM25 sparse retrieval index over document chunks.

    Stores the BM25 model alongside chunk references so that search results
    can return full chunk data (text, metadata) rather than just scores.

    Attributes:
        index: The BM25Okapi model after building.
        chunks: List of Chunk dataclass instances, stored so we can return
            full chunk data from search results.
        tokenized_corpus: The tokenized version of each chunk's text,
            kept for debugging and index statistics.
    """

    def __init__(self) -> None:
        """Create an empty BM25Index. Call build() to populate it."""
        self.index: BM25Okapi | None = None
        self.chunks: list = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list) -> None:
        """Tokenize all chunk texts and build the BM25 index.

        This is the main ingestion method. It processes each chunk's text
        through the tokenizer and feeds the tokenized corpus to BM25Okapi.

        Args:
            chunks: List of Chunk dataclass instances from ingestion.chunker.
                Each chunk must have `text`, `chunk_id`, `chapter`, `section`,
                `page_start`, and `page_end` attributes.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks list.")

        logger.info("Building BM25 index from %d chunks...", len(chunks))

        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]

        self.index = BM25Okapi(self.tokenized_corpus)

        # Log useful statistics about the index
        vocab = set()
        total_tokens = 0
        for doc_tokens in self.tokenized_corpus:
            vocab.update(doc_tokens)
            total_tokens += len(doc_tokens)

        avg_doc_length = total_tokens / len(self.tokenized_corpus)

        logger.info(
            "BM25 index built: %d documents, %d unique terms, "
            "%.1f avg tokens/document",
            len(self.tokenized_corpus),
            len(vocab),
            avg_doc_length,
        )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the BM25 index with a text query.

        Tokenizes the query using the same pipeline as indexing, then
        scores all documents and returns the top-k highest-scoring ones.

        Args:
            query: The user's search query as raw text.
            top_k: Number of top results to return.

        Returns:
            List of dicts sorted by BM25 score (descending), each with:
                - chunk_id (str): Chunk identifier, prefixed for consistency
                  with the vector store.
                - text (str): The chunk's full text.
                - metadata (dict): Chapter, section, and page info.
                - score (float): BM25 relevance score.

        Raises:
            RuntimeError: If search is called before build().
        """
        if self.index is None:
            raise RuntimeError(
                "BM25 index not built. Call build() before search()."
            )

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization: '%s'", query)
            return []

        scores = self.index.get_scores(query_tokens)

        # Get indices of top-k scores, sorted descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank_idx in ranked_indices:
            score = float(scores[rank_idx])
            # Skip zero-score documents (no term overlap at all)
            if score <= 0:
                continue

            chunk = self.chunks[rank_idx]
            results.append({
                "chunk_id": f"chunk_{chunk.chunk_id}",
                "text": chunk.text,
                "metadata": {
                    "chapter": chunk.chapter or "",
                    "section": chunk.section or "",
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
                "score": score,
            })

        logger.debug(
            "BM25 search for '%s' returned %d results (top score: %.3f)",
            query[:80],
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results

    def save(self, path: Path | str) -> None:
        """Persist the BM25 index and chunk references to disk.

        Uses pickle serialization. The saved file includes the BM25Okapi
        model, the chunk list, and the tokenized corpus so the index can
        be fully restored without re-processing.

        Args:
            path: File path to save the pickled index.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError("Cannot save: BM25 index not built yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "index": self.index,
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("BM25 index saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a previously saved BM25 index from disk.

        This avoids re-tokenizing the entire corpus on every startup.

        Args:
            path: File path to the pickled index.

        Returns:
            A fully populated BM25Index instance ready for search.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.index = data["index"]
        instance.chunks = data["chunks"]
        instance.tokenized_corpus = data["tokenized_corpus"]

        logger.info(
            "BM25 index loaded from %s (%d documents)",
            path,
            len(instance.chunks),
        )
        return instance
