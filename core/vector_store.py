"""
ChromaDB vector store for dense retrieval.

This module wraps ChromaDB to store document chunk embeddings and retrieve
the most semantically similar chunks for a given query. ChromaDB is an
open-source embedding database that handles the nearest-neighbor search
efficiently using approximate algorithms (HNSW).

HOW IT WORKS:
    1. During ingestion, each chunk's text and its embedding vector are stored
       in a ChromaDB collection along with metadata (chapter, section, pages).
    2. At query time, the query embedding is compared against all stored
       embeddings using cosine similarity.
    3. ChromaDB returns the closest vectors - these are the most semantically
       relevant chunks.

WHY THIS EXISTS:
    Dense retrieval using embeddings captures semantic meaning that keyword
    search misses. For example, a query about "building good routines" will
    match chunks about "habit formation" even though the exact words differ.
    ChromaDB provides persistent storage so we do not need to re-embed all
    chunks every time the application restarts.

PERSISTENCE:
    ChromaDB stores data on disk at config.CHROMA_DIR. The collection persists
    across restarts. Use `collection_exists()` to check if data is already
    ingested, and `reset()` to wipe and re-ingest.
"""

import logging

import chromadb

import config

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages a ChromaDB collection for storing and querying chunk embeddings.

    This class provides a clean interface over ChromaDB's API, handling ID
    generation, metadata formatting, and result normalization so the rest
    of the system works with plain Python dicts.

    Attributes:
        client: The ChromaDB persistent client instance.
        collection: The ChromaDB collection holding our embeddings.
    """

    def __init__(self) -> None:
        """Initialize ChromaDB client and get or create the collection.

        The client is configured with persistent storage at config.CHROMA_DIR
        so data survives restarts. The collection name comes from
        config.CHROMA_COLLECTION.
        """
        logger.info("Initializing ChromaDB at: %s", config.CHROMA_DIR)

        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
        )

        logger.info(
            "ChromaDB collection '%s' ready (%d chunks stored)",
            config.CHROMA_COLLECTION,
            self.collection.count(),
        )

    def configuration_space(self) -> str:
        """Return the collection distance metric."""
        return self.collection.configuration["hnsw"]["space"]

    def require_cosine(self) -> None:
        """Reject an old index that cannot return cosine scores."""
        if self.configuration_space() != "cosine":
            raise RuntimeError("Vector index uses an old metric. Rebuild with ingest --force.")

    def add_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks with their embeddings and metadata in ChromaDB.

        Each chunk gets a string ID (required by ChromaDB), its text stored
        as a document, its embedding vector, and structured metadata for
        filtering and display.

        Args:
            chunks: List of Chunk dataclass instances from ingestion.chunker.
                Each chunk has attributes: chunk_id, text, chapter, section,
                page_start, page_end.
            embeddings: Parallel list of embedding vectors, one per chunk.
                Must be the same length as chunks.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if not chunks:
            logger.warning("No chunks to add.")
            return

        # ChromaDB requires string IDs. We use the chunk's integer ID
        # converted to a string with a prefix for readability.
        ids = [f"chunk_{chunk.chunk_id}" for chunk in chunks]

        documents = [chunk.text for chunk in chunks]

        # Metadata must be flat dicts with string/int/float values.
        # ChromaDB does not support nested structures.
        metadatas = [
            {
                "chapter": chunk.chapter or "",
                "section": chunk.section or "",
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
            for chunk in chunks
        ]

        logger.info("Adding %d chunks to ChromaDB...", len(chunks))

        # ChromaDB has a batch size limit, so we add in batches to avoid
        # memory issues with very large document sets.
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
            logger.info("  Added batch %d-%d / %d", i, end, len(chunks))

        logger.info(
            "Successfully stored %d chunks. Total in collection: %d",
            len(chunks),
            self.collection.count(),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Find the most semantically similar chunks to a query embedding.

        Uses ChromaDB's built-in approximate nearest neighbor search. The
        collection uses cosine distance.

        Args:
            query_embedding: The query's embedding vector (from EmbeddingEngine).
            top_k: Number of results to return. Defaults to 10.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of dicts, each with keys:
                - chunk_id (str): The chunk identifier.
                - text (str): The chunk's full text.
                - metadata (dict): Chapter, section, and page info.
                - score (float): Cosine similarity (1 - cosine distance).
        """
        self.require_cosine()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        # ChromaDB returns nested lists because it supports batched queries.
        # We always query one embedding at a time, so we index [0].
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        output = []
        for i, chunk_id in enumerate(ids):
            score = 1.0 - distances[i]
            output.append({
                "chunk_id": chunk_id,
                "text": documents[i],
                "metadata": metadatas[i],
                "score": score,
            })

        logger.debug("Dense search returned %d results", len(output))
        return output

    def count(self) -> int:
        """Return the number of chunks currently stored in the collection.

        Returns:
            Integer count of stored chunks.
        """
        return self.collection.count()

    def collection_exists(self) -> bool:
        """Check whether the collection already contains data.

        This is used to skip re-ingestion if the PDF has already been
        processed and stored.

        Returns:
            True if the collection has at least one stored chunk.
        """
        return self.collection.count() > 0

    def reset(self) -> None:
        """Delete and recreate the collection, removing all stored data.

        Use this when you want to re-ingest from scratch (e.g., after
        changing chunking parameters or fixing a bug in ingestion).
        """
        logger.warning(
            "Resetting collection '%s' - deleting all %d chunks",
            config.CHROMA_COLLECTION,
            self.collection.count(),
        )
        self.client.delete_collection(config.CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
        )
        logger.info("Collection '%s' recreated (empty).", config.CHROMA_COLLECTION)
