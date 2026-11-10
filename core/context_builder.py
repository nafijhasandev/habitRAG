"""
Context builder for assembling retrieved chunks into an LLM prompt.

WHY THIS EXISTS:
After retrieval and reranking, we have a set of highly relevant text chunks.
But we cannot simply concatenate them and send them to the LLM. Several
problems need to be solved first:

1. ORDERING: Chunks may have been retrieved in relevance-score order, but
   for the LLM to reason well about the content, presenting them in their
   original book order (by page number) produces more coherent context. The
   LLM can follow the author's logical flow rather than jumping around.

2. DEDUPLICATION: The hybrid retrieval pipeline (dense + sparse + expansion)
   often retrieves overlapping or near-identical chunks. Sending duplicates
   wastes precious context window tokens and can cause the LLM to
   over-emphasize repeated content. We remove chunks that overlap >80% with
   an already-included chunk.

3. FORMATTING: Each chunk needs a clear header (chapter, page range) so the
   LLM knows where the information came from and can cite sources in its
   answer.

4. TRUNCATION: The LLM has a finite context window, and we must leave room
   for the system prompt and the generated answer. If total chunk text exceeds
   config.MAX_CONTEXT_LENGTH, we keep the highest-scored chunks and drop the
   rest.

5. PROMPT ASSEMBLY: The final prompt combines a query-type-specific system
   prompt (from query_classifier) with the formatted context and the user's
   question, structured as a message list for the chat completions API.
"""

import logging
from typing import Any

import config
from core.query_classifier import get_prompt_template

logger = logging.getLogger(__name__)


def _compute_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute a simple character-level overlap ratio between two texts.

    Uses the ratio of shared characters (as sets of character bigrams) to
    the total unique bigrams. This is a fast approximation of text similarity
    that works well for detecting near-duplicate chunks.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        A float between 0.0 (no overlap) and 1.0 (identical bigram sets).
    """
    return _bigram_overlap(_text_bigrams(text_a), _text_bigrams(text_b))


def _text_bigrams(value: str) -> set[str]:
    return {value[i : i + 2] for i in range(len(value) - 1)}


def _bigram_overlap(bigrams_a: set[str], bigrams_b: set[str]) -> float:
    if not bigrams_a or not bigrams_b:
        return 0.0

    intersection = bigrams_a & bigrams_b
    union = bigrams_a | bigrams_b

    return len(intersection) / len(union)


def _deduplicate_chunks(
    chunks: list[dict[str, Any]], threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Remove near-duplicate chunks based on text overlap.

    Iterates through chunks in order, keeping a chunk only if it does not
    overlap more than `threshold` with any previously kept chunk.

    Args:
        chunks: List of chunk dicts, each with a 'text' key.
        threshold: Maximum allowed overlap ratio (0.0 to 1.0). Chunks
            exceeding this threshold relative to any kept chunk are dropped.

    Returns:
        The filtered list of chunks with near-duplicates removed.
    """
    if not chunks:
        return []

    kept: list[tuple[dict[str, Any], set[str]]] = []
    for chunk in chunks:
        bigrams = _text_bigrams(chunk["text"])
        is_duplicate = False
        for kept_chunk, kept_bigrams in kept:
            overlap = _bigram_overlap(bigrams, kept_bigrams)
            if overlap > threshold:
                is_duplicate = True
                logger.debug(
                    "Dropping near-duplicate chunk %s (%.1f%% overlap with %s).",
                    chunk.get("chunk_id", "?"),
                    overlap * 100,
                    kept_chunk.get("chunk_id", "?"),
                )
                break
        if not is_duplicate:
            kept.append((chunk, bigrams))

    if len(kept) < len(chunks):
        logger.info(
            "Deduplication removed %d/%d chunks.", len(chunks) - len(kept), len(chunks)
        )

    return [chunk for chunk, _ in kept]


def format_source_location(metadata: dict[str, Any]) -> str:
    """Format chapter and PDF page metadata for a citation."""
    chapter = str(metadata.get("chapter") or "").strip()
    if chapter.isdecimal():
        chapter = f"Chapter {chapter}"
    pages = f"Pages {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}"
    return f"{chapter}, {pages}" if chapter else pages


def _format_chunk(chunk: dict[str, Any]) -> str:
    """Format a single chunk with a descriptive header.

    Args:
        chunk: A chunk dict with 'text' and 'metadata' keys. Metadata
            should contain 'chapter', 'page_start', and 'page_end'.

    Returns:
        A formatted string like:
        [Chapter 5, Pages 72-75]
        <chunk text>
    """
    metadata = chunk.get("metadata", {})
    header = f"[{format_source_location(metadata)}]"
    topic = str(metadata.get("section") or "").strip()
    topic_line = f"Topic: {topic}\n" if topic else ""
    return f"{header}\n{topic_line}{chunk['text']}"


def build_context(
    chunks: list[dict[str, Any]], max_length: int | None = None
) -> str:
    """Build a formatted context string from retrieved chunks.

    This function prepares chunks for inclusion in an LLM prompt by:
    1. Sorting by book position (page_start) for reading coherence
    2. Removing near-duplicate chunks (>80% overlap)
    3. Formatting each chunk with a chapter/page header
    4. Truncating to fit within max_length, keeping highest-scored chunks

    Args:
        chunks: Retrieved and reranked chunk dicts. Each must have 'text'
            and 'metadata' keys. May optionally have 'rerank_score'.
        max_length: Maximum total character length of the context string.
            Defaults to config.MAX_CONTEXT_LENGTH.

    Returns:
        A formatted context string ready for inclusion in an LLM prompt.
        Returns an empty string if no chunks are provided.
    """
    if max_length is None:
        max_length = config.MAX_CONTEXT_LENGTH

    if not chunks:
        logger.warning("No chunks provided to build_context.")
        return ""

    # Step 1: Sort by book position for coherent reading order
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.get("metadata", {}).get("page_start", 0),
    )

    # Step 2: Remove near-duplicates
    deduped_chunks = _deduplicate_chunks(sorted_chunks)

    # Step 3: Format each chunk
    formatted_parts = [(chunk, _format_chunk(chunk)) for chunk in deduped_chunks]

    # Step 4: Truncate if total length exceeds max_length.
    # Strategy: if we overflow, prioritize chunks with higher rerank scores
    total_text = "\n\n".join(part for _, part in formatted_parts)

    if len(total_text) <= max_length:
        logger.info(
            "Context built from %d chunks (%d chars).",
            len(formatted_parts),
            len(total_text),
        )
        return total_text

    # Need to truncate: sort by rerank score (descending) to keep the best
    scored_parts = sorted(
        formatted_parts,
        key=lambda item: item[0].get("rerank_score", 0.0),
        reverse=True,
    )

    kept_parts: list[tuple[dict[str, Any], str]] = []
    current_length = 0
    separator_length = 2  # "\n\n" between chunks

    for chunk, formatted in scored_parts:
        added_length = len(formatted) + (separator_length if kept_parts else 0)
        if current_length + added_length > max_length:
            break
        kept_parts.append((chunk, formatted))
        current_length += added_length

    # Re-sort kept chunks by book position for reading coherence
    kept_parts.sort(
        key=lambda item: item[0].get("metadata", {}).get("page_start", 0)
    )

    result = "\n\n".join(part for _, part in kept_parts)
    logger.info(
        "Context truncated: kept %d/%d chunks (%d chars, max %d).",
        len(kept_parts),
        len(formatted_parts),
        len(result),
        max_length,
    )
    return result


def build_prompt(
    query: str, context: str, query_type: str
) -> list[dict[str, str]]:
    """Assemble the final message list for the chat completions API.

    Combines:
    1. A system prompt tailored to the query type (from query_classifier)
    2. A user message containing the context block and the question

    The resulting message list follows the chat completions API format:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Args:
        query: The user's original question.
        context: The formatted context string from build_context().
        query_type: One of "factual", "analytical", "comparison", "summary".

    Returns:
        A list of message dicts ready for the chat completions API.
    """
    system_prompt = (
        get_prompt_template(query_type)
        + " Answer only from the provided context. If it does not contain the"
        " answer, say that the information is not in the book."
        + " Cite each substantive point with a chapter and page from the context."
        " Do not invent citations."
    )

    user_message = (
        f"Context from the book:\n"
        f"---\n"
        f"{context}\n"
        f"---\n\n"
        f"Question: {query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    logger.debug(
        "Built prompt with %d chars system, %d chars user message.",
        len(system_prompt),
        len(user_message),
    )

    return messages
