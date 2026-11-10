"""
Text chunking with overlap and metadata preservation.

This module is the second step in the RAG ingestion pipeline. It takes the list
of Page objects from pdf_processor and splits them into overlapping chunks of
roughly equal size. Each chunk retains metadata about which chapter, section,
and page range it came from.

Why this exists:
    Embedding models have a limited input window (typically 256-512 tokens) and
    work best on focused passages. Entire pages are too long and too noisy.
    Chunking breaks the text into retrieval-friendly segments while preserving
    enough context through overlaps so that ideas spanning a chunk boundary are
    not lost. The metadata carried on each chunk lets the retriever and LLM
    cite sources precisely.

How it works internally:
    1. All pages are concatenated into a stream of "paragraph blocks," where
       each block remembers which page/chapter/section it came from.
    2. Paragraphs are accumulated into a growing chunk until adding the next
       paragraph would exceed CHUNK_SIZE.
    3. When a chunk is "full," we finalize it, then start the next chunk with
       CHUNK_OVERLAP characters of trailing text from the previous chunk.
    4. We never split mid-sentence: if a paragraph itself exceeds CHUNK_SIZE,
       we split it at the last sentence boundary that fits.
    5. Each chunk gets a deterministic ID based on its text. Repeated text gets
       an occurrence suffix so every stored chunk has a unique ID.
"""

import hashlib
import logging
import re
from dataclasses import dataclass

import config
from ingestion.pdf_processor import Page

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk ready for embedding, with provenance metadata.

    Attributes:
        chunk_id: Text hash with an occurrence suffix for repeated text.
        text: The chunk's text content.
        chapter: Chapter this chunk belongs to, or None.
        section: Section this chunk belongs to, or None.
        page_start: First page number contributing to this chunk (1-based).
        page_end: Last page number contributing to this chunk (1-based).
    """

    chunk_id: str
    text: str
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int


# Sentence-ending punctuation followed by a space or end-of-string.
# Used to find safe split points so we never cut mid-sentence.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """Split pages into overlapping chunks with metadata.

    This is the main entry point for chunking. It processes the list of pages
    sequentially, splits their combined text into chunks of approximately
    config.CHUNK_SIZE characters with config.CHUNK_OVERLAP characters of overlap,
    and tags each chunk with the chapter/section/page metadata.

    Args:
        pages: List of Page objects from pdf_processor.extract_pages().

    Returns:
        A list of Chunk objects ready for embedding.

    Raises:
        ValueError: If pages is empty.
    """
    if not pages:
        raise ValueError("No pages to chunk")

    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP

    logger.info(
        "Chunking %d pages (chunk_size=%d, overlap=%d)",
        len(pages), chunk_size, chunk_overlap,
    )

    # Build paragraph blocks with metadata attached to each one
    blocks = _build_paragraph_blocks(pages)
    logger.info("Split into %d paragraph blocks", len(blocks))

    if not blocks:
        return []

    chunks = _accumulate_chunks(blocks, chunk_size, chunk_overlap)

    # A title alone is not useful retrieval context when its topic has body text.
    topic_pages_with_content = {
        (chunk.chapter, chunk.section, page)
        for chunk in chunks
        if chunk.text.strip() != chunk.section
        for page in range(chunk.page_start, chunk.page_end + 1)
    }
    chunks = [
        chunk for chunk in chunks
        if chunk.text.strip() != chunk.section
        or (chunk.chapter, chunk.section, chunk.page_start) not in topic_pages_with_content
    ]

    occurrences: dict[str, int] = {}
    for chunk in chunks:
        count = occurrences.get(chunk.chunk_id, 0)
        occurrences[chunk.chunk_id] = count + 1
        if count:
            chunk.chunk_id = f"{chunk.chunk_id}_{count}"

    _log_chunk_stats(chunks)
    return chunks


@dataclass
class _ParagraphBlock:
    """Internal: a paragraph with its source metadata.

    Not exported. Used as an intermediate representation between pages and chunks
    so we can track which page/chapter/section each paragraph came from.
    """

    text: str
    page_num: int
    chapter: str | None
    section: str | None


def _build_paragraph_blocks(pages: list[Page]) -> list[_ParagraphBlock]:
    """Assign each paragraph to its most recent topic heading."""
    blocks: list[_ParagraphBlock] = []
    current_chapter: str | None = None
    current_topic: str | None = None

    for page in pages:
        if page.chapter != current_chapter:
            current_chapter = page.chapter
            current_topic = page.chapter
        if page.topic_headings is None and page.section:
            current_topic = page.section

        headings = set(page.topic_headings or [])
        paragraphs = [paragraph.strip() for paragraph in page.text.split("\n\n")]
        index = 0
        while index < len(paragraphs):
            para = paragraphs[index]
            index += 1
            if not para:
                continue
            if len(para) == 1 and para.isalpha():
                continue
            if para in headings:
                parts = [para]
                while index < len(paragraphs) and paragraphs[index] in headings:
                    parts.append(paragraphs[index])
                    index += 1
                current_topic = " ".join(parts)
                para = current_topic
            blocks.append(_ParagraphBlock(
                text=para,
                page_num=page.page_num,
                chapter=page.chapter,
                section=current_topic,
            ))

    return blocks


def _accumulate_chunks(
    blocks: list[_ParagraphBlock],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Accumulate paragraph blocks into chunks respecting size and overlap.

    The algorithm walks through blocks sequentially:
    - It appends each block's text to a growing buffer.
    - When the buffer would exceed chunk_size, the current buffer is finalized
      as a chunk, and a new buffer starts with the overlap tail of the old one.
    - If a single block exceeds chunk_size, it is split at sentence boundaries.

    Args:
        blocks: Paragraph blocks with metadata.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of finalized Chunk objects.
    """
    chunks: list[Chunk] = []

    # Current accumulator state
    buffer_texts: list[str] = []
    buffer_len = 0
    page_start: int = blocks[0].page_num
    page_end: int = blocks[0].page_num
    current_chapter: str | None = blocks[0].chapter
    current_section: str | None = blocks[0].section

    for block in blocks:
        if (
            block.chapter != current_chapter or block.section != current_section
        ) and buffer_texts:
            chunks.append(_finalize_chunk(
                "\n\n".join(buffer_texts),
                current_chapter, current_section,
                page_start, page_end,
            ))
            buffer_texts = []
            buffer_len = 0
            page_start = block.page_num

        current_chapter = block.chapter
        current_section = block.section

        # If this single paragraph exceeds chunk_size, split it at sentence boundaries
        if len(block.text) > chunk_size:
            # Finalize whatever is in the buffer first
            if buffer_texts:
                chunks.append(_finalize_chunk(
                    "\n\n".join(buffer_texts),
                    current_chapter, current_section,
                    page_start, page_end,
                ))
                buffer_texts = []
                buffer_len = 0

            sentence_chunks = _split_long_text(block.text, chunk_size)
            for sc in sentence_chunks:
                chunks.append(_finalize_chunk(
                    sc, current_chapter, current_section,
                    block.page_num, block.page_num,
                ))

            # Start fresh buffer; use overlap from last sentence chunk
            overlap_text = _get_overlap_tail(sentence_chunks[-1], chunk_overlap)
            if overlap_text:
                buffer_texts = [overlap_text]
                buffer_len = len(overlap_text)
            page_start = block.page_num
            page_end = block.page_num
            continue

        # Would adding this block exceed chunk_size?
        separator_len = 2 if buffer_texts else 0  # "\n\n" between paragraphs
        new_len = buffer_len + separator_len + len(block.text)

        if new_len > chunk_size and buffer_texts:
            # Finalize current chunk
            combined = "\n\n".join(buffer_texts)
            chunks.append(_finalize_chunk(
                combined, current_chapter, current_section,
                page_start, page_end,
            ))

            # Start new buffer with overlap from the tail of the previous chunk
            overlap_text = _get_overlap_tail(combined, chunk_overlap)
            if overlap_text:
                buffer_texts = [overlap_text]
                buffer_len = len(overlap_text)
                page_start = page_end
            else:
                buffer_texts = []
                buffer_len = 0
                page_start = block.page_num
            separator_len = 2 if buffer_texts else 0

        # Add block to buffer
        buffer_texts.append(block.text)
        buffer_len += separator_len + len(block.text)
        page_end = block.page_num

    # Finalize any remaining text in the buffer
    if buffer_texts:
        combined = "\n\n".join(buffer_texts)
        if combined.strip():
            chunks.append(_finalize_chunk(
                combined, current_chapter, current_section,
                page_start, page_end,
            ))

    return chunks


def _finalize_chunk(
    text: str,
    chapter: str | None,
    section: str | None,
    page_start: int,
    page_end: int,
) -> Chunk:
    """Create a Chunk with a deterministic hash-based ID.

    The chunk_id is the first 12 hex characters of the SHA-256 hash of the
    text content. This makes IDs deterministic: re-ingesting the same PDF
    produces the same IDs, which is useful for deduplication.

    Args:
        text: The chunk text.
        chapter: Chapter metadata.
        section: Section metadata.
        page_start: First page number.
        page_end: Last page number.

    Returns:
        A fully constructed Chunk object.
    """
    chunk_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chapter=chapter,
        section=section,
        page_start=page_start,
        page_end=page_end,
    )


def _get_overlap_tail(text: str, overlap: int) -> str:
    """Extract the trailing `overlap` characters from text for chunk overlap.

    The overlap ensures that ideas spanning a chunk boundary appear in both
    the ending chunk and the beginning of the next chunk. We try to start the
    overlap at a sentence boundary so it reads naturally.

    Args:
        text: The full chunk text.
        overlap: Number of characters to take from the tail.

    Returns:
        The overlap string, which may be shorter than `overlap` if the text
        itself is shorter.
    """
    if len(text) <= overlap:
        return text

    tail = text[-overlap:]

    # Try to start at a sentence boundary for cleaner overlap
    match = _SENTENCE_BOUNDARY.search(tail)
    if match:
        tail = tail[match.end():]

    return tail.strip()


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a text block that exceeds chunk_size at sentence boundaries.

    When a single paragraph is longer than chunk_size (rare, but possible with
    dense paragraphs), we cannot just accumulate it. Instead, we find the last
    sentence boundary before the size limit and split there.

    Args:
        text: A long text string exceeding chunk_size.
        chunk_size: Maximum characters per chunk.

    Returns:
        A list of text segments. A sentence longer than chunk_size stays whole.
    """
    segments: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        # Find the last sentence boundary within the chunk_size window
        window = remaining[:chunk_size]
        boundaries = list(_SENTENCE_BOUNDARY.finditer(window))

        if boundaries:
            # Split at the last sentence boundary
            split_pos = boundaries[-1].end()
        else:
            # Keep an unusually long sentence intact instead of cutting it.
            next_boundary = _SENTENCE_BOUNDARY.search(remaining, chunk_size)
            if next_boundary is None:
                segments.append(remaining.strip())
                remaining = ""
                break
            split_pos = next_boundary.end()

        segments.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()

    if remaining:
        segments.append(remaining)

    return segments


def _log_chunk_stats(chunks: list[Chunk]) -> None:
    """Log summary statistics about the generated chunks.

    Provides a quick overview of the chunking results for debugging and tuning:
    count, average size, min size, and max size.

    Args:
        chunks: The list of generated chunks.
    """
    if not chunks:
        logger.warning("No chunks were generated")
        return

    sizes = [len(c.text) for c in chunks]
    logger.info(
        "Chunking complete: %d chunks | avg=%d chars | min=%d | max=%d",
        len(chunks),
        sum(sizes) // len(sizes),
        min(sizes),
        max(sizes),
    )
