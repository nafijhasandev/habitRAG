"""
PDF text extraction with chapter and section detection.

This module is the first step in the RAG ingestion pipeline. It reads a PDF file
page by page, extracts the text content, and identifies structural elements like
chapters and sections. The output is a list of Page objects that carry both the
raw text and its structural metadata.

Why this exists:
    Raw PDFs are unstructured binary blobs. Before we can chunk text, embed it,
    and search over it, we need to extract clean plaintext and tag it with
    structural metadata (chapter, section). That metadata flows through the
    entire pipeline and eventually helps the retriever and LLM understand
    *where* a piece of information comes from in the book.

How it works internally:
    1. PyMuPDF (fitz) opens the PDF and iterates page by page.
    2. For each page, raw text is extracted and cleaned (whitespace normalization,
       hyphenation repair, header/footer removal).
    3. Large heading text identifies chapter numbers and titles, while regex
       patterns identify section headings.
       When a heading is found, all subsequent pages inherit that chapter/section
       until a new heading appears.
    4. The result is a flat list of Page dataclasses ready for the chunker.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

import config

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """A single page of extracted text with its structural metadata.

    Attributes:
        page_num: 1-based page number from the PDF.
        text: Cleaned text content of the page.
        chapter: Detected chapter title (e.g. "Chapter 1: ..."), or None if
                 the page falls before any chapter heading (e.g. preface).
        section: Detected section heading within the chapter, or None.
        lines: Cleaned text lines in reading order.
    """

    page_num: int
    text: str
    chapter: str | None
    section: str | None
    lines: list[str] = field(default_factory=list)
    topic_headings: list[str] | None = None


# ---------------------------------------------------------------------------
# Chapter / section detection patterns
# ---------------------------------------------------------------------------

# Section headings: lines in ALL CAPS or Title Case that are short (likely a heading).
# We require at least 4 chars to avoid matching stray abbreviations.
_SECTION_PATTERN = re.compile(
    r"^\s{0,4}([A-Z][A-Za-z\s]{3,60})$",
    re.MULTILINE,
)

# Common header/footer fragments to strip (page numbers, running headers).
_HEADER_FOOTER_PATTERN = re.compile(
    r"^\s*\d{1,3}\s*$"          # standalone page numbers
    r"|^\s*atomic\s+habits\s*$"  # running book title header
    r"|^\s*james\s+clear\s*$",   # running author header
    re.IGNORECASE | re.MULTILINE,
)

_PART_HEADINGS = {
    "FUNDAMENTALS", "THE 1ST LAW", "THE 2ND LAW", "THE 3RD LAW",
    "THE 4TH LAW", "ADVANCED TACTICS",
}


def extract_pages(pdf_path: Path | None = None) -> list[Page]:
    """Extract text from every page of the PDF and detect structure.

    This is the main entry point for PDF processing. It opens the PDF, iterates
    through each page, cleans the text, and tracks chapter/section headings as
    they appear. Each returned Page carries the most recently detected chapter
    and section so downstream consumers always know the structural context.

    Args:
        pdf_path: Path to the PDF file. Defaults to config.PDF_PATH.

    Returns:
        A list of Page objects, one per non-empty page in the PDF.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    pdf_path = pdf_path or config.PDF_PATH

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Opening PDF: %s", pdf_path.name)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    total_pages = len(doc)
    logger.info("PDF has %d pages", total_pages)

    pages: list[Page] = []
    current_chapter: str | None = None
    current_section: str | None = None

    for page_idx in range(total_pages):
        raw_text, topic_headings = _page_text_and_topic_headings(doc[page_idx])

        # Skip blank or image-only pages
        if not raw_text or raw_text.isspace():
            logger.warning("Page %d: blank/image-only, skipping", page_idx + 1)
            continue

        cleaned = _clean_text(raw_text)

        # Skip pages that became empty after cleaning (e.g. only header/footer)
        if not cleaned.strip():
            logger.warning("Page %d: empty after cleaning, skipping", page_idx + 1)
            continue

        # Detect chapter and section headings on this page
        heading_lines = _large_heading_lines(doc[page_idx])
        chapter_match = _detect_chapter(heading_lines)
        section_match = _detect_section(cleaned)

        if "Conclusion" in heading_lines[:2] or any(
            heading in _PART_HEADINGS for heading in heading_lines[:2]
        ):
            current_chapter = None
            current_section = None
        elif chapter_match:
            current_chapter = chapter_match
            # A new chapter resets the section context
            current_section = None
            logger.debug("Page %d: new chapter -> %s", page_idx + 1, current_chapter)

        if section_match:
            current_section = section_match
            logger.debug("Page %d: new section -> %s", page_idx + 1, current_section)

        pages.append(Page(
            page_num=page_idx + 1,
            text=cleaned,
            chapter=current_chapter,
            section=current_section,
            lines=cleaned.splitlines(),
            topic_headings=topic_headings,
        ))

        if (page_idx + 1) % 50 == 0:
            logger.info("Processed %d / %d pages", page_idx + 1, total_pages)

    doc.close()
    logger.info(
        "Extraction complete: %d non-empty pages from %d total",
        len(pages),
        total_pages,
    )
    return pages


def _page_text_and_topic_headings(page: fitz.Page) -> tuple[str, list[str]]:
    """Keep PDF text blocks separate so topic headings remain identifiable."""
    blocks = []
    headings = []
    for block in page.get_text("dict")["blocks"]:
        lines = block.get("lines", [])
        if not lines:
            continue
        raw = "\n".join(
            "".join(span["text"] for span in line["spans"]) for line in lines
        ).strip()
        if not raw:
            continue
        blocks.append(raw)
        heading = _topic_heading(raw, lines)
        if heading:
            headings.append(heading)
    return "\n\n".join(blocks), headings


def _topic_heading(text: str, lines: list[dict]) -> str | None:
    """Recognize standalone topic headings using the PDF's font sizes."""
    heading = re.sub(r"\s+", " ", text).strip()
    spans = [span for line in lines for span in line["spans"] if span["text"].strip()]
    if not spans or len(heading) > 110 or len(heading.split()) > 12:
        return None
    if len(heading) < 4 or heading[-1] in ".!?;":
        return None
    largest_font = max(span["size"] for span in spans)
    if heading.isupper() and largest_font >= 14:
        return heading
    if largest_font >= 16 and heading[0].isupper():
        return heading
    return None


def _clean_text(raw: str) -> str:
    """Clean raw PDF-extracted text.

    Applies several heuristic fixes for common PDF extraction artifacts:
    - Removes detected headers and footers (running titles, standalone page numbers).
    - Repairs words that were hyphenated at line breaks (e.g. "impor-\\ntant" -> "important").
    - Normalizes runs of whitespace into single spaces while preserving paragraph
      breaks (double newlines).

    Args:
        raw: Raw text string from PyMuPDF.

    Returns:
        Cleaned text string.
    """
    # Strip common headers/footers
    text = _HEADER_FOOTER_PATTERN.sub("", raw)

    # Fix hyphenation at line breaks: "impor-\ntant" -> "important"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse single newlines into spaces (preserving paragraph breaks).
    # First, protect paragraph breaks (2+ newlines) by replacing with a sentinel.
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace remaining single newlines with spaces
    lines = text.split("\n\n")
    cleaned_paragraphs = []
    for para in lines:
        # Within a paragraph, join wrapped lines
        joined = para.replace("\n", " ")
        # Collapse multiple spaces
        joined = re.sub(r" {2,}", " ", joined)
        cleaned_paragraphs.append(joined.strip())

    text = "\n\n".join(p for p in cleaned_paragraphs if p)
    return text


def _large_heading_lines(page: fitz.Page) -> list[str]:
    """Return the page's large heading lines in reading order."""
    headings = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(
                span["text"] for span in line["spans"] if span["size"] >= 20
            ).strip()
            if text:
                headings.append(text)
    return headings


def _detect_chapter(headings: list[str]) -> str | None:
    """Read a numbered chapter title from the first large heading lines."""
    for index in (0, 1):
        if index >= len(headings) or not re.fullmatch(r"[1-9]\d?", headings[index]):
            continue
        title = " ".join(headings[index + 1 : index + 3])
        if title:
            return f"Chapter {headings[index]}: {title}"
    return None


def _detect_section(text: str) -> str | None:
    """Detect a section heading in the page text.

    Looks for short lines in Title Case or ALL CAPS that likely represent
    section headings within a chapter. Filters out common false positives
    like single short words.

    Args:
        text: Cleaned page text.

    Returns:
        The section heading string, or None if no section heading is found.
    """
    # Only consider lines near the top of the page (first ~500 chars)
    # to avoid matching capitalized phrases in body text
    top_text = text[:500]

    matches = _SECTION_PATTERN.findall(top_text)
    if not matches:
        return None

    for candidate in matches:
        candidate = candidate.strip()
        # Skip if it looks like a normal sentence (too many lowercase words)
        words = candidate.split()
        if len(words) < 2:
            continue
        # Check if it is Title Case or ALL CAPS
        is_title_case = all(w[0].isupper() for w in words if w.isalpha())
        is_all_caps = candidate == candidate.upper()
        if is_title_case or is_all_caps:
            return candidate

    return None
