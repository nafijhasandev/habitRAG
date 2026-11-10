"""Normalize generated citations against the passages actually retrieved."""

import re
from typing import Any

_CITATION = re.compile(
    r"[\[(]?\bChapter\s+(\d+)(?:\s*:[^,\n)\]]*)?,\s*"
    r"(?:p\.?|pages?)\s*(\d+)(?:\s*-\s*\d+)?[)\]]?",
    re.IGNORECASE,
)
_CHAPTER_NUMBER = re.compile(r"\bChapter\s+(\d+)\b", re.IGNORECASE)
_CHAPTER_ONLY = re.compile(r"\(Chapter\s+(\d+)\)", re.IGNORECASE)


def _source_citation(source: dict[str, Any]) -> tuple[str, int, int] | None:
    metadata = source.get("metadata") or {}
    chapter = str(metadata.get("chapter") or "").strip()
    match = _CHAPTER_NUMBER.search(chapter)
    if chapter.isdecimal():
        chapter_number = chapter
    elif match:
        chapter_number = match.group(1)
    else:
        return None

    try:
        page_start = int(metadata["page_start"])
        page_end = int(metadata["page_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if page_start < 1 or page_end < page_start:
        return None
    return chapter_number, page_start, page_end


def format_cited_answer(answer: str, sources: list[dict[str, Any]]) -> str:
    """Keep only citations supported by retrieved chapter and page metadata."""
    allowed = [citation for source in sources if (citation := _source_citation(source))]
    valid_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal valid_count
        chapter, page = match.group(1), int(match.group(2))
        if any(number == chapter and start <= page <= end for number, start, end in allowed):
            valid_count += 1
            return f"(Chapter {chapter}, p.{page})"
        return ""

    cleaned = _CITATION.sub(replace, answer)

    def replace_chapter_only(match: re.Match[str]) -> str:
        nonlocal valid_count
        chapter = match.group(1)
        for number, start, _ in allowed:
            if number == chapter:
                valid_count += 1
                return f"(Chapter {chapter}, p.{start})"
        return ""

    cleaned = _CHAPTER_ONLY.sub(replace_chapter_only, cleaned).strip()
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.!?,;:])", r"\1", cleaned)
    if valid_count or not allowed or not cleaned:
        return cleaned

    chapter, page, _ = allowed[0]
    citation = f" (Chapter {chapter}, p.{page})"
    if cleaned[-1] in ".!?":
        return f"{cleaned[:-1]}{citation}{cleaned[-1]}"
    return cleaned + citation
