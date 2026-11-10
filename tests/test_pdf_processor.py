"""Tests for the PDF processor module."""

import os

import pymupdf as fitz

import config
from ingestion.pdf_processor import Page, _detect_chapter, extract_pages


class TestPDFProcessor:
    """Test PDF text extraction.

    These tests require the actual PDF file to be present.
    They are skipped if the file is missing.
    """

    def setup_method(self):
        self.pdf_exists = os.path.exists(config.PDF_PATH)

    def test_extract_returns_pages(self):
        if not self.pdf_exists:
            return  # Skip if PDF not found
        pages = extract_pages(config.PDF_PATH)
        assert len(pages) > 0

    def test_pages_are_page_objects(self):
        if not self.pdf_exists:
            return
        pages = extract_pages(config.PDF_PATH)
        assert all(isinstance(p, Page) for p in pages)

    def test_pages_have_text(self):
        if not self.pdf_exists:
            return
        pages = extract_pages(config.PDF_PATH)
        # At least some pages should have text
        text_pages = [p for p in pages if p.text.strip()]
        assert len(text_pages) > 10
        assert all(page.lines == page.text.splitlines() for page in text_pages)

    def test_page_numbers_preserve_pdf_positions(self):
        if not self.pdf_exists:
            return
        pages = extract_pages(config.PDF_PATH)
        page_numbers = [page.page_num for page in pages]
        assert page_numbers == sorted(set(page_numbers))
        assert all(page_num > 0 for page_num in page_numbers)

    def test_some_chapters_detected(self):
        if not self.pdf_exists:
            return
        pages = extract_pages(config.PDF_PATH)
        chapters = {p.chapter for p in pages if p.chapter}
        assert len(chapters) == 20
        by_page = {page.page_num: page for page in pages}
        assert by_page[50].chapter == "Chapter 3: How to Build Better Habits in 4 Simple Steps"
        assert by_page[131].chapter == (
            "Chapter 13: How to Stop Procrastinating by Using the Two-Minute Rule"
        )
        assert by_page[199].chapter is None


def test_chapter_detection_ignores_numbered_examples():
    assert _detect_chapter(["3. Response: You drink a cup of coffee."]) is None


def test_blank_pdf_page_is_skipped_without_losing_page_number(tmp_path, caplog):
    path = tmp_path / "sample.pdf"
    with fitz.open() as document:
        document.new_page()
        second = document.new_page()
        second.insert_text((72, 72), "A habit starts with a cue.")
        document.save(path)

    pages = extract_pages(path)

    assert len(pages) == 1
    assert pages[0].page_num == 2
    assert pages[0].lines == ["A habit starts with a cue."]
    assert "Page 1: blank/image-only, skipping" in caplog.text
    assert _detect_chapter(["I", "13", "How to Stop Procrastinating", "Two-Minute Rule"]) == (
        "Chapter 13: How to Stop Procrastinating Two-Minute Rule"
    )


def test_extract_preserves_topic_headings_within_a_page(tmp_path):
    path = tmp_path / "topics.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "FIRST TOPIC", fontsize=14.4)
        page.insert_text((72, 105), "First topic details.", fontsize=14.4)
        page.insert_text((72, 155), "SECOND TOPIC", fontsize=14.4)
        page.insert_text((72, 188), "Second topic details.", fontsize=14.4)
        page.insert_text((72, 230), "TABLE LABEL", fontsize=10.8)
        document.save(path)

    pages = extract_pages(path)

    assert pages[0].topic_headings == ["FIRST TOPIC", "SECOND TOPIC"]
    assert "FIRST TOPIC\n\nFirst topic details." in pages[0].text
    assert "SECOND TOPIC\n\nSecond topic details." in pages[0].text
    assert "TABLE LABEL" in pages[0].text
