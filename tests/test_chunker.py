"""Tests for the chunker module."""

import config
from ingestion.chunker import Chunk, _split_long_text, chunk_pages
from ingestion.pdf_processor import Page, extract_pages


def _make_pages(texts: list[str], chapter: str = "1") -> list[Page]:
    """Helper to create Page objects from plain text strings."""
    return [
        Page(page_num=i + 1, text=t, chapter=chapter, section=None)
        for i, t in enumerate(texts)
    ]


class TestChunker:
    """Test text chunking behavior."""

    def test_basic_chunking(self):
        """A short text should produce a single chunk."""
        pages = _make_pages(["This is a short paragraph."])
        chunks = chunk_pages(pages)
        assert len(chunks) >= 1
        assert "short paragraph" in chunks[0].text

    def test_chunk_has_metadata(self):
        """Each chunk should carry chapter and page metadata."""
        pages = _make_pages(["Some text about habits."], chapter="3")
        chunks = chunk_pages(pages)
        assert chunks[0].chapter == "3"
        assert chunks[0].page_start == 1

    def test_chapter_boundary_keeps_citations_separate(self):
        pages = [
            Page(1, "First chapter text.", "Chapter 1: First", None),
            Page(2, "Second chapter text.", "Chapter 2: Second", None),
        ]

        chunks = chunk_pages(pages)

        assert [(chunk.chapter, chunk.page_start, chunk.page_end) for chunk in chunks] == [
            ("Chapter 1: First", 1, 1),
            ("Chapter 2: Second", 2, 2),
        ]

    def test_section_boundary_keeps_citations_separate(self):
        pages = [
            Page(1, "First section passage.", "Chapter 1", "Identity"),
            Page(2, "Second section passage.", "Chapter 1", "Environment"),
        ]

        chunks = chunk_pages(pages)

        assert [(chunk.section, chunk.page_start, chunk.page_end) for chunk in chunks] == [
            ("Identity", 1, 1),
            ("Environment", 2, 2),
        ]

    def test_topic_headings_split_a_page_and_continue_across_pages(self):
        pages = [
            Page(
                1,
                "FIRST TOPIC\n\nFirst topic details.\n\nSECOND TOPIC\n\nSecond topic starts.",
                "Chapter 1",
                None,
                topic_headings=["FIRST TOPIC", "SECOND TOPIC"],
            ),
            Page(2, "Second topic continues.", "Chapter 1", None, topic_headings=[]),
        ]

        chunks = chunk_pages(pages)

        assert [chunk.section for chunk in chunks] == ["FIRST TOPIC", "SECOND TOPIC"]
        assert "Second topic" not in chunks[0].text
        assert chunks[1].page_start == 1
        assert chunks[1].page_end == 2

    def test_long_topic_keeps_one_label_without_title_only_chunk(self):
        body = "A habit starts with a cue and a response. " * 100
        pages = [Page(
            1, f"LONG TOPIC\n\n{body}", "Chapter 1", None,
            topic_headings=["LONG TOPIC"],
        )]

        chunks = chunk_pages(pages)

        assert len(chunks) > 1
        assert all(chunk.section == "LONG TOPIC" for chunk in chunks)
        assert all(chunk.text != "LONG TOPIC" for chunk in chunks)
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    def test_heading_only_page_keeps_its_topic_chunk(self):
        pages = [
            Page(1, "LONG TOPIC", "Chapter 1", None, topic_headings=["LONG TOPIC"]),
            Page(2, "The topic continues on this page. " * 40, "Chapter 1", None,
                 topic_headings=[]),
        ]

        chunks = chunk_pages(pages)

        assert any(chunk.page_start == 1 and chunk.section == "LONG TOPIC" for chunk in chunks)
        assert any(chunk.page_start == 2 and chunk.section == "LONG TOPIC" for chunk in chunks)

    def test_long_sentence_is_not_split_mid_sentence(self):
        sentence = "A habit becomes automatic after many repetitions of the same action."
        text = sentence + " A second sentence follows."

        parts = _split_long_text(text, chunk_size=35)

        assert parts == [sentence, "A second sentence follows."]

    def test_single_oversized_sentence_stays_whole(self):
        sentence = "A habit grows through repetition " * 20

        assert _split_long_text(sentence, chunk_size=35) == [sentence.strip()]

    def test_standalone_drop_cap_is_not_indexed(self):
        pages = [Page(17, "T\n\nThe chapter title and passage.", "Chapter 1: First", None)]

        chunks = chunk_pages(pages)

        assert len(chunks) == 1
        assert chunks[0].text == "The chapter title and passage."

    def test_page_range_tracks_overlap_at_size_boundary(self, monkeypatch):
        monkeypatch.setattr(config, "CHUNK_SIZE", 40)
        monkeypatch.setattr(config, "CHUNK_OVERLAP", 5)
        pages = [
            Page(1, "A first page sentence.", "Chapter 1: First", None),
            Page(2, "A second page sentence.", "Chapter 1: First", None),
        ]

        chunks = chunk_pages(pages)

        assert chunks[0].page_start == chunks[0].page_end == 1
        assert chunks[1].page_start == 1
        assert chunks[1].page_end == 2

    def test_chunk_id_is_deterministic(self):
        """Same text should always produce the same chunk ID."""
        pages = _make_pages(["Deterministic chunk content here."])
        chunks_a = chunk_pages(pages)
        chunks_b = chunk_pages(pages)
        assert chunks_a[0].chunk_id == chunks_b[0].chunk_id

    def test_chunk_id_differs_for_different_text(self):
        """Different texts should produce different chunk IDs."""
        pages_a = _make_pages(["First text about cues."])
        pages_b = _make_pages(["Second text about rewards."])
        chunks_a = chunk_pages(pages_a)
        chunks_b = chunk_pages(pages_b)
        assert chunks_a[0].chunk_id != chunks_b[0].chunk_id

    def test_repeated_text_has_unique_deterministic_ids(self):
        repeated_text = "A habit starts with a cue. " * 100
        pages = _make_pages([repeated_text, repeated_text])

        first_run = chunk_pages(pages)
        second_run = chunk_pages(pages)

        first_ids = [chunk.chunk_id for chunk in first_run]
        assert len(first_ids) == len(set(first_ids))
        assert first_ids == [chunk.chunk_id for chunk in second_run]

    def test_long_text_produces_multiple_chunks(self):
        """Text exceeding chunk_size should be split into multiple chunks."""
        # Create a text that is definitely longer than CHUNK_SIZE (1500 chars)
        long_text = ("This is a sentence about building good habits. " * 100)
        pages = _make_pages([long_text])
        chunks = chunk_pages(pages)
        assert len(chunks) > 1

    def test_empty_pages_produce_no_chunks(self):
        """Pages with no text should produce no chunks."""
        pages = _make_pages(["", "   ", "\n\n"])
        chunks = chunk_pages(pages)
        assert len(chunks) == 0

    def test_output_type(self):
        """Chunks should be Chunk dataclass instances."""
        pages = _make_pages(["Test text."])
        chunks = chunk_pages(pages)
        assert all(isinstance(c, Chunk) for c in chunks)


def test_book_pages_and_detected_topics_are_covered():
    pages = extract_pages()
    chunks = chunk_pages(pages)

    assert all(
        any(chunk.page_start <= page.page_num <= chunk.page_end for chunk in chunks)
        for page in pages
    )
    assert all(
        any(
            chunk.page_start <= page.page_num <= chunk.page_end
            and chunk.section and heading in chunk.section
            for chunk in chunks
        )
        for page in pages
        for heading in page.topic_headings or []
    )
