"""Tests for the context builder module."""

from core.context_builder import (
    _compute_overlap_ratio,
    build_context,
    build_prompt,
    format_source_location,
)


def _make_chunk(
    chunk_id: str,
    text: str,
    page_start: int = 1,
    page_end: int = 2,
    chapter: str = "1",
    rerank_score: float = 0.5,
) -> dict:
    """Helper to create a chunk dict for testing."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "chapter": chapter,
            "page_start": page_start,
            "page_end": page_end,
            "section": None,
        },
        "rerank_score": rerank_score,
    }


class TestOverlapRatio:
    """Test the character bigram overlap computation."""

    def test_identical_texts(self):
        ratio = _compute_overlap_ratio("hello world", "hello world")
        assert ratio == 1.0

    def test_completely_different(self):
        ratio = _compute_overlap_ratio("aaaa", "zzzz")
        assert ratio == 0.0

    def test_empty_input(self):
        assert _compute_overlap_ratio("", "hello") == 0.0
        assert _compute_overlap_ratio("hello", "") == 0.0

    def test_partial_overlap(self):
        ratio = _compute_overlap_ratio("hello world", "hello there")
        assert 0.0 < ratio < 1.0


class TestBuildContext:
    """Test context assembly from chunks."""

    def test_basic_formatting(self):
        chunks = [_make_chunk("c1", "Habits are the compound interest of self-improvement.")]
        context = build_context(chunks)
        assert "Chapter 1" in context
        assert "compound interest" in context

    def test_topic_label_is_included_with_passage(self):
        chunk = _make_chunk("c1", "Start with two minutes.")
        chunk["metadata"]["section"] = "THE TWO-MINUTE RULE"

        context = build_context([chunk])

        assert "Topic: THE TWO-MINUTE RULE" in context
        assert "Start with two minutes." in context

    def test_full_chapter_title_is_not_prefixed_twice(self):
        chunks = [_make_chunk("c1", "The two-minute rule.", chapter="Chapter 13: Two-Minute Rule")]

        context = build_context(chunks)

        assert "[Chapter 13: Two-Minute Rule, Pages 1-2]" in context
        assert "Chapter Chapter" not in context

    def test_sorts_by_page(self):
        chunks = [
            _make_chunk("c2", "Later content", page_start=50),
            _make_chunk("c1", "Earlier content", page_start=10),
        ]
        context = build_context(chunks)
        assert context.index("Earlier") < context.index("Later")

    def test_removes_duplicates(self):
        chunks = [
            _make_chunk("c1", "This exact text is duplicated exactly as is."),
            _make_chunk("c2", "This exact text is duplicated exactly as is."),
        ]
        context = build_context(chunks)
        # Should only appear once
        assert context.count("duplicated exactly") == 1

    def test_removes_near_duplicates_but_keeps_distinct_passages(self):
        original = "A habit is a behavior repeated often enough to become automatic."
        chunks = [
            _make_chunk("c1", original),
            _make_chunk("c2", original + " Today."),
            _make_chunk("c3", "Identity shapes the habits we repeat."),
        ]

        context = build_context(chunks)

        assert context.count(original) == 1
        assert "Identity shapes" in context

    def test_empty_chunks(self):
        context = build_context([])
        assert context == ""

    def test_truncation(self):
        chunks = [
            _make_chunk("c1", "A" * 500, rerank_score=0.9),
            _make_chunk("c2", "B" * 500, rerank_score=0.5),
            _make_chunk("c3", "C" * 500, rerank_score=0.3),
        ]
        context = build_context(chunks, max_length=600)
        # Should fit at most 1-2 chunks
        assert len(context) <= 600


class TestBuildPrompt:
    """Test prompt assembly."""

    def test_returns_message_list(self):
        messages = build_prompt("What is habit stacking?", "Some context.", "factual")
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_system_message_first(self):
        messages = build_prompt("Test?", "Context.", "factual")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_context(self):
        messages = build_prompt("Test?", "My context block.", "analytical")
        assert "My context block" in messages[1]["content"]
        assert "Test?" in messages[1]["content"]

    def test_different_query_types(self):
        for q_type in ("factual", "analytical", "comparison", "summary"):
            messages = build_prompt("Test?", "Context.", q_type)
            assert "Atomic Habits" in messages[0]["content"]
            assert "Answer only from the provided context" in messages[0]["content"]
            assert "Do not invent citations" in messages[0]["content"]


def test_source_location_without_chapter():
    assert format_source_location({"chapter": "", "page_start": 199, "page_end": 200}) == (
        "Pages 199-200"
    )
