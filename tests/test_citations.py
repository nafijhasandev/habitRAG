"""Tests for citation formatting against retrieved passage metadata."""

from core.citations import format_cited_answer


def _source(chapter: str = "Chapter 5: The Best Way to Start a New Habit") -> dict:
    return {"metadata": {"chapter": chapter, "page_start": 73, "page_end": 75}}


def test_normalizes_supported_chapter_and_page_citation():
    answer = "Habit stacking links two actions (Chapter 5, Pages 73-75)."

    assert format_cited_answer(answer, [_source()]) == (
        "Habit stacking links two actions (Chapter 5, p.73)."
    )


def test_normalizes_bracketed_source_header():
    answer = "Habit stacking links actions [Chapter 5, Pages 73-75]."

    assert format_cited_answer(answer, [_source()]) == (
        "Habit stacking links actions (Chapter 5, p.73)."
    )


def test_removes_unsupported_citation_and_uses_retrieved_location():
    answer = "Habit stacking links actions (Chapter 99, p.999)."

    assert format_cited_answer(answer, [_source()]) == (
        "Habit stacking links actions (Chapter 5, p.73)."
    )


def test_adds_citation_when_model_omits_it():
    assert format_cited_answer("Link a new habit to an existing one.", [_source()]) == (
        "Link a new habit to an existing one (Chapter 5, p.73)."
    )


def test_adds_page_to_supported_chapter_only_citation():
    answer = "Habit stacking joins actions (Chapter 5)."

    assert format_cited_answer(answer, [_source()]) == (
        "Habit stacking joins actions (Chapter 5, p.73)."
    )


def test_does_not_invent_citation_without_valid_metadata():
    source = {"metadata": {"chapter": "Preface", "page_start": 1, "page_end": 1}}

    assert format_cited_answer("An introduction.", [source]) == "An introduction."
    assert format_cited_answer("An introduction.", []) == "An introduction."
    assert format_cited_answer("", [_source()]) == ""
