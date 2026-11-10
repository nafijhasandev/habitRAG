"""Tests for the immediate book-scope check."""

import pytest

from core.scope_checker import is_in_scope


@pytest.mark.parametrize(
    "question",
    [
        "What are the four laws of behavior change?",
        "How does habit stacking work?",
        "Summarize chapter 13.",
        "What is the two-minute rule?",
        "Why did British Cycling improve?",
        "How can I stay motivated to exercise?",
        "What does the book say about small improvements?",
        "Why do small changes compound?",
        "What is the Diderot Effect?",
    ],
)
def test_book_questions_stay_in_scope(question):
    assert is_in_scope(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "What is today's weather in Dhaka?",
        "Who won the World Cup?",
        "How do I write a Python decorator?",
        "",
    ],
)
def test_unrelated_questions_are_rejected(question):
    assert not is_in_scope(question)
