"""Fast, conservative scope check for questions about Atomic Habits."""

import re

OUT_OF_SCOPE_ANSWER = (
    "That information is not in Atomic Habits by James Clear. "
    "Ask about the book's ideas, examples, or chapters."
)

_BOOK_PHRASES = (
    "atomic habits",
    "james clear",
    "the book",
    "this book",
    "chapter ",
    "two-minute rule",
    "two minute rule",
    "2-minute rule",
    "2 minute rule",
    "four laws",
    "1 percent better",
    "one percent better",
    "habit stacking",
    "temptation bundling",
    "marginal gains",
    "small changes",
    "small improvements",
    "tiny changes",
)

_BOOK_WORDS = {
    "habit", "habits", "routine", "routines", "cue", "cues", "craving",
    "cravings", "reward", "rewards", "behavior", "behaviour", "identity",
    "goals", "goal", "systems", "system", "motivation", "motivated",
    "procrastination", "procrastinating", "discipline", "consistency",
    "repetition", "environment", "exercise", "reading", "meditation",
    "goldilocks", "diderot", "plateau", "willpower", "selfcontrol", "cycling",
}


def is_in_scope(question: str) -> bool:
    """Allow book-related questions without searching document sources."""
    normalized = question.casefold()
    if any(phrase in normalized for phrase in _BOOK_PHRASES):
        return True
    words = set(re.findall(r"[a-z]+", normalized))
    return bool(words & _BOOK_WORDS)
