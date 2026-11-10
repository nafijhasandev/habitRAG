"""
Rule-based query classification for adaptive RAG responses.

WHY THIS EXISTS:
Not all questions are equal. A factual question like "What is the 2-minute rule?"
needs a direct, concise answer with a citation. An analytical question like
"Why do habits plateau after initial progress?" requires synthesizing evidence
from multiple passages and reasoning about cause-effect. A comparison question
like "What is the difference between goals and systems?" benefits from a
structured side-by-side format.

By classifying the query type before generating the answer, we can:
1. Select a system prompt template that guides the LLM toward the right format
   and depth of response for that query type.
2. Potentially adjust retrieval parameters (e.g., more chunks for summaries).
3. Improve user satisfaction by matching response style to intent.

HOW IT WORKS:
This module uses simple keyword pattern matching rather than a trained classifier.
The rules check for known indicator words/phrases (e.g., "compare", "why",
"summarize") and assign the query to one of four categories. This is fast,
transparent, and easy to debug. A learned classifier could be added later if
the rules prove insufficient, but for a focused single-book RAG system, keyword
rules work well because user queries tend to follow predictable patterns.

The four query types:
- "factual": direct questions seeking a specific fact or definition
- "analytical": questions requiring reasoning, explanation, or cause-effect
- "comparison": questions asking to contrast two or more concepts
- "summary": requests for an overview or synthesis of a topic
"""

import logging
import re

logger = logging.getLogger(__name__)


# Each entry maps a query type to its indicator patterns.
# Order matters: comparison and summary are checked before analytical,
# because "how does X compare" should be classified as comparison, not analytical.
_CLASSIFICATION_RULES: list[tuple[str, list[str]]] = [
    (
        "comparison",
        [
            "compare",
            "vs",
            "versus",
            "difference between",
            "contrast",
            "similarities",
        ],
    ),
    (
        "summary",
        [
            "summarize",
            "summary",
            "overview",
            "main points",
            "key ideas",
        ],
    ),
    (
        "analytical",
        [
            "why",
            "how does",
            "explain",
            "what causes",
            "relationship between",
            "impact of",
        ],
    ),
]


# System prompt templates guide the LLM toward the appropriate response format.
_PROMPT_TEMPLATES: dict[str, str] = {
    "factual": (
        "You are a helpful assistant answering questions about the book "
        "'Atomic Habits' by James Clear. Provide a direct, concise answer "
        "based on the provided context. Cite the specific chapter and page "
        "numbers from the context when possible. If the context does not "
        "contain the answer, say so clearly rather than guessing."
    ),
    "analytical": (
        "You are a helpful assistant answering questions about the book "
        "'Atomic Habits' by James Clear. This question requires deeper "
        "reasoning. Synthesize information from multiple parts of the context "
        "to build a thorough explanation. Explain cause-and-effect "
        "relationships and underlying mechanisms. Reference specific concepts "
        "and examples from the book to support your reasoning."
    ),
    "comparison": (
        "You are a helpful assistant answering questions about the book "
        "'Atomic Habits' by James Clear. Structure your response as a clear "
        "comparison. Identify the key dimensions of difference and similarity. "
        "Use a structured format (e.g., bullet points or labeled sections) to "
        "make the comparison easy to follow. Reference specific examples from "
        "the book for each point."
    ),
    "summary": (
        "You are a helpful assistant answering questions about the book "
        "'Atomic Habits' by James Clear. Provide a comprehensive overview "
        "that captures the main ideas. Organize the summary logically, "
        "covering the key themes and supporting points. Be thorough but "
        "concise -- prioritize the most important concepts from the context."
    ),
}


def classify_query(query: str) -> tuple[str, str]:
    """Classify a query into one of four types based on keyword patterns.

    The function scans the lowered query for indicator words/phrases in a
    priority order (comparison > summary > analytical > factual). The first
    match wins. If no pattern matches, the query defaults to "factual".

    Args:
        query: The user's natural language question.

    Returns:
        A tuple of (query_type, reason) where:
        - query_type: one of "factual", "analytical", "comparison", "summary"
        - reason: a human-readable string explaining which pattern matched,
          useful for logging and debugging
    """
    query_lower = query.lower().strip()

    for query_type, patterns in _CLASSIFICATION_RULES:
        for pattern in patterns:
            # Use word boundary matching to avoid false positives
            # (e.g., "show" matching inside "shower")
            if re.search(r"\b" + re.escape(pattern) + r"\b", query_lower):
                reason = f"Matched pattern '{pattern}' for type '{query_type}'"
                logger.info(
                    "Query classified as '%s': %s (query: '%.80s')",
                    query_type,
                    reason,
                    query,
                )
                return query_type, reason

    reason = "No specific pattern matched; defaulting to 'factual'"
    logger.info("Query classified as 'factual': %s (query: '%.80s')", reason, query)
    return "factual", reason


def get_prompt_template(query_type: str) -> str:
    """Return the system prompt template appropriate for the given query type.

    Each query type has a tailored system prompt that guides the LLM toward
    the right response format:
    - factual: direct answer with citation
    - analytical: deeper reasoning with multi-source synthesis
    - comparison: structured side-by-side comparison
    - summary: comprehensive overview of key themes

    Args:
        query_type: One of "factual", "analytical", "comparison", "summary".

    Returns:
        The system prompt string for the specified query type.

    Raises:
        ValueError: If query_type is not a recognized type.
    """
    if query_type not in _PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown query type '{query_type}'. "
            f"Expected one of: {list(_PROMPT_TEMPLATES.keys())}"
        )

    return (
        _PROMPT_TEMPLATES[query_type]
        + " Use Markdown bold (**...**) for main topic names and cited chapter titles."
    )
