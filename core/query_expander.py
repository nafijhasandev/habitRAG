"""
Query expansion using an LLM to improve retrieval recall.

WHY THIS EXISTS:
A single user query often fails to match all relevant passages in the corpus.
For example, the query "How to break bad habits?" might miss passages that
discuss "eliminating unwanted behaviors" or "inverting the laws of behavior
change" because the exact keywords differ. This vocabulary mismatch problem
is one of the fundamental challenges in information retrieval.

Query expansion mitigates this by generating alternative phrasings of the
original query. Each phrasing uses different vocabulary and angles, so when
we search for all of them, we cast a wider net over the corpus. The union of
results from all expanded queries has higher recall (finds more relevant
passages) than searching with the original query alone.

HOW IT WORKS:
1. The user's query is sent to the LLM with a prompt asking for 2-3
   alternative phrasings that would find relevant passages.
2. The LLM returns alternatives that rephrase the query, use synonyms,
   and approach the topic from different angles.
3. All phrasings (original + alternatives) are used for retrieval.
4. Results are merged and deduplicated before reranking.

The LLM is a good tool for expansion because it understands semantics and
can generate paraphrases that a rule-based system would miss. If the LLM
is unavailable, we gracefully fall back to using just the original query.
"""

import logging

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands a query into multiple alternative phrasings using an LLM.

    This improves retrieval recall by searching with semantically equivalent
    but lexically different queries, catching passages that the original
    query alone might miss.

    Attributes:
        llm_client: An LLM client instance with an expand_query method.
    """

    def __init__(self, llm_client: object) -> None:
        """Initialize with an LLM client for generating expansions.

        Args:
            llm_client: An object with an async `expand_query(query) -> list[str]`
                method. Typically an instance of core.llm_client.LLMClient.
        """
        self.llm_client = llm_client

    async def expand(self, query: str) -> list[str]:
        """Generate alternative phrasings of the query for broader retrieval.

        Returns the original query plus 2-3 LLM-generated alternatives.
        If the LLM is unavailable or returns an error, falls back to
        returning just the original query so the pipeline can continue.

        Args:
            query: The user's original search query.

        Returns:
            A list of queries starting with the original, followed by
            alternative phrasings. Minimum length is 1 (the original).
        """
        if not query.strip():
            return [query]

        try:
            expanded = await self.llm_client.expand_query(query)

            if not expanded:
                logger.warning("LLM returned empty expansion, using original query.")
                return [query]

            # Deduplicate while preserving order, original query first
            seen: set[str] = set()
            unique_queries: list[str] = []
            for q in [query] + expanded:
                normalized = q.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_queries.append(q.strip())

            logger.info(
                "Query expanded from 1 to %d variants: %s",
                len(unique_queries),
                unique_queries,
            )
            return unique_queries

        except Exception:
            # Graceful degradation: if the LLM is down, the pipeline
            # continues with just the original query
            logger.warning(
                "Query expansion failed, falling back to original query.",
                exc_info=True,
            )
            return [query]
