"""Tests for the retriever module (hybrid retrieval + RRF)."""

from unittest.mock import Mock

import pytest

import config
from core.bm25_index import BM25Index
from core.retriever import HybridRetriever, _reciprocal_rank_fusion
from ingestion.chunker import Chunk


def _result(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"Passage about {chunk_id}",
        "metadata": {"chapter": "Chapter 1", "page_start": 1, "page_end": 1},
        "score": 1.0,
    }


def test_hybrid_retrieval_promotes_passage_found_by_both_sources():
    embedding_engine, vector_store, bm25_index = Mock(), Mock(), Mock()
    embedding_engine.embed_query.return_value = [0.1, 0.2]
    vector_store.search.return_value = [_result("dense_only"), _result("habit_stacking")]
    bm25_index.search.return_value = [_result("sparse_only"), _result("habit_stacking")]

    results = HybridRetriever(vector_store, bm25_index, embedding_engine).retrieve(
        "What is habit stacking?", top_k=3
    )

    assert [result["chunk_id"] for result in results] == [
        "habit_stacking", "dense_only", "sparse_only"
    ]
    assert results[0]["sources"] == "both"
    embedding_engine.embed_query.assert_called_once_with("What is habit stacking?")
    vector_store.search.assert_called_once_with([0.1, 0.2], top_k=config.RETRIEVAL_TOP_K)
    bm25_index.search.assert_called_once_with(
        "What is habit stacking?", top_k=config.RETRIEVAL_TOP_K
    )


def test_hybrid_retrieval_keeps_relevant_passages_from_either_source():
    vector_store, bm25_index = Mock(), Mock()
    vector_store.search.return_value = [_result("semantic_match")]
    bm25_index.search.return_value = [_result("exact_phrase_match")]
    retriever = HybridRetriever(vector_store, bm25_index, Mock())

    results = retriever.retrieve("two-minute rule", top_k=2)

    assert {result["chunk_id"] for result in results} == {
        "semantic_match", "exact_phrase_match"
    }
    assert {result["sources"] for result in results} == {"dense", "sparse"}


def test_hybrid_retrieval_returns_no_passages_when_both_sources_are_empty():
    vector_store, bm25_index = Mock(), Mock()
    vector_store.search.return_value = []
    bm25_index.search.return_value = []

    results = HybridRetriever(vector_store, bm25_index, Mock()).retrieve(
        "habit stacking"
    )

    assert results == []


@pytest.mark.parametrize(
    ("question", "expected_chunk_id"),
    [
        ("What is habit stacking?", "chunk_stacking"),
        ("Explain the two-minute rule", "chunk_two_minute"),
    ],
)
def test_keyword_search_finds_expected_passage_in_small_book_corpus(
    question: str, expected_chunk_id: str
):
    passages = [
        Chunk("stacking", "Habit stacking adds a new habit after an existing habit.",
              "Chapter 5", None, 73, 73),
        Chunk("two_minute", "The two-minute rule makes a new habit easy to start.",
              "Chapter 13", None, 162, 162),
        Chunk("identity", "Identity-based habits focus on who you wish to become.",
              "Chapter 2", None, 30, 30),
    ]
    bm25_index = BM25Index()
    bm25_index.build(passages)
    vector_store = Mock()
    vector_store.search.return_value = []
    retriever = HybridRetriever(vector_store, bm25_index, Mock())

    results = retriever.retrieve(question, top_k=1)

    assert results[0]["chunk_id"] == expected_chunk_id
    assert results[0]["sources"] == "sparse"


class TestReciprocalRankFusion:
    """Test the RRF merging logic in isolation."""

    def test_rrf_merges_two_lists(self):
        """Items appearing in both lists should score higher."""
        dense_results = [
            {"chunk_id": "a", "text": "chunk a", "metadata": {}, "score": 0.9},
            {"chunk_id": "b", "text": "chunk b", "metadata": {}, "score": 0.8},
            {"chunk_id": "c", "text": "chunk c", "metadata": {}, "score": 0.7},
        ]

        sparse_results = [
            {"chunk_id": "b", "text": "chunk b", "metadata": {}, "score": 5.0},
            {"chunk_id": "d", "text": "chunk d", "metadata": {}, "score": 4.0},
            {"chunk_id": "a", "text": "chunk a", "metadata": {}, "score": 3.0},
        ]

        merged = _reciprocal_rank_fusion(
            dense_results, sparse_results, k=60
        )

        # Items in both lists should be ranked higher
        ids = [m["chunk_id"] for m in merged]
        assert "a" in ids
        assert "b" in ids

        # a and b appear in both lists, so should rank above d and c
        a_idx = ids.index("a")
        b_idx = ids.index("b")
        d_idx = ids.index("d")
        assert a_idx < d_idx or b_idx < d_idx

    def test_rrf_empty_lists(self):
        """Empty input lists should produce empty output."""
        merged = _reciprocal_rank_fusion([], [], k=60)
        assert merged == []

    def test_rrf_one_empty_list(self):
        """One empty list should still return results from the other."""
        dense_results = [
            {"chunk_id": "a", "text": "chunk a", "metadata": {}, "score": 0.9},
        ]
        merged = _reciprocal_rank_fusion(dense_results, [], k=60)
        assert len(merged) == 1
        assert merged[0]["chunk_id"] == "a"

    def test_rrf_assigns_sources(self):
        """Each result should indicate which retriever found it."""
        dense = [
            {"chunk_id": "a", "text": "a", "metadata": {}, "score": 0.9},
            {"chunk_id": "b", "text": "b", "metadata": {}, "score": 0.8},
        ]
        sparse = [
            {"chunk_id": "b", "text": "b", "metadata": {}, "score": 5.0},
            {"chunk_id": "c", "text": "c", "metadata": {}, "score": 4.0},
        ]
        merged = _reciprocal_rank_fusion(dense, sparse, k=60)

        sources_map = {m["chunk_id"]: m["sources"] for m in merged}
        assert sources_map["a"] == "dense"
        assert sources_map["b"] == "both"
        assert sources_map["c"] == "sparse"
