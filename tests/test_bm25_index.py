"""Tests for BM25 ranking and persisted index roundtrips."""

import pytest

from core.bm25_index import BM25Index
from ingestion.chunker import Chunk


def _chunks():
    return [
        Chunk("stacking", "Habit stacking adds a habit after a daily routine.",
              "Chapter 5", None, 73, 73),
        Chunk("identity", "Identity shapes the behavior you repeat.",
              "Chapter 2", None, 30, 30),
        Chunk("environment", "Environment design makes cues visible.",
              "Chapter 6", None, 82, 82),
    ]


def test_keyword_search_ranks_matching_passage_first():
    index = BM25Index()
    index.build(_chunks())

    results = index.search("habit stacking", top_k=2)

    assert results[0]["chunk_id"] == "chunk_stacking"
    assert results[0]["score"] > 0
    assert results[0]["metadata"]["page_start"] == 73


def test_saved_index_loads_with_same_ranking(tmp_path):
    index = BM25Index()
    index.build(_chunks())
    path = tmp_path / "bm25.pkl"
    index.save(path)

    restored = BM25Index.load(path)

    assert [result["chunk_id"] for result in restored.search("habit stacking")] == [
        result["chunk_id"] for result in index.search("habit stacking")
    ]


def test_empty_index_and_query_fail_clearly():
    index = BM25Index()
    with pytest.raises(ValueError, match="empty chunks"):
        index.build([])
    with pytest.raises(RuntimeError, match="not built"):
        index.search("habit")
