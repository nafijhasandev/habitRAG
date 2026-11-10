"""Tests for persistent cosine search and metadata roundtrips."""

import pytest

import config
from core.vector_store import VectorStore
from ingestion.chunker import Chunk


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    return VectorStore()


def _chunks():
    return [
        Chunk("stacking", "Habit stacking joins two routines.", "Chapter 5", None, 73, 73),
        Chunk("identity", "Identity shapes repeated behavior.", "Chapter 2", None, 30, 30),
    ]


def test_empty_collection_uses_cosine_distance(store):
    assert store.count() == 0
    assert store.collection_exists() is False
    assert store.configuration_space() == "cosine"


def test_insert_query_and_metadata_filter_roundtrip(store):
    store.add_chunks(_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=2)
    filtered = store.search([1.0, 0.0], where={"chapter": "Chapter 2"})

    assert store.count() == 2
    assert results[0]["chunk_id"] == "chunk_stacking"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["metadata"]["page_start"] == 73
    assert [item["chunk_id"] for item in filtered] == ["chunk_identity"]


def test_reset_preserves_cosine_collection(store):
    store.add_chunks(_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    store.reset()

    assert store.count() == 0
    assert store.configuration_space() == "cosine"


def test_mismatched_embeddings_fail_before_writing(store):
    with pytest.raises(ValueError, match="Mismatch"):
        store.add_chunks(_chunks(), [[1.0, 0.0]])

    assert store.count() == 0


def test_legacy_distance_metric_is_rejected(store, monkeypatch):
    monkeypatch.setattr(store, "configuration_space", lambda: "l2")

    with pytest.raises(RuntimeError, match="ingest --force"):
        store.require_cosine()
