"""Tests for ingestion orchestration without loading models or writing indexes."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from ingestion import pipeline


@pytest.fixture
def ingestion_components(monkeypatch):
    pages = [object()]
    chunks = [SimpleNamespace(text="A habit starts with a cue.", chunk_id="abc")]
    vector_store = Mock()
    vector_store.count.return_value = 0
    embedding_engine = Mock()
    embedding_engine.embed_texts.return_value = [[0.1, 0.2]]
    bm25_index = Mock()

    for module_name, class_name, instance in (
        ("core.vector_store", "VectorStore", vector_store),
        ("core.embeddings", "EmbeddingEngine", embedding_engine),
        ("core.bm25_index", "BM25Index", bm25_index),
    ):
        module = ModuleType(module_name)
        setattr(module, class_name, Mock(return_value=instance))
        monkeypatch.setitem(sys.modules, module_name, module)

    extract_pages = Mock(return_value=pages)
    chunk_pages = Mock(return_value=chunks)
    save_chunks_json = Mock()
    monkeypatch.setattr(pipeline, "extract_pages", extract_pages)
    monkeypatch.setattr(pipeline, "chunk_pages", chunk_pages)
    monkeypatch.setattr(pipeline, "_save_chunks_json", save_chunks_json)

    return SimpleNamespace(
        pages=pages,
        chunks=chunks,
        vector_store=vector_store,
        embedding_engine=embedding_engine,
        bm25_index=bm25_index,
        extract_pages=extract_pages,
        chunk_pages=chunk_pages,
        save_chunks_json=save_chunks_json,
    )


def test_ingestion_wires_all_stages(ingestion_components):
    components = ingestion_components

    pipeline.run_ingestion()

    components.extract_pages.assert_called_once_with()
    components.chunk_pages.assert_called_once_with(components.pages)
    components.embedding_engine.embed_texts.assert_called_once_with(
        [components.chunks[0].text]
    )
    components.vector_store.add_chunks.assert_called_once_with(
        components.chunks, [[0.1, 0.2]]
    )
    components.bm25_index.build.assert_called_once_with(components.chunks)
    components.bm25_index.save.assert_called_once_with(config.BM25_INDEX_PATH)
    components.save_chunks_json.assert_called_once_with(components.chunks)
    components.vector_store.reset.assert_not_called()


def test_existing_index_skips_ingestion(ingestion_components):
    components = ingestion_components
    components.vector_store.count.return_value = 3

    pipeline.run_ingestion()

    components.extract_pages.assert_not_called()
    components.embedding_engine.embed_texts.assert_not_called()
    components.vector_store.add_chunks.assert_not_called()
    components.bm25_index.build.assert_not_called()


def test_legacy_vector_metric_requires_rebuild(ingestion_components):
    components = ingestion_components
    components.vector_store.count.return_value = 3
    components.vector_store.require_cosine.side_effect = RuntimeError(
        "Vector index uses an old metric. Rebuild with ingest --force."
    )

    with pytest.raises(RuntimeError, match="ingest --force"):
        pipeline.run_ingestion()

    components.extract_pages.assert_not_called()


def test_force_replaces_existing_index(ingestion_components):
    components = ingestion_components
    components.vector_store.count.return_value = 3

    pipeline.run_ingestion(force=True)

    components.vector_store.reset.assert_called_once_with()
    components.vector_store.add_chunks.assert_called_once_with(
        components.chunks, [[0.1, 0.2]]
    )


def test_missing_pdf_does_not_reset_existing_index(ingestion_components):
    components = ingestion_components
    components.extract_pages.side_effect = FileNotFoundError("missing PDF")

    with pytest.raises(FileNotFoundError, match="missing PDF"):
        pipeline.run_ingestion(force=True)

    components.vector_store.reset.assert_not_called()
    components.embedding_engine.embed_texts.assert_not_called()
    components.vector_store.add_chunks.assert_not_called()


def test_no_chunks_stops_before_embedding(ingestion_components):
    components = ingestion_components
    components.chunk_pages.return_value = []

    with pytest.raises(ValueError, match="No text chunks"):
        pipeline.run_ingestion(force=True)

    components.embedding_engine.embed_texts.assert_not_called()
    components.vector_store.reset.assert_not_called()
    components.vector_store.add_chunks.assert_not_called()


def test_embedding_failure_does_not_reset_existing_index(ingestion_components):
    components = ingestion_components
    components.embedding_engine.embed_texts.side_effect = RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        pipeline.run_ingestion(force=True)

    components.vector_store.reset.assert_not_called()
    components.vector_store.add_chunks.assert_not_called()
    components.bm25_index.build.assert_not_called()


def test_store_failure_stops_later_stages(ingestion_components):
    components = ingestion_components
    components.vector_store.add_chunks.side_effect = OSError("store failed")

    with pytest.raises(OSError, match="store failed"):
        pipeline.run_ingestion()

    components.bm25_index.build.assert_not_called()
    components.save_chunks_json.assert_not_called()


def test_bm25_save_failure_is_reported(ingestion_components):
    components = ingestion_components
    components.bm25_index.save.side_effect = OSError("index failed")

    with pytest.raises(OSError, match="index failed"):
        pipeline.run_ingestion()

    components.save_chunks_json.assert_not_called()
