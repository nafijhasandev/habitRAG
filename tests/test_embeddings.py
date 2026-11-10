"""Tests for embedding shape, normalization options, and input validation."""

from unittest.mock import Mock

import numpy as np
import pytest

from core import embeddings


@pytest.fixture
def engine(monkeypatch):
    model = Mock()
    monkeypatch.setattr(embeddings, "SentenceTransformer", Mock(return_value=model))
    return embeddings.EmbeddingEngine(), model


def test_embeds_texts_in_batches_with_normalization(engine):
    embedding_engine, model = engine
    model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

    vectors = embedding_engine.embed_texts(["Habit stacking", "Two-minute rule"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert all(len(vector) == 2 for vector in vectors)
    model.encode.assert_called_once_with(
        ["Habit stacking", "Two-minute rule"],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embeds_query_with_normalization(engine):
    embedding_engine, model = engine
    model.encode.return_value = np.array([0.6, 0.8])

    vector = embedding_engine.embed_query("habit stacking")

    assert vector == [0.6, 0.8]
    model.encode.assert_called_once_with(
        "habit stacking", normalize_embeddings=True, show_progress_bar=False
    )


def test_rejects_empty_inputs_before_calling_model(engine):
    embedding_engine, model = engine

    with pytest.raises(ValueError, match="empty list"):
        embedding_engine.embed_texts([])
    with pytest.raises(ValueError, match="empty query"):
        embedding_engine.embed_query("  ")

    model.encode.assert_not_called()
