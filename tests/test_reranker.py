"""Tests for cross-encoder ordering and empty inputs."""

from unittest.mock import Mock

import numpy as np

from core import reranker


def _chunks():
    return [
        {"chunk_id": "a", "text": "A cue starts a habit.", "metadata": {}},
        {"chunk_id": "b", "text": "Habit stacking joins routines.", "metadata": {}},
    ]


def test_reranker_orders_by_cross_encoder_score(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([0.1, 0.9])
    monkeypatch.setattr(reranker, "CrossEncoder", Mock(return_value=model))
    engine = reranker.Reranker()

    results = engine.rerank("What is habit stacking?", _chunks(), top_n=1)

    assert [result["chunk_id"] for result in results] == ["b"]
    assert results[0]["rerank_score"] == 0.9
    model.predict.assert_called_once_with([
        ("What is habit stacking?", "A cue starts a habit."),
        ("What is habit stacking?", "Habit stacking joins routines."),
    ])


def test_reranker_does_not_call_model_for_empty_inputs(monkeypatch):
    model = Mock()
    monkeypatch.setattr(reranker, "CrossEncoder", Mock(return_value=model))
    engine = reranker.Reranker()

    assert engine.rerank("habit stacking", []) == []
    assert engine.rerank("  ", _chunks(), top_n=1) == _chunks()[:1]
    model.predict.assert_not_called()
