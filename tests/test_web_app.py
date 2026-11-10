"""Tests for web query validation and safe streaming errors."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import config
from web import app as web_app


def test_index_serves_chat_page():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert "Ask Atomic Habits" in response.text


def test_book_pdf_serves_configured_file(monkeypatch, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(config, "PDF_PATH", pdf)

    response = TestClient(web_app.app).get("/book.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-1.4"


def test_health_reports_configured_hosted_model(monkeypatch):
    monkeypatch.setattr(web_app, "_llm_client", SimpleNamespace(
        is_available=AsyncMock(return_value=True)
    ))
    monkeypatch.setattr(web_app, "_engine", object())
    monkeypatch.setattr(web_app, "VectorStore", lambda: SimpleNamespace(count=lambda: 4))

    response = TestClient(web_app.app).get("/api/health")

    assert response.json() == {
        "status": "ok",
        "llm_configured": True,
        "index_loaded": True,
        "chunk_count": 4,
    }


def test_query_request_trims_question_and_accepts_upper_top_k():
    request = web_app.QueryRequest(
        question="  What is habit stacking?  ", top_k=config.RERANK_CANDIDATES
    )

    assert request.question == "What is habit stacking?"
    assert request.top_k == config.RERANK_CANDIDATES


def test_omitted_top_k_uses_query_type_default():
    request = web_app.QueryRequest(question="What is habit stacking?")

    assert request.top_k is None


def test_query_stream_sends_sources_after_answer(monkeypatch):
    async def stream(question, top_k):
        assert question == "What is habit stacking?"
        assert top_k is None
        yield {"type": "metadata", "query_type": "factual"}
        yield {"type": "chunk", "content": "Habit stacking links routines."}
        yield {"type": "sources", "sources": [{"chapter": "Chapter 5"}]}
        yield {"type": "done", "answer": "Habit stacking links routines (Chapter 5, p.73)."}

    monkeypatch.setattr(web_app, "_engine", SimpleNamespace(query_stream=stream))

    response = TestClient(web_app.app).post(
        "/api/query", json={"question": "What is habit stacking?"}
    )

    events = [json.loads(line.removeprefix("data: "))
              for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events] == [
        "metadata", "chunk", "sources", "done"
    ]
    assert events[-1]["answer"].endswith("(Chapter 5, p.73).")


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "Valid?", "top_k": 0},
        {"question": "Valid?", "top_k": config.RERANK_CANDIDATES + 1},
        {"question": "Valid?", "top_k": "5"},
        {"question": "Valid?", "top_k": True},
    ],
)
def test_invalid_query_returns_422(payload):
    response = TestClient(web_app.app).post("/api/query", json=payload)

    assert response.status_code == 422


def test_unexpected_stream_error_hides_exception(monkeypatch):
    async def failing_stream(question, top_k):
        raise RuntimeError("private token and file path")
        yield {}

    monkeypatch.setattr(
        web_app, "_engine", SimpleNamespace(query_stream=failing_stream)
    )

    response = TestClient(web_app.app).post(
        "/api/query", json={"question": "What is a habit?"}
    )

    assert response.status_code == 200
    event = json.loads(response.text.removeprefix("data: ").strip())
    assert event == {"type": "error", "message": "Query failed. Please try again."}
    assert "private token" not in response.text
