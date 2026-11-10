"""Tests for the query pipeline and its failure paths."""

from unittest.mock import AsyncMock, Mock, call

import pytest

import config
from core.query_engine import QueryEngine
from core.scope_checker import OUT_OF_SCOPE_ANSWER


def _chunk(chunk_id: str, text: str, page: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "chapter": "Chapter 3: How to Build Better Habits",
            "page_start": page,
            "page_end": page,
        },
        "rerank_score": 0.9,
    }


def test_expanded_queries_send_only_top_fifteen_unique_candidates_to_reranker():
    retriever = Mock()
    first = [dict(_chunk(f"first_{index}", "A habit passage.", index + 1), score=0.01)
             for index in range(15)]
    second = [dict(_chunk(f"second_{index}", "Another habit passage.", index + 30), score=0.02)
              for index in range(15)]
    retriever.retrieve.side_effect = [first, second]
    engine = QueryEngine(retriever, Mock(), Mock(), Mock())

    candidates = engine._retrieve_for_queries(["habits", "routines"])

    assert len(candidates) == config.RERANK_CANDIDATES
    assert {candidate["chunk_id"] for candidate in candidates} == {
        f"second_{index}" for index in range(15)
    }


@pytest.mark.asyncio
async def test_query_runs_expansion_retrieval_reranking_and_generation():
    cue = _chunk("cue", "A cue starts the habit loop.", 42)
    stacking = _chunk("stacking", "Habit stacking links a new habit to an existing one.", 43)
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    retriever.retrieve.side_effect = [[cue], [cue, stacking]]
    reranker.rerank.return_value = [stacking, cue]
    llm_client.is_available = AsyncMock(return_value=True)
    llm_client.generate = AsyncMock(return_value="Link the habits (Chapter 3, p.43).")
    query_expander.expand = AsyncMock(return_value=[
        "What is habit stacking?", "link a new habit to an existing one"
    ])
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    result = await engine.query("What is habit stacking?", top_k=2)

    assert result["answer"] == "Link the habits (Chapter 3, p.43)."
    assert [source["chunk_id"] for source in result["sources"]] == ["stacking", "cue"]
    assert result["sources"][0]["metadata"]["page_start"] == 43
    assert result["sources"][0]["chapter"].startswith("Chapter 3")
    assert result["sources"][0]["pages"] == "43-43"
    assert "Habit stacking links" in result["sources"][0]["text"]
    assert result["timing"]["total"] >= 0
    assert retriever.retrieve.call_args_list == [
        call("What is habit stacking?", top_k=config.RERANK_CANDIDATES),
        call(
            "link a new habit to an existing one", top_k=config.RERANK_CANDIDATES
        ),
    ]
    reranker.rerank.assert_called_once()
    rerank_question, rerank_candidates = reranker.rerank.call_args.args
    assert rerank_question == "What is habit stacking?"
    assert [candidate["chunk_id"] for candidate in rerank_candidates] == [
        "cue", "stacking"
    ]
    assert rerank_candidates[0]["score"] > rerank_candidates[1]["score"]
    assert reranker.rerank.call_args.kwargs == {"top_n": 2}
    prompt = llm_client.generate.call_args.args[0]
    assert "Habit stacking links a new habit" in prompt[1]["content"]
    assert "Chapter 3: How to Build Better Habits, Pages 43-43" in prompt[1]["content"]


@pytest.mark.asyncio
async def test_stream_yields_sources_after_answer_and_finishes():
    stacking = _chunk("stacking", "Habit stacking connects two routines.", 43)
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [stacking]
    reranker.rerank.return_value = [stacking]
    llm_client.is_available = AsyncMock(return_value=True)
    query_expander.expand = AsyncMock(return_value=["What is habit stacking?"])

    async def answer_chunks():
        yield "Connect "
        yield "two routines."

    llm_client.generate = AsyncMock(return_value=answer_chunks())
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    events = [event async for event in engine.query_stream("What is habit stacking?")]

    assert [event["type"] for event in events] == [
        "metadata", "chunk", "chunk", "sources", "done"
    ]
    assert events[3]["sources"][0]["chunk_id"] == "stacking"
    assert "".join(event["content"] for event in events[1:3]) == "Connect two routines."
    assert events[-1]["answer"] == "Connect two routines (Chapter 3, p.43)."
    assert events[-1]["timing"]["total"] >= 0
    llm_client.generate.assert_awaited_once()
    assert llm_client.generate.call_args.kwargs == {"stream": True}


@pytest.mark.asyncio
async def test_query_with_no_retrieval_results_skips_reranker_and_generation():
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    retriever.retrieve.return_value = []
    llm_client.is_available = AsyncMock(return_value=False)
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    result = await engine.query("What is habit stacking?")

    assert result["sources"] == []
    assert "could not find" in result["answer"]
    reranker.rerank.assert_not_called()
    llm_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_expansion_failure_uses_original_question():
    stacking = _chunk("stacking", "Habit stacking connects routines.", 43)
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [stacking]
    reranker.rerank.return_value = [stacking]
    llm_client.is_available = AsyncMock(return_value=True)
    llm_client.generate = AsyncMock(return_value="Use an existing routine as a cue.")
    query_expander.expand = AsyncMock(side_effect=RuntimeError("expansion failed"))
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    result = await engine.query("What is habit stacking?")

    assert result["answer"] == (
        "Use an existing routine as a cue (Chapter 3, p.43)."
    )
    retriever.retrieve.assert_called_once_with(
        "What is habit stacking?", top_k=config.RERANK_CANDIDATES
    )


@pytest.mark.asyncio
async def test_unavailable_llm_still_returns_retrieved_sources():
    stacking = _chunk("stacking", "Habit stacking connects routines.", 43)
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [stacking]
    reranker.rerank.return_value = [stacking]
    llm_client.is_available = AsyncMock(return_value=False)
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    result = await engine.query("What is habit stacking?")

    assert "unavailable" in result["answer"]
    assert result["sources"][0]["chunk_id"] == "stacking"
    query_expander.expand.assert_not_called()
    llm_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "query_type", "expected_top_k"),
    [
        ("What is habit stacking?", "factual", 3),
        ("Why do habits plateau?", "analytical", 5),
        ("Summarize the chapter on habits", "summary", 8),
    ],
)
async def test_query_type_selects_srs_retrieval_depth(
    question: str, query_type: str, expected_top_k: int
):
    chunk = _chunk("habit", "A passage about habits.", 42)
    retriever, reranker, llm_client = Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [chunk]
    reranker.rerank.return_value = [chunk]
    llm_client.is_available = AsyncMock(return_value=False)
    engine = QueryEngine(retriever, reranker, llm_client, Mock())

    result = await engine.query(question)

    assert result["query_type"] == query_type
    assert reranker.rerank.call_args.kwargs["top_n"] == expected_top_k


@pytest.mark.asyncio
async def test_comparison_retrieves_each_subject_separately():
    chunk = _chunk("habit", "A passage about habits.", 42)
    retriever, reranker, llm_client = Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [chunk]
    reranker.rerank.return_value = [chunk]
    llm_client.is_available = AsyncMock(return_value=False)
    engine = QueryEngine(retriever, reranker, llm_client, Mock())
    question = "Compare habit stacking vs. temptation bundling"

    result = await engine.query(question)

    assert result["query_type"] == "comparison"
    assert [item.args[0] for item in retriever.retrieve.call_args_list] == [
        "habit stacking", "temptation bundling", question
    ]
    assert reranker.rerank.call_args.kwargs["top_n"] == 5


@pytest.mark.asyncio
async def test_unrelated_query_skips_every_pipeline_component():
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    result = await engine.query("What is the capital of France?")

    assert result["answer"] == OUT_OF_SCOPE_ANSWER
    assert result["sources"] == []
    assert result["query_type"] == "out_of_scope"
    retriever.retrieve.assert_not_called()
    reranker.rerank.assert_not_called()
    llm_client.is_available.assert_not_called()
    query_expander.expand.assert_not_called()


@pytest.mark.asyncio
async def test_unrelated_stream_returns_answer_without_retrieval():
    retriever, reranker, llm_client, query_expander = Mock(), Mock(), Mock(), Mock()
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    events = [event async for event in engine.query_stream("What is the capital of France?")]

    assert [event["type"] for event in events] == [
        "metadata", "chunk", "sources", "done"
    ]
    assert events[2]["sources"] == []
    assert events[1]["content"] == OUT_OF_SCOPE_ANSWER
    retriever.retrieve.assert_not_called()
    reranker.rerank.assert_not_called()
    llm_client.is_available.assert_not_called()
    query_expander.expand.assert_not_called()


@pytest.mark.asyncio
async def test_generation_error_does_not_reveal_exception():
    chunk = {
        "chunk_id": "chunk_1",
        "text": "A habit starts with a cue.",
        "metadata": {"chapter": "Chapter 3: Habits", "page_start": 42, "page_end": 42},
    }
    retriever = Mock()
    retriever.retrieve.return_value = [chunk]
    reranker = Mock()
    reranker.rerank.return_value = [chunk]
    llm_client = Mock()
    llm_client.is_available = AsyncMock(return_value=True)
    llm_client.generate = AsyncMock(side_effect=RuntimeError("private token"))
    query_expander = Mock()
    query_expander.expand = AsyncMock(return_value=["What is a habit?"])
    engine = QueryEngine(retriever, reranker, llm_client, query_expander)

    events = [event async for event in engine.query_stream("What is a habit?")]

    errors = [event for event in events if event["type"] == "error"]
    assert errors == [
        {"type": "error", "message": "Answer generation failed. Please try again."}
    ]
    assert events[-1]["type"] == "done"
