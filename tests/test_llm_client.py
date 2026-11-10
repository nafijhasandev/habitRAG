"""Tests for hosted model selection, fallback, and streaming failures."""

import httpx
import pytest

import config
from core.llm_client import LLMClient


async def _client_with_transport(monkeypatch, handler):
    monkeypatch.setattr(config, "GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setattr(config, "AI_GATEWAY_API_KEY", "vercel-test-key")
    client = LLMClient()
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _answer(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.mark.asyncio
async def test_no_keys_reports_unavailable_and_does_not_request(monkeypatch):
    for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "AI_GATEWAY_API_KEY"):
        monkeypatch.setattr(config, name, "")
    client = LLMClient()
    try:
        assert await client.is_available() is False
        with pytest.raises(RuntimeError, match="All configured"):
            await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_groq_is_primary_and_uses_configured_model(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return _answer("A cited answer.")

    client = await _client_with_transport(monkeypatch, respond)
    try:
        assert await client.is_available() is True
        answer = await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()

    assert answer == "A cited answer."
    assert len(requests) == 1
    assert requests[0].url.host == "api.groq.com"
    assert requests[0].headers["authorization"] == "Bearer groq-test-key"
    assert b'"model":"openai/gpt-oss-120b"' in requests[0].content


@pytest.mark.asyncio
async def test_rate_limit_falls_back_to_openrouter_free(monkeypatch):
    hosts = []

    def respond(request):
        hosts.append(request.url.host)
        if request.url.host == "api.groq.com":
            return httpx.Response(429)
        assert b'"model":"openrouter/free"' in request.content
        return _answer("Fallback answer.")

    client = await _client_with_transport(monkeypatch, respond)
    try:
        answer = await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()

    assert answer == "Fallback answer."
    assert hosts == ["api.groq.com", "openrouter.ai"]


@pytest.mark.asyncio
async def test_missing_keys_are_skipped_and_model_can_be_overridden(monkeypatch):
    def respond(request):
        assert request.url.host == "openrouter.ai"
        assert b'"model":"chosen-free-model"' in request.content
        return _answer("Configured answer.")

    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setattr(config, "AI_GATEWAY_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "chosen-free-model")
    client = LLMClient()
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        answer = await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()
    assert answer == "Configured answer."


@pytest.mark.asyncio
async def test_vercel_is_second_fallback(monkeypatch):
    hosts = []

    def respond(request):
        hosts.append(request.url.host)
        if request.url.host != "ai-gateway.vercel.sh":
            return httpx.Response(503)
        assert b'"model":"nvidia/nemotron-3.5-lightning-free"' in request.content
        return _answer("Vercel answer.")

    client = await _client_with_transport(monkeypatch, respond)
    try:
        answer = await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()

    assert answer == "Vercel answer."
    assert hosts == ["api.groq.com", "openrouter.ai", "ai-gateway.vercel.sh"]


@pytest.mark.asyncio
async def test_all_provider_errors_do_not_expose_response(monkeypatch):
    def respond(request):
        return httpx.Response(429, text="private provider details")

    client = await _client_with_transport(monkeypatch, respond)
    try:
        with pytest.raises(RuntimeError, match="All configured") as error:
            await client.generate([{"role": "user", "content": "habits"}])
    finally:
        await client.close()
    assert "private provider details" not in str(error.value)


@pytest.mark.asyncio
async def test_stream_falls_back_before_first_token(monkeypatch):
    hosts = []

    def respond(request):
        hosts.append(request.url.host)
        if request.url.host == "api.groq.com":
            return httpx.Response(429)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"Habit "}}]}\n\n'
                 'data: {"choices":[{"delta":{"content":"stacking."}}]}\n\n'
                 'data: [DONE]\n\n',
        )

    client = await _client_with_transport(monkeypatch, respond)
    try:
        stream = await client.generate([{"role": "user", "content": "habits"}], stream=True)
        chunks = [chunk async for chunk in stream]
    finally:
        await client.close()

    assert chunks == ["Habit ", "stacking."]
    assert hosts == ["api.groq.com", "openrouter.ai"]


@pytest.mark.asyncio
async def test_stream_does_not_restart_after_partial_answer(monkeypatch):
    hosts = []

    def respond(request):
        hosts.append(request.url.host)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n'
                 'data: {broken\n\n',
        )

    client = await _client_with_transport(monkeypatch, respond)
    try:
        stream = await client.generate([{"role": "user", "content": "habits"}], stream=True)
        with pytest.raises(RuntimeError, match="stream interrupted"):
            [chunk async for chunk in stream]
    finally:
        await client.close()
    assert hosts == ["api.groq.com"]


@pytest.mark.asyncio
async def test_query_expansion_removes_model_numbering(monkeypatch):
    def respond(request):
        return _answer('1. "building a routine"\n2) "habit cues"\n- rewards')

    client = await _client_with_transport(monkeypatch, respond)
    try:
        alternatives = await client.expand_query("How do habits form?")
    finally:
        await client.close()

    assert alternatives == ["building a routine", "habit cues", "rewards"]
