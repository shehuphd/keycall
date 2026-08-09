"""Embeddings: request shapes, ordering, and the gates.

Verified live 2026-08-09: OpenAI returns 1536-dimension vectors from
/embeddings and Gemini 3072 from batchEmbedContents. Anthropic, DeepSeek,
Perplexity, and Moonshot expose no embeddings endpoint at all.
"""

import json

import httpx
import pytest

from keycall import EmbeddingRequest, ErrorCode, KeyCall, KeyCallError

CANARY = "sk-canary-embedding-key"


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def test_inputs_are_validated_before_anything_is_sent():
    with pytest.raises(ValueError):
        EmbeddingRequest(model="m", inputs=[])
    with pytest.raises(ValueError):
        EmbeddingRequest(model="m", inputs=["fine", ""])
    with pytest.raises(TypeError):
        EmbeddingRequest(model="m", inputs=["fine", 3])


def test_openai_embeddings_batch_and_keep_input_order():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        # Deliberately out of order: the index is what orders them.
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )

    client = make_client("openai", handler)
    result = client.embed(model="text-embedding-3-small", inputs=["first", "second"])
    client.close()

    assert captured["path"].endswith("/embeddings")
    assert captured["body"]["input"] == ["first", "second"]
    assert [part.values for part in result.parts] == [(0.1, 0.2), (0.3, 0.4)]
    assert result.operation.value == "embedding"
    assert result.usage.total_tokens == 4


def test_gemini_embeddings_use_the_batch_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]},
        )

    client = make_client("gemini", handler)
    result = client.embed(model="gemini-embedding-001", inputs=["first", "second"])
    client.close()

    assert captured["path"].endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert len(captured["body"]["requests"]) == 2
    assert captured["body"]["requests"][0]["content"]["parts"][0]["text"] == "first"
    assert [part.values for part in result.parts] == [(0.1, 0.2), (0.3, 0.4)]


def test_a_mismatched_vector_count_is_a_typed_error():
    """Returning fewer vectors than inputs would silently shift every
    caller's index by one, which is worse than failing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    client = make_client("openai", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.embed(model="text-embedding-3-small", inputs=["one", "two"])
    client.close()
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "cannot be matched" in str(excinfo.value)


def test_providers_without_embeddings_refuse_before_the_network():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    for provider in ("anthropic", "deepseek", "perplexity", "moonshot"):
        client = make_client(provider, handler)
        with pytest.raises(KeyCallError) as excinfo:
            client.embed(model="whatever", inputs=["hi"])
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        message = str(excinfo.value)
        assert "no embeddings API" in message
        assert "openai" in message and "gemini" in message

    assert not calls, "the gate must fire before any request goes out"


@pytest.mark.anyio
async def test_async_embed_matches_the_sync_client():
    from keycall import AsyncKeyCall

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"index": 0, "embedding": [0.5, 0.6]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    async with AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.embed(model="text-embedding-3-small", inputs=["only"])

    assert [part.values for part in result.parts] == [(0.5, 0.6)]
