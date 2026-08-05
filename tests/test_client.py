import pickle

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    ErrorCode,
    KeyCall,
    KeyCallError,
    Message,
    ModelCategory,
    ProviderProtocol,
    TextInput,
)

CANARY = "sk-canary-1a2b3c4d5e6f"

OPENAI_MODELS = {
    "data": [
        {"id": "gpt-4o-mini"},
        {"id": "gpt-4o"},
        {"id": "text-embedding-3-small"},
        {"id": "whisper-1"},
        {"id": "dall-e-3"},
        {"id": "some-mystery-model"},
    ]
}

OPENAI_GENERATION = {
    "id": "resp_123",
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 1,
        "total_tokens": 13,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def openai_mock(list_payload=OPENAI_MODELS, generation_payload=OPENAI_GENERATION):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=list_payload, headers={"x-request-id": "req_1"})
        if request.url.path == "/v1/responses":
            return httpx.Response(200, json=generation_payload, headers={"x-request-id": "req_2"})
        return httpx.Response(404, json={"error": {"message": "no such path"}})

    return httpx.MockTransport(handler)


def make_client(**overrides):
    kwargs = {
        "provider": "openai",
        "api_key": CANARY,
        "httpx_transport": openai_mock(),
    }
    kwargs.update(overrides)
    return KeyCall(**kwargs)


def test_identity_properties_exposed_but_no_api_key():
    client = make_client()
    assert client.provider == "openai"
    assert client.protocol is ProviderProtocol.OPENAI
    assert client.base_url == "https://api.openai.com/v1"
    assert not hasattr(client, "api_key")


def test_identity_is_immutable():
    client = make_client()
    with pytest.raises(AttributeError):
        client.provider = "anthropic"
    with pytest.raises(AttributeError):
        client.api_key = "sk-new"
    with pytest.raises(AttributeError):
        del client.provider


def test_repr_never_contains_credential():
    client = make_client()
    assert CANARY not in repr(client)
    assert CANARY not in str(client)


def test_pickle_blocked():
    with pytest.raises(TypeError):
        pickle.dumps(make_client())


def test_context_manager_closes_and_closed_client_refuses_calls():
    with make_client() as client:
        assert not client.closed
    assert client.closed
    with pytest.raises(RuntimeError):
        client.list_models()


def test_list_models_defaults_to_text_and_excludes_unknown():
    with make_client() as client:
        discovery = client.list_models(refresh=True)
    ids = [model.id for model in discovery.models]
    assert "gpt-4o-mini" in ids and "gpt-4o" in ids
    assert "text-embedding-3-small" not in ids
    assert "whisper-1" not in ids
    assert "dall-e-3" not in ids
    assert "some-mystery-model" not in ids  # unknown never enters default picker
    assert discovery.categories == frozenset({ModelCategory.TEXT_GENERATION})
    assert not discovery.from_cache


def test_list_models_category_filter_and_unknown_opt_in():
    with make_client() as client:
        images = client.list_models(categories={ModelCategory.IMAGE_GENERATION}, refresh=True)
        assert [m.id for m in images.models] == ["dall-e-3"]
        unknowns = client.list_models(categories={ModelCategory.UNKNOWN})
        assert [m.id for m in unknowns.models] == ["some-mystery-model"]


def test_list_models_rejects_plain_strings():
    with make_client() as client, pytest.raises(TypeError):
        client.list_models(categories={"text_generation"})


def test_list_models_cache_hit_and_refresh():
    with make_client() as client:
        first = client.list_models(refresh=True)
        second = client.list_models()
        third = client.list_models(refresh=True)
    assert not first.from_cache
    assert second.from_cache
    assert not third.from_cache


def test_generate_text_end_to_end():
    with make_client() as client:
        result = client.generate_text(
            model="gpt-4o-mini",
            messages=[Message(role="user", content=[TextInput(text="Say ok")])],
            max_output_tokens=16,
        )
    assert result.text == "ok"
    assert result.usage.total_tokens == 13
    assert result.round_trip_duration_ms > 0
    assert result.provider_request_id == "req_2"
    assert result.finish_reason == "completed"


def test_invalid_key_maps_to_typed_error_without_leaking_credential():
    def handler(request: httpx.Request) -> httpx.Response:
        # Provider echoes the key back — the scrubber must remove it.
        return httpx.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {CANARY}"}}
        )

    client = make_client(httpx_transport=httpx.MockTransport(handler))
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    error = excinfo.value
    assert error.code is ErrorCode.INVALID_API_KEY
    assert not error.retryable
    assert CANARY not in str(error)
    assert CANARY not in repr(error)


@pytest.mark.anyio
async def test_async_client_parity():
    async with AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=openai_mock()
    ) as client:
        discovery = await client.list_models(refresh=True)
        assert any(model.id == "gpt-4o-mini" for model in discovery.models)
        result = await client.generate_text(
            model="gpt-4o-mini",
            messages=[Message(role="user", content=[TextInput(text="Say ok")])],
        )
        assert result.text == "ok"
    assert client.closed


def test_custom_target_client_hits_custom_base_url():
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "lab-llama-3"}]})

    client = KeyCall(
        provider="university-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(handler),
    )
    discovery = client.list_models(refresh=True)
    assert seen_urls == ["https://llm.example.edu/v1/models"]
    assert [m.id for m in discovery.models] == ["lab-llama-3"]
