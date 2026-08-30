import json
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


# Each provider's observed bad-key rejection: status code and body shape as the
# provider sends them, key echoes included where the provider echoes the key.
# A provider added to the catalog without an entry here fails the test below.
BAD_KEY_RESPONSES = {
    "openai": (401, {"error": {"message": f"Incorrect API key provided: {CANARY}"}}),
    "anthropic": (
        401,
        {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
        },
    ),
    "gemini": (
        400,
        {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
            }
        },
    ),
    "deepseek": (
        401,
        {
            "error": {
                "message": f"Authentication Fails, Your api key: {CANARY} is invalid",
                "type": "authentication_error",
            }
        },
    ),
    "perplexity": (401, {"error": {"message": "Unauthorized", "type": "unauthorized"}}),
    "moonshot": (
        401,
        {
            "error": {
                "message": "auth failed: invalid api key",
                "type": "invalid_authentication_error",
            }
        },
    ),
    "xai": (401, {"error": f"Incorrect API key provided: {CANARY}", "code": "..."}),
    "assemblyai": (401, {"error": "Authentication error, API token missing/invalid"}),
    "deepgram": (401, {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}),
}


def _all_providers():
    from keycall._registry import supported_providers

    return supported_providers()


@pytest.mark.parametrize("provider", _all_providers())
def test_every_provider_maps_bad_key_to_typed_error(provider):
    assert provider in BAD_KEY_RESPONSES, (
        f"{provider} has no recorded bad-key response; add its observed "
        "rejection shape to BAD_KEY_RESPONSES so its mapping is covered"
    )
    status, body = BAD_KEY_RESPONSES[provider]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    client = KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    error = excinfo.value
    client.close()
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


def test_moonshot_pinned_sampling_values_are_gated_before_the_network():
    """Reported by an integrator for kimi-k3; the live probe found every
    kimi model pins temperature=1.0 and top_p=0.95 and 400s on anything
    else (verified 2026-08-09)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    client = KeyCall(
        provider="moonshot", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    ask = [Message(role="user", content=[TextInput(text="hi")])]

    for model in ("kimi-k3", "kimi-k2.6", "kimi-k2.7-code"):
        with pytest.raises(KeyCallError) as excinfo:
            client.generate_text(model=model, messages=ask, temperature=0.2)
        assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE
        # The message must name the value that works, not just refuse.
        assert "only temperature=1" in str(excinfo.value)

        with pytest.raises(KeyCallError) as excinfo:
            client.generate_text(model=model, messages=ask, top_p=0.5)
        assert "only top_p=0.95" in str(excinfo.value)

    assert not calls, "the gate must fire before any request goes out"
    client.close()


def test_moonshot_permitted_sampling_values_pass_the_gate():
    """The pinned values are accepted, so the gate must not reject them —
    a blanket 'no sampling params' rule would break working calls."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "kimi-k3",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = KeyCall(
        provider="moonshot", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="kimi-k3",
        messages=[Message(role="user", content=[TextInput(text="hi")])],
        temperature=1.0,
        top_p=0.95,
    )
    client.close()
    assert result.text == "ok"
    assert seen["body"]["temperature"] == 1.0
    assert seen["body"]["top_p"] == 0.95


def test_stale_catalog_sets_the_flag_and_says_so():
    """catalog_stale was a declared field nothing ever set; a caller
    reading it always saw 'fresh' however old the bundled data was."""
    import keycall._client as client_module

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    fresh = client.list_models(refresh=True)
    assert fresh.catalog_stale is False
    assert not any("catalog" in w for w in fresh.warnings)

    original = client_module.catalog_is_stale
    client_module.catalog_is_stale = lambda: True
    try:
        stale = client.list_models(refresh=True)
    finally:
        client_module.catalog_is_stale = original
    client.close()

    assert stale.catalog_stale is True
    assert any("catalog was last verified" in w for w in stale.warnings)
