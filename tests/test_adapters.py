import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-adapter-key"


def run_generation(provider, handler, **kwargs):
    client = KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )
    return client.generate_text(
        model="test-model",
        messages=[
            Message(role="system", content=[TextInput(text="Be brief.")]),
            Message(role="user", content=[TextInput(text="Hello")]),
            Message(role="assistant", content=[TextInput(text="Hi!")]),
            Message(role="user", content=[TextInput(text="Bye")]),
        ],
        max_output_tokens=32,
    )


def test_openai_body_shape_and_assistant_output_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
        )

    result = run_generation("openai", handler)
    body = captured["body"]
    assert body["max_output_tokens"] == 32
    roles = [item["role"] for item in body["input"]]
    assert roles == ["system", "user", "assistant", "user"]
    types = [item["content"][0]["type"] for item in body["input"]]
    assert types == ["input_text", "input_text", "output_text", "input_text"]
    assert result.text == "done"


def test_anthropic_system_extraction_and_required_max_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
            headers={"request-id": "req_abc"},
        )

    result = run_generation("anthropic", handler)
    body = captured["body"]
    assert body["system"] == "Be brief."
    assert all(m["role"] in ("user", "assistant") for m in body["messages"])
    assert body["max_tokens"] == 32
    assert result.provider_request_id == "req_abc"
    assert result.finish_reason == "end_turn"
    # Anthropic reports no total; it must stay None, never fabricated.
    assert result.usage.total_tokens is None
    assert result.usage.input_tokens == 9


def test_anthropic_default_max_tokens_when_unspecified():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "x"}], "usage": {}},
        )

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    client.generate_text(
        model="m", messages=[Message(role="user", content=[TextInput(text="hi")])]
    )
    assert captured["body"]["max_tokens"] == 4096


def test_gemini_model_role_system_instruction_and_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "responseId": "resp_g1",
                "modelVersion": "test-model-001",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "done"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 5,
                },
            },
        )

    result = run_generation("gemini", handler)
    assert captured["path"].endswith("/models/test-model:generateContent")
    body = captured["body"]
    assert body["systemInstruction"]["parts"] == [{"text": "Be brief."}]
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user"]
    assert result.provider_request_id == "resp_g1"
    assert result.usage.total_tokens == 5


def test_gemini_list_uses_provider_metadata_over_rules():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                        "inputTokenLimit": 1000000,
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {"name": "models/unlisted-experiment"},
                ]
            },
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    discovery = client.list_models(refresh=True)
    assert [m.id for m in discovery.models] == ["gemini-2.5-flash"]
    model = discovery.models[0]
    assert model.classification_source == "provider_metadata"
    assert model.context_limit == 1000000


def test_gemini_invalid_key_400_maps_to_invalid_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY


def test_compat_adapter_string_content_and_usage():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "deepseek-chat"}]})
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 2,
                    "total_tokens": 8,
                    "prompt_cache_hit_tokens": 0,
                },
            },
        )

    result = run_generation("deepseek", handler)
    body = captured["body"]
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert isinstance(body["messages"][1]["content"], str)
    assert body["max_tokens"] == 32
    assert result.text == "done"
    assert result.usage.total_tokens == 8


def test_anthropic_paginated_list():
    pages = [
        {
            "data": [{"id": "claude-a"}],
            "has_more": True,
            "last_id": "claude-a",
        },
        {"data": [{"id": "claude-b"}], "has_more": False},
    ]
    seen_params = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json=pages[len(seen_params) - 1])

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    discovery = client.list_models(refresh=True)
    assert [m.id for m in discovery.models] == ["claude-a", "claude-b"]
    assert seen_params[1].get("after_id") == "claude-a"


def test_non_text_input_parts_raise_typed_error():
    from keycall import ImageInput

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[TextInput(text="what is this"), ImageInput(url="https://x.example/i.png")],
                )
            ],
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_gemini_non_text_families_stay_out_of_the_text_picker():
    """These advertise generateContent and then refuse a text call: the
    Interactions-only models, the computer-use preview, and Lyria (music).
    Verified against the live list 2026-08-09."""
    from keycall import ModelCategory

    advertised = [
        "gemini-3.5-flash",
        "gemini-omni-flash-preview",
        "deep-research-pro-preview-12-2025",
        "antigravity-preview-05-2026",
        "gemini-2.5-computer-use-preview-10-2025",
        "lyria-3-pro-preview",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
                    for m in advertised
                ]
            },
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    text = [m.id for m in client.list_models(refresh=True).models]
    every = client.list_models(categories=set(ModelCategory), refresh=True).models
    client.close()

    assert text == ["gemini-3.5-flash"]
    # Excluded, not dropped: they stay listable under UNKNOWN.
    assert len(every) == len(advertised)
    assert {m.id for m in every if ModelCategory.UNKNOWN in m.categories} == set(advertised[1:])


def test_gemini_retired_model_error_names_the_maintained_aliases():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "status": "NOT_FOUND",
                    "message": (
                        "This model models/gemini-2.5-flash is no longer available "
                        "to new users."
                    ),
                }
            },
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="gemini-2.5-flash",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    client.close()
    assert excinfo.value.code is ErrorCode.MODEL_NOT_AVAILABLE
    message = str(excinfo.value)
    assert "no longer available" in message
    assert "gemini-flash-latest" in message, "the error should name a model that works"


def test_gemini_ordinary_not_found_keeps_the_provider_message_alone():
    """Only the retirement case earns the extra guidance; a plain 404 must
    not be padded with advice that doesn't apply."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"status": "NOT_FOUND", "message": "models/typo is not found"}},
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="typo", messages=[Message(role="user", content=[TextInput(text="hi")])]
        )
    client.close()
    assert "gemini-flash-latest" not in str(excinfo.value)
