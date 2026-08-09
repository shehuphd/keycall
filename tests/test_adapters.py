import json

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
    from keycall import AudioInput

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
                    content=[TextInput(text="listen"), AudioInput(url="https://x.example/a.mp3")],
                )
            ],
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    # OpenAI takes no audio, and the refusal has to name who does rather
    # than leaving the caller to work it out.
    message = str(excinfo.value)
    assert "does not accept audio input" in message
    assert "gemini" in message


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


# A 1x1 PNG: the signature is what the media-type sniffer reads.
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _image_body(provider, part, **kwargs):
    from keycall import ImageInput  # noqa: F401

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_payload(provider))

    client = KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )
    try:
        client.generate_text(
            model="m",
            messages=[Message(role="user", content=[TextInput(text="what colour?"), part])],
        )
    finally:
        client.close()
    return seen["body"]


def _ok_payload(provider):
    if provider == "anthropic":
        return {
            "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "blue"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    if provider == "gemini":
        return {
            "modelVersion": "m",
            "candidates": [{"content": {"parts": [{"text": "blue"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        }
    if provider == "openai":
        return {
            "model": "m",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "blue"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        }
    return {
        "model": "m",
        "choices": [{"message": {"content": "blue"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


def test_image_bytes_map_to_each_provider_wire_shape():
    """Shapes verified live 2026-08-09, one solid-colour PNG per provider."""
    from keycall import ImageInput

    part = ImageInput(data=PNG_BYTES)

    openai = _image_body("openai", part)["input"][0]["content"]
    image = next(c for c in openai if c["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")

    anthropic = _image_body("anthropic", part)["messages"][0]["content"]
    block = next(b for b in anthropic if b["type"] == "image")
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"

    gemini = _image_body("gemini", part)["contents"][0]["parts"]
    inline = next(p for p in gemini if "inlineData" in p)["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert inline["data"]

    compat = _image_body("moonshot", part)["messages"][0]["content"]
    entry = next(c for c in compat if c["type"] == "image_url")
    assert entry["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_media_type_is_sniffed_not_trusted():
    """Anthropic and Gemini reject a mismatched media_type, and a caller
    passing bytes often doesn't know the format."""
    from keycall import ImageInput

    mislabelled = ImageInput(data=PNG_BYTES, media_type="image/jpeg")
    block = next(
        b
        for b in _image_body("anthropic", mislabelled)["messages"][0]["content"]
        if b["type"] == "image"
    )
    assert block["source"]["media_type"] == "image/png"


def test_unidentifiable_image_bytes_are_a_typed_error():
    from keycall import ImageInput

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="m",
            messages=[Message(role="user", content=[ImageInput(data=b"not an image")])],
        )
    client.close()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "media_type" in str(excinfo.value)


def test_image_gates_name_the_form_that_works():
    """Support splits by form: Gemini and Moonshot read bytes but refuse a
    URL, and DeepSeek's API is text only (all verified 2026-08-09)."""
    from keycall import ImageInput

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    def refuse(provider, part):
        client = KeyCall(
            provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
        )
        with pytest.raises(KeyCallError) as excinfo:
            client.generate_text(
                model="m", messages=[Message(role="user", content=[part])]
            )
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        return str(excinfo.value)

    remote = ImageInput(url="https://x.example/i.png")
    for provider in ("gemini", "moonshot"):
        message = refuse(provider, remote)
        assert "does not fetch image URLs" in message
        assert "data=" in message, "the error should name the form that works"

    unsupported = refuse("deepseek", ImageInput(data=PNG_BYTES))
    assert "does not accept image input" in unsupported
    assert "openai" in unsupported, "list the providers that do support images"


def test_images_belong_in_user_messages():
    from keycall import ImageInput

    client = KeyCall(
        provider="openai",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="m",
            messages=[Message(role="assistant", content=[ImageInput(data=PNG_BYTES)])],
        )
    client.close()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 24
PDF_BYTES = b"%PDF-1.4\n1 0 obj\nendobj\ntrailer\n%%EOF\n"


def test_audio_and_file_map_to_each_provider_wire_shape():
    """Shapes verified live 2026-08-09 with a WAV tone and a one-line PDF."""
    from keycall import AudioInput, FileInput

    gemini_audio = _image_body("gemini", AudioInput(data=WAV_BYTES))["contents"][0]["parts"]
    inline = next(p for p in gemini_audio if "inlineData" in p)["inlineData"]
    assert inline["mimeType"] == "audio/wav"

    document = FileInput(data=PDF_BYTES, filename="report.pdf")

    openai = _image_body("openai", document)["input"][0]["content"]
    entry = next(c for c in openai if c["type"] == "input_file")
    assert entry["filename"] == "report.pdf"
    assert entry["file_data"].startswith("data:application/pdf;base64,")

    anthropic = _image_body("anthropic", document)["messages"][0]["content"]
    block = next(b for b in anthropic if b["type"] == "document")
    assert block["source"]["media_type"] == "application/pdf"

    gemini_file = _image_body("gemini", document)["contents"][0]["parts"]
    doc = next(p for p in gemini_file if "inlineData" in p)["inlineData"]
    assert doc["mimeType"] == "application/pdf"


def test_audio_and_file_gates_name_who_does_support_them():
    """Audio is Gemini-only across the supported providers, and neither
    modality should fail with a vague refusal."""
    from keycall import AudioInput, FileInput

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    def refuse(provider, part):
        client = KeyCall(
            provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
        )
        with pytest.raises(KeyCallError) as excinfo:
            client.generate_text(model="m", messages=[Message(role="user", content=[part])])
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        return str(excinfo.value)

    audio = refuse("openai", AudioInput(data=WAV_BYTES))
    assert "does not accept audio input" in audio
    assert "gemini" in audio

    files = refuse("deepseek", FileInput(data=PDF_BYTES))
    assert "does not accept file input" in files
    assert "openai" in files and "anthropic" in files

    # A URL form nobody verified must not be smuggled through as bytes.
    remote = refuse("gemini", FileInput(url="https://x.example/doc.pdf"))
    assert "does not fetch file URLs" in remote


def test_non_token_billing_surfaces_in_provider_units():
    """Perplexity charges per request on top of tokens, so a budget built
    on token counts alone cannot see it. The field existed since 0.2.0 and
    nothing ever populated it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "sonar",
                "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 17,
                    "total_tokens": 24,
                    "search_context_size": "low",
                    "cost": {
                        "input_tokens_cost": 1e-05,
                        "output_tokens_cost": 2e-05,
                        "request_cost": 0.005,
                        "total_cost": 0.00502,
                    },
                },
            },
        )

    client = KeyCall(
        provider="perplexity", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="sonar", messages=[Message(role="user", content=[TextInput(text="hi")])]
    )
    client.close()

    units = dict(result.usage.provider_units or ())
    assert units["request_cost"] == 0.005
    assert units["total_cost"] == 0.00502
    # Descriptive fields are not units and must not be coerced into one.
    assert "search_context_size" not in units
    # Token counts stay where they were.
    assert result.usage.total_tokens == 24


def test_providers_without_non_token_billing_report_none():
    """None means the provider said nothing, which is different from zero."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = KeyCall(
        provider="deepseek", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="deepseek-chat", messages=[Message(role="user", content=[TextInput(text="hi")])]
    )
    client.close()
    assert result.usage.provider_units is None
