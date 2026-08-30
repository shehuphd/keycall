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


def test_gemini_list_classifies_a_bidi_only_model_as_realtime():
    """A realtime-only model reports no generateContent at all, only the
    bidi method, so this is provider metadata's own signal, not the
    identifier-rule override path other non-text modalities take."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash-native-audio-latest",
                        "supportedGenerationMethods": ["bidiGenerateContent"],
                    },
                ]
            },
        )

    from keycall import ModelCategory

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    discovery = client.list_models(categories={ModelCategory.REALTIME}, refresh=True)
    client.close()
    assert [m.id for m in discovery.models] == ["gemini-2.5-flash-native-audio-latest"]
    assert discovery.models[0].classification_source == "provider_metadata"


def test_xai_list_appends_grok_voice_from_the_catalog():
    """Grok Voice is absent from GET /v1/models (checked live 2026-08-15),
    so it has to be carried from the catalog or a realtime-capable key
    would show no models at all under that category."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "grok-4"}, {"id": "grok-4-fast"}]},
        )

    from keycall import ModelCategory

    client = KeyCall(provider="xai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler))
    discovery = client.list_models(categories=set(ModelCategory), refresh=True)
    client.close()
    voice = next(m for m in discovery.models if m.id == "grok-voice-latest")
    assert ModelCategory.REALTIME in voice.categories
    assert voice.classification_source == "keycall_catalog"
    assert voice.warnings
    # The two discovered models are untouched by the merge.
    assert {m.id for m in discovery.models} == {"grok-4", "grok-4-fast", "grok-voice-latest"}


def test_xai_list_does_not_duplicate_grok_voice_if_discovery_starts_listing_it():
    from keycall import ModelCategory

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "grok-voice-latest"}]})

    client = KeyCall(provider="xai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler))
    discovery = client.list_models(categories={ModelCategory.REALTIME}, refresh=True)
    client.close()
    matches = [m for m in discovery.models if m.id == "grok-voice-latest"]
    assert len(matches) == 1
    assert matches[0].classification_source == "keycall_rule"


def test_context_limit_reads_every_spelling_and_stays_none_otherwise():
    """Three providers report the input ceiling under three different
    names, and three report nothing. The normalizing layer's job is to read
    each spelling into one field rather than make callers learn all three,
    and to leave the field None where a provider is silent rather than
    invent a number a caller would budget against."""
    from keycall.adapters._base import context_limit

    assert context_limit({"inputTokenLimit": 1_048_576}) == 1_048_576  # Gemini
    assert context_limit({"max_input_tokens": 200_000}) == 200_000  # Anthropic
    assert context_limit({"context_length": 262_144}) == 262_144  # Moonshot
    # OpenAI and DeepSeek entries carry no ceiling at all.
    assert context_limit({"id": "gpt-5.6-luna", "created": 1_780_000_000}) is None
    assert context_limit({"id": "deepseek-v4-flash"}) is None
    # Strings are accepted because a JSON API may quote a number; anything
    # that isn't a usable count is refused rather than coerced.
    assert context_limit({"context_length": "128000"}) == 128_000
    assert context_limit({"context_length": 0}) is None
    assert context_limit({"context_length": -1}) is None
    assert context_limit({"context_length": "unlimited"}) is None
    assert context_limit({"context_length": True}) is None
    assert context_limit({"context_length": None}) is None


def test_anthropic_and_moonshot_populate_the_context_limit():
    """Both report a ceiling the caller can use, under names neither shares
    with the other or with Gemini."""
    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "claude-opus-5",
                        "display_name": "Claude Opus 5",
                        "created_at": "2026-02-01T00:00:00Z",
                        "max_input_tokens": 200000,
                    }
                ],
                "has_more": False,
            },
        )

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(anthropic_handler)
    )
    model = client.list_models(refresh=True).models[0]
    client.close()
    assert model.context_limit == 200000

    def moonshot_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "kimi-k2.6", "created": 1_770_000_000, "context_length": 262144},
                    {"id": "kimi-plain"},
                ]
            },
        )

    client = KeyCall(
        provider="moonshot", api_key=CANARY, httpx_transport=httpx.MockTransport(moonshot_handler)
    )
    models = {m.id: m for m in client.list_models(refresh=True).models}
    client.close()
    assert models["kimi-k2.6"].context_limit == 262144
    # Same adapter, same code path, silent entry: None rather than a guess.
    assert models["kimi-plain"].context_limit is None


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


COMPAT_PROVIDERS = ("deepseek", "moonshot", "xai", "perplexity")


def compat_usage_handler(usage):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    return handler


@pytest.mark.parametrize("provider", COMPAT_PROVIDERS)
def test_compat_reasoning_tokens_normalized(provider):
    usage = {
        "prompt_tokens": 6,
        "completion_tokens": 9,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 7},
    }
    result = run_generation(provider, compat_usage_handler(usage))
    assert result.usage.reasoning_tokens == 7


@pytest.mark.parametrize("provider", COMPAT_PROVIDERS)
@pytest.mark.parametrize(
    "usage_extra",
    [
        {},  # no completion_tokens_details at all
        {"completion_tokens_details": None},  # explicit null
        {"completion_tokens_details": {}},  # details without the field
    ],
)
def test_compat_unreported_reasoning_tokens_stay_none(provider, usage_extra):
    usage = {"prompt_tokens": 6, "completion_tokens": 9, "total_tokens": 15, **usage_extra}
    result = run_generation(provider, compat_usage_handler(usage))
    assert result.usage.reasoning_tokens is None


def test_compat_reasoning_tokens_zero_stays_zero():
    usage = {
        "prompt_tokens": 6,
        "completion_tokens": 9,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    result = run_generation("deepseek", compat_usage_handler(usage))
    assert result.usage.reasoning_tokens == 0


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
    on token counts alone can't see it. The field existed since 0.2.0 and
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
    # Descriptive fields aren't units and must not be coerced into one.
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


def test_truncation_is_reported_in_one_vocabulary_across_providers():
    """Each provider names a spent output budget differently. A caller
    shouldn't have to learn four spellings to notice its answer was cut
    off, so the condition is normalized into one warning."""
    from keycall._client import was_truncated

    # The four wire protocols, as their adapters render them.
    assert was_truncated("incomplete:max_output_tokens")  # OpenAI Responses
    assert was_truncated("max_tokens")                    # Anthropic
    assert was_truncated("MAX_TOKENS")                    # Gemini shouts
    assert was_truncated("length")                        # Chat Completions
    # A complete answer, in each provider's own word for it.
    assert not was_truncated("completed")
    assert not was_truncated("end_turn")
    assert not was_truncated("STOP")
    assert not was_truncated("stop")
    assert not was_truncated("tool_calls")
    assert not was_truncated(None)
    assert not was_truncated("")


def test_truncated_reply_carries_a_warning_saying_what_to_change():
    """The finish reason already says it, but only to someone who knows
    that provider's vocabulary, and it sits among timing and token counts."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "The answer begins and then"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 9, "output_tokens": 8},
            },
        )

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="claude-opus-5",
        messages=[Message(role="user", content=[TextInput(text="hi")])],
        max_output_tokens=8,
    )
    client.close()

    assert result.finish_reason == "max_tokens"
    warning = next((w for w in result.warnings if "max_output_tokens ran out" in w), None)
    assert warning is not None, f"no truncation warning in {result.warnings}"
    # It has to say what to do, not only what happened.
    assert "raise max_output_tokens" in warning
    # And explain the reasoning-model trap, which is how people hit this.
    assert "reasoning" in warning


def test_complete_reply_carries_no_truncation_warning():
    """A warning on every reply would train people to ignore it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        )

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="claude-opus-5",
        messages=[Message(role="user", content=[TextInput(text="hi")])],
    )
    client.close()
    assert not any("max_output_tokens ran out" in w for w in result.warnings)


def test_tracking_parameters_are_stripped_from_citation_urls():
    """OpenAI appends ?utm_source=openai to every URL its web search cites
    (verified live 2026-08-10; no other provider does, and OpenAI offers no
    option to turn it off). It attributes the click to OpenAI in the
    destination's analytics and follows the link into whatever a caller
    renders or stores, so KeyCall removes it."""
    from keycall import Citation

    # The case as reported.
    assert (
        Citation(url="https://www.imdb.com/name/nm0561030/bio/?utm_source=openai").url
        == "https://www.imdb.com/name/nm0561030/bio/"
    )
    # The whole utm_ family, and only that family.
    assert (
        Citation(url="https://e.com/p?utm_source=a&utm_medium=b&utm_campaign=c&id=42").url
        == "https://e.com/p?id=42"
    )
    # Case-insensitive, because a provider may shout.
    assert Citation(url="https://e.com/p?UTM_Source=a").url == "https://e.com/p"


def test_everything_other_than_tracking_survives_untouched():
    """A URL with nothing to strip must come back byte-identical: guessing
    at 'cruft' would break links rather than tidy them."""
    from keycall import Citation

    for url in [
        "https://e.com/p?id=42",
        "https://e.com/p",
        "https://e.com/p?a=1&b=2",                     # order preserved
        "https://e.com/p?q=hello+world&x=%2Fa%2Fb",    # encoding preserved
        "https://e.com/p#section",
        # Gemini's redirect is the citation by Google's design, not cruft.
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
        # A parameter that merely mentions the word must not be caught.
        "https://e.com/p?custom_utm_note=keep",
        "not a url at all",
    ]:
        assert Citation(url=url).url == url, url

    # A fragment survives even when tracking is removed from the query.
    assert Citation(url="https://e.com/p?utm_source=a#top").url == "https://e.com/p#top"


def test_surviving_parameters_are_passed_through_byte_for_byte():
    """Removing one parameter must not rewrite the others.

    Parsing the query and re-encoding it is the obvious implementation and
    silently rewrites `%20` to `+` and turns a valueless `&flag` into
    `&flag=`. Usually equivalent, occasionally not, and a signed or opaque
    parameter is exactly where "usually" fails.
    """
    from keycall import Citation

    cases = [
        # A session id whose mixed case must be untouched.
        (
            "https://example.com/?utm_source=chatgpt&session_id=wxwi232ADCW232aseren977A&page=3",
            "https://example.com/?session_id=wxwi232ADCW232aseren977A&page=3",
        ),
        # Percent-encoding survives rather than being normalized to "+".
        ("https://e.com/p?utm_source=a&q=hello%20world", "https://e.com/p?q=hello%20world"),
        ("https://e.com/p?utm_source=a&q=hello+world", "https://e.com/p?q=hello+world"),
        # A valueless flag keeps its shape; an empty value keeps its "=".
        ("https://e.com/p?utm_source=a&empty=&flag", "https://e.com/p?empty=&flag"),
        # A signed parameter, where a rewrite would invalidate the signature.
        ("https://e.com/p?utm_source=a&sig=ABC%2Fdef%3D%3D", "https://e.com/p?sig=ABC%2Fdef%3D%3D"),
        # Mixed-case keys and values, and a fragment after the query.
        ("https://e.com/Path?a=1&utm_medium=x&B=CaSe#Frag", "https://e.com/Path?a=1&B=CaSe#Frag"),
    ]
    for url, expected in cases:
        assert Citation(url=url).url == expected, url


def test_stripping_lets_dedupe_collapse_the_same_source():
    """Two citations for one page that differ only by a tracking parameter
    are one source. Before stripping they read as two."""
    from keycall import Citation
    from keycall.adapters._base import dedupe_citations

    same = [
        Citation(url="https://e.com/a?utm_source=openai", title="A"),
        Citation(url="https://e.com/a", title="A"),
    ]
    assert len(dedupe_citations(same)) == 1
