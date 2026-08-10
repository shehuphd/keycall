"""Image generation: request shapes, output parts, and the gates.

Verified live 2026-08-10: OpenAI's /images/generations returns base64 PNG
with token usage; Gemini's image models answer on the ordinary
generateContent path with an inlineData JPEG part. Anthropic, DeepSeek,
Perplexity, and Moonshot generate no images at all.
"""

import json

import httpx
import pytest

from keycall import ErrorCode, ImageGenerationRequest, KeyCall, KeyCallError

CANARY = "sk-canary-image-key"


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def test_prompt_is_validated_before_anything_is_sent():
    with pytest.raises(ValueError):
        ImageGenerationRequest(model="m", prompt="")
    with pytest.raises(ValueError):
        ImageGenerationRequest(model="m", prompt="   ")


def test_openai_uses_the_images_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "created": 1,
                "output_format": "png",
                "data": [{"b64_json": "QUJD"}],
                "usage": {"input_tokens": 18, "output_tokens": 1056, "total_tokens": 1074},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_image(model="gpt-image-1", prompt="a blue circle")
    client.close()

    assert captured["path"].endswith("/images/generations")
    assert captured["body"] == {"model": "gpt-image-1", "prompt": "a blue circle"}
    assert result.operation.value == "image_generation"
    assert [(p.media_type, p.base64_data) for p in result.parts] == [("image/png", "QUJD")]
    assert result.usage.total_tokens == 1074


def test_openai_media_type_follows_the_reported_format():
    """The images endpoint can return webp or jpeg; hardcoding png would
    mislabel the bytes a caller then writes to disk."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output_format": "webp",
                "data": [{"b64_json": "QUJD"}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_image(model="gpt-image-1", prompt="a blue circle")
    client.close()
    assert result.parts[0].media_type == "image/webp"


def test_gemini_generates_images_on_the_ordinary_content_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-3.1-flash-image",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/jpeg", "data": "WFla"}}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 1389,
                                  "totalTokenCount": 1402},
            },
        )

    client = make_client("gemini", handler)
    result = client.generate_image(model="gemini-3.1-flash-image", prompt="a blue circle")
    client.close()

    assert captured["path"].endswith("/models/gemini-3.1-flash-image:generateContent")
    assert [(p.media_type, p.base64_data) for p in result.parts] == [("image/jpeg", "WFla")]
    assert result.usage.total_tokens == 1402


def test_text_alongside_the_image_is_surfaced_not_dropped():
    """A refusal or caveat arrives as a text part next to the picture."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "I adjusted the request slightly."},
                                {"inlineData": {"mimeType": "image/png", "data": "QUJD"}},
                            ]
                        }
                    }
                ],
            },
        )

    client = make_client("gemini", handler)
    result = client.generate_image(model="gemini-3.1-flash-image", prompt="a blue circle")
    client.close()
    assert len(result.parts) == 1
    assert any("adjusted the request" in w for w in result.warnings)


def test_a_response_with_no_image_is_a_typed_error():
    """An empty result would leave the caller guessing why no picture came
    back; the provider said something, and it shouldn't be swallowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "I can't make that."}]}}]},
        )

    client = make_client("gemini", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_image(model="gemini-3.1-flash-image", prompt="something refused")
    client.close()
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    message = str(excinfo.value)
    assert "no image" in message
    # The model said why; repeating it beats a bare "no image".
    assert "I can't make that." in message


def test_providers_without_image_generation_refuse_before_the_network():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    for provider in ("anthropic", "deepseek", "perplexity", "moonshot"):
        client = make_client(provider, handler)
        with pytest.raises(KeyCallError) as excinfo:
            client.generate_image(model="whatever", prompt="a blue circle")
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        message = str(excinfo.value)
        assert "cannot generate images" in message
        assert "openai" in message and "gemini" in message

    assert not calls, "the gate must fire before any request goes out"


@pytest.mark.anyio
async def test_async_generate_image_matches_the_sync_client():
    from keycall import AsyncKeyCall

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output_format": "png",
                "data": [{"b64_json": "QUJD"}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )

    async with AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.generate_image(model="gpt-image-1", prompt="a blue circle")

    assert [p.base64_data for p in result.parts] == ["QUJD"]
