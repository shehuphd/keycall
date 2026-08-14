"""Speech generation: request bodies, output parts, and the gates.

Verified live 2026-08-12: OpenAI's /audio/speech answers with the audio
file itself — no JSON envelope, Content-Type audio/mpeg on success but
application/json on an error from the very same route, which is why the
transport classifies each response from its own Content-Type rather than
from a flag declared per request. Gemini's TTS models answer on the
ordinary generateContent path with an inlineData part, just like image
generation, but raw PCM (audio/L16;codec=pcm;rate=24000) rather than a
playable container. Anthropic, DeepSeek, Perplexity, and Moonshot generate
no speech at all — Anthropic's "voice mode" is a consumer-app feature
built on a third-party TTS subcontractor, not a public API endpoint.
"""

import base64
import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, SpeechGenerationRequest

CANARY = "sk-canary-speech-key"


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def test_text_is_validated_before_anything_is_sent():
    with pytest.raises(ValueError):
        SpeechGenerationRequest(model="m", text="")
    with pytest.raises(ValueError):
        SpeechGenerationRequest(model="m", text="   ")


def test_openai_returns_raw_audio_bytes_not_json():
    """The one operation in the package whose successful response is not a
    JSON envelope. The transport has to read that from the response's own
    Content-Type — this proves it does, end to end through the client
    itself, not through a synthetic RequestSpec."""
    captured = {}
    audio_bytes = b"\xff\xf3\xc4\xc4not really mp3 but exercises the path"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=audio_bytes, headers={"content-type": "audio/mpeg"})

    client = make_client("openai", handler)
    result = client.generate_speech(model="gpt-4o-mini-tts", text="Hello there.")
    client.close()

    assert captured["path"].endswith("/audio/speech")
    assert captured["body"] == {"model": "gpt-4o-mini-tts", "input": "Hello there."}
    assert "voice" not in captured["body"], "an unset voice must not be sent at all"
    assert result.operation.value == "speech_generation"
    assert len(result.parts) == 1
    clip = result.parts[0]
    assert clip.media_type == "audio/mpeg"
    assert base64.b64decode(clip.base64_data) == audio_bytes


def test_openai_sends_voice_only_when_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"abc", headers={"content-type": "audio/mpeg"})

    client = make_client("openai", handler)
    client.generate_speech(model="gpt-4o-mini-tts", text="Hi.", voice="coral")
    client.close()
    assert captured["body"]["voice"] == "coral"


def test_openai_tts1_family_rejects_a_missing_voice_and_the_message_survives():
    """Voice is optional on gpt-4o-mini-tts but required on
    tts-1 and tts-1-hd, which answer 400 with a Pydantic-shaped message
    when it's omitted (both live-verified 2026-08-12). KeyCall does not
    paper over this with a default voice the caller never chose — the
    provider's own message already says precisely what to add, so it is
    passed through rather than replaced."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "1 validation error for Request\nbody -> voice\n  field required (type=value_error.missing)",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = make_client("openai", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="tts-1", text="Hi.")
    client.close()
    assert "voice" in str(excinfo.value)
    assert "field required" in str(excinfo.value)


def test_openai_media_type_follows_the_response_header():
    """response_format on this endpoint isn't sent by KeyCall, so the
    format is whatever OpenAI's own default currently is — read from the
    header it sent, not assumed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"RIFF....WAVE", headers={"content-type": "audio/wav"})

    client = make_client("openai", handler)
    result = client.generate_speech(model="tts-1", text="Hi.")
    client.close()
    assert result.parts[0].media_type == "audio/wav"


def test_a_json_error_on_the_binary_route_is_still_a_normal_provider_error():
    """OpenAI's own speech endpoint answers a 404 with application/json,
    not audio — content-type sniffing must classify each response on its
    own, not by what the operation nominally returns."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "The model `bogus` does not exist.", "code": "model_not_found"}},
            headers={"content-type": "application/json; charset=utf-8"},
        )

    client = make_client("openai", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="bogus", text="Hi.")
    client.close()
    assert excinfo.value.code is ErrorCode.MODEL_NOT_AVAILABLE
    assert "does not exist" in str(excinfo.value)


def test_an_empty_binary_body_is_a_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"", headers={"content-type": "audio/mpeg"})

    client = make_client("openai", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="tts-1", text="Hi.")
    client.close()
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_gemini_generates_speech_on_the_ordinary_content_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-2.5-flash-preview-tts",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/L16;codec=pcm;rate=24000",
                                        "data": "AQID",
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 47,
                    "totalTokenCount": 52,
                },
            },
        )

    client = make_client("gemini", handler)
    result = client.generate_speech(model="gemini-2.5-flash-preview-tts", text="Hello there.")
    client.close()

    assert captured["path"].endswith("/models/gemini-2.5-flash-preview-tts:generateContent")
    assert captured["body"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert "speechConfig" not in captured["body"]["generationConfig"], (
        "an unset voice must not send a speechConfig block at all"
    )
    assert result.parts[0].media_type == "audio/L16;codec=pcm;rate=24000"
    assert result.parts[0].base64_data == "AQID"
    assert result.usage.total_tokens == 52


def test_gemini_sends_a_voice_config_only_when_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "audio/L16", "data": "AQ=="}}]}}
                ]
            },
        )

    client = make_client("gemini", handler)
    client.generate_speech(model="gemini-2.5-flash-preview-tts", text="Hi.", voice="Kore")
    client.close()
    voice_config = captured["body"]["generationConfig"]["speechConfig"]["voiceConfig"]
    assert voice_config["prebuiltVoiceConfig"]["voiceName"] == "Kore"


def test_gemini_text_only_reply_is_a_typed_error_naming_what_it_said():
    """A refusal or clarifying question arrives as words instead of
    speech — the same posture as image generation's equivalent case."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "I can't say that aloud."}]}}]},
        )

    client = make_client("gemini", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="gemini-2.5-flash-preview-tts", text="something refused")
    client.close()
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    message = str(excinfo.value)
    assert "no audio" in message
    assert "I can't say that aloud." in message


def test_the_gate_message_is_built_from_the_speech_capability_key(monkeypatch):
    """The refusal names which providers can help, by re-reading the
    catalog for `speech_generation`. `.operation` on the raised error is
    set independently in the same statement, so it can't tell this apart
    from a gate that queried the wrong key by accident (e.g.
    "image_generation", whose provider set is identical today) — the
    only way to prove the right string is used is to watch what's asked
    for."""
    import keycall.adapters._base as base_module

    seen = []
    original = base_module.providers_with

    def spy(capability):
        seen.append(capability)
        return original(capability)

    monkeypatch.setattr(base_module, "providers_with", spy)

    client = make_client("anthropic", lambda request: httpx.Response(500))
    with pytest.raises(KeyCallError):
        client.generate_speech(model="whatever", text="Hi.")
    client.close()
    assert seen == ["speech_generation"]


def test_providers_without_speech_generation_refuse_before_the_network():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    for provider in ("anthropic", "deepseek", "perplexity", "moonshot"):
        client = make_client(provider, handler)
        with pytest.raises(KeyCallError) as excinfo:
            client.generate_speech(model="whatever", text="Hi.")
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        # Pinned to the operation field, not just the message text: today
        # speech_generation and image_generation happen to list the same
        # two providers, so a gate that read the wrong capability entirely
        # would still produce byte-identical wording — only the .operation
        # value distinguishes it.
        assert excinfo.value.operation == "speech_generation"
        message = str(excinfo.value)
        assert "cannot generate speech" in message
        assert "openai" in message and "gemini" in message

    assert not calls, "the gate must fire before any request goes out"


@pytest.mark.anyio
async def test_async_generate_speech_matches_the_sync_client():
    from keycall import AsyncKeyCall

    audio_bytes = b"async-path-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=audio_bytes, headers={"content-type": "audio/mpeg"})

    async with AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.generate_speech(model="gpt-4o-mini-tts", text="Hi.")

    assert base64.b64decode(result.parts[0].base64_data) == audio_bytes
