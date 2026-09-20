"""Dictation: one round trip, verbatim transcript plus a cleaned rewrite.

Live-probed 2026-09-17 against dictation.assemblyai.com (a different host
from the REST and streaming APIs). A multipart POST with a JSON config
part and a wav/pcm audio part answers text (verbatim, never LLM-altered),
words[] with per-word confidence but no timing, overall confidence,
audio_duration_ms, session_id, request_time_ms, and llm_response/llm_error
(the cleaned rewrite, on by default). AssemblyAI is the one provider with
the endpoint; everyone else refuses toward it.
"""

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    DictationRequest,
    DictationResult,
    DictationWord,
    ErrorCode,
    KeyCall,
    KeyCallError,
)

CANARY = "sk-canary-dictate-key"

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
MP3 = b"ID3" + b"\x00" * 40


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


def dictation_response(**overrides):
    body = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "words": [
            {"text": "The", "confidence": 0.99},
            {"text": "fox", "confidence": 1.0},
        ],
        "confidence": 0.996,
        "audio_duration_ms": 2645,
        "session_id": "sess-abc",
        "request_time_ms": 212.5,
        "llm_response": "The quick brown fox jumps over the lazy dog.",
        "llm_error": None,
    }
    body.update(overrides)
    return body


# --- gates and validation, all before the network ---


def test_public_exports_are_reachable_from_the_package():
    for name in ("DictationRequest", "DictationResult", "DictationWord"):
        import keycall

        assert name in keycall.__all__
        assert getattr(keycall, name) is not None


def test_dictationless_provider_refuses_and_names_the_supporting_one():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.dictate(audio=WAV)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "assemblyai" in info.value.message


def test_compressed_audio_refuses_before_any_call():
    client = make_client("assemblyai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.dictate(audio=MP3)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "wav" in info.value.message and "pcm" in info.value.message


def test_request_validation_rejects_malformed_asks():
    with pytest.raises(ValueError):
        DictationRequest(data=b"")  # empty audio
    with pytest.raises(ValueError):
        DictationRequest(data=WAV, keyterms=("ok", ""))  # blank keyterm


def test_unrecognized_audio_asks_for_a_media_type():
    client = make_client("assemblyai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.dictate(audio=b"\x00\x01\x02\x03raw pcm-ish bytes")
    # media_type_for cannot identify raw bytes and asks for a media_type.
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_raw_pcm_is_accepted_when_the_caller_labels_it():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json=dictation_response())

    client = make_client("assemblyai", handler)
    client.dictate(audio=b"\x00\x01" * 800, media_type="audio/pcm")
    assert b'name="audio"' in captured["body"]
    assert b"audio/pcm" in captured["body"]


# --- the wire ---


def test_dictation_posts_config_and_audio_parts_to_the_dictation_host():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization", "")
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json=dictation_response())

    client = make_client("assemblyai", handler)
    result = client.dictate(
        audio=WAV,
        context_prompt="a nature sentence",
        keyterms=["quick brown fox", "lazy dog"],
        language="en",
        cleanup_instruction="Fix punctuation only.",
    )

    assert captured["url"] == "https://dictation.assemblyai.com/v1/transcribe/live"
    assert captured["method"] == "POST"
    # Raw key in the Authorization header, no Bearer prefix.
    assert captured["auth"] == CANARY
    assert captured["content_type"].startswith("multipart/form-data")

    body = captured["body"]
    assert b'name="config"' in body and b"application/json" in body
    assert b'name="audio"' in body and b"audio/wav" in body
    assert b'"stt_prompt": "a nature sentence"' in body
    assert b'"keyterms_prompt"' in body and b"lazy dog" in body
    assert b'"language_codes"' in body
    assert b'"llm_instruction": "Fix punctuation only."' in body

    assert isinstance(result, DictationResult)
    assert result.verbatim == "The quick brown fox jumps over the lazy dog."
    assert result.cleaned == "The quick brown fox jumps over the lazy dog."
    assert result.cleanup_error is None
    assert result.confidence == 0.996
    assert result.audio_duration_ms == 2645
    assert result.provider_request_id == "sess-abc"
    assert result.provider_processing_ms == 212.5
    assert result.round_trip_duration_ms is not None
    assert result.words == (
        DictationWord(text="The", confidence=0.99),
        DictationWord(text="fox", confidence=1.0),
    )
    assert result.warnings == ()


def test_no_options_sends_an_empty_config_part():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json=dictation_response())

    client = make_client("assemblyai", handler)
    client.dictate(audio=WAV)
    # The config part is present and empty: defaults, no fields.
    body = captured["body"]
    assert b'name="config"' in body
    assert b"{}" in body
    assert b"stt_prompt" not in body


def test_failed_cleanup_keeps_verbatim_and_warns():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=dictation_response(llm_response=None, llm_error="timeout")
        )

    client = make_client("assemblyai", handler)
    result = client.dictate(audio=WAV)
    assert result.verbatim.startswith("The quick brown fox")
    assert result.cleaned is None
    assert result.cleanup_error == "timeout"
    assert result.warnings and "timeout" in result.warnings[0]


def test_response_without_a_transcript_is_a_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"words": [], "confidence": 0.5})

    client = make_client("assemblyai", handler)
    with pytest.raises(KeyCallError) as info:
        client.dictate(audio=WAV)
    assert info.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_provider_error_surfaces_its_own_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "status": 400,
                "title": "Bad Request",
                "detail": "invalid config part: nope: Extra inputs are not permitted",
            },
        )

    client = make_client("assemblyai", handler)
    with pytest.raises(KeyCallError) as info:
        client.dictate(audio=WAV)
    # The dictation host reports {status, title, detail}, not the OpenAI
    # {error: {message}} shape; its actionable detail still reaches the
    # caller rather than a bare "unexpected status".
    assert info.value.status_code == 400
    assert "Extra inputs are not permitted" in info.value.message


@pytest.mark.anyio
async def test_async_dictation_matches_the_sync_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dictation_response())

    client = AsyncKeyCall(
        provider="assemblyai",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(handler),
    )
    result = await client.dictate(audio=WAV, keyterms=["fox"])
    assert result.verbatim.startswith("The quick brown fox")
    assert result.cleaned is not None
    await client.close()
