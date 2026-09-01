import base64
import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, ModelCategory

CANARY = "sk-canary-elevenlabs-key-000"

MODELS_PAYLOAD = [
    {"model_id": "eleven_flash_v2_5", "name": "Flash v2.5", "can_do_text_to_speech": True},
    {"model_id": "eleven_v3", "name": "Eleven v3", "can_do_text_to_speech": True},
    {
        "model_id": "eleven_english_sts_v2",
        "name": "English STS",
        "can_do_text_to_speech": False,
        "can_do_voice_conversion": True,
    },
]

VOICES_PAYLOAD = {
    "voices": [
        {"voice_id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "category": "premade"},
        {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "category": "premade"},
    ]
}


def make_client(handler, provider="elevenlabs"):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def standard_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json=MODELS_PAYLOAD)
    if request.url.path == "/v1/voices":
        return httpx.Response(200, json=VOICES_PAYLOAD)
    if request.url.path.startswith("/v1/text-to-speech/"):
        return httpx.Response(200, content=b"ID3fake-mp3-bytes", headers={"content-type": "audio/mpeg"})
    return httpx.Response(404, json={"detail": {"message": "no such path"}})


# --- listing and classification ---------------------------------------------


def test_auth_header_is_xi_api_key_bare():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["xi"] = request.headers.get("xi-api-key")
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json=MODELS_PAYLOAD)

    client = make_client(handler)
    client.list_models(categories={ModelCategory.SPEECH_GENERATION}, refresh=True)
    client.close()
    assert captured["xi"] == CANARY
    assert captured["authorization"] is None


def test_models_classify_from_capability_booleans_and_catalog():
    client = make_client(standard_handler)
    discovery = client.list_models(categories={ModelCategory.UNKNOWN,
                                               ModelCategory.SPEECH_GENERATION,
                                               ModelCategory.TRANSCRIPTION}, refresh=True)
    client.close()
    by_id = {m.id: m for m in discovery.models}
    assert ModelCategory.SPEECH_GENERATION in by_id["eleven_flash_v2_5"].categories
    assert by_id["eleven_flash_v2_5"].classification_source == "provider_metadata"
    # A voice-conversion model drives no KeyCall operation and says so.
    sts = by_id["eleven_english_sts_v2"]
    assert sts.categories == frozenset({ModelCategory.UNKNOWN})
    assert any("voice-conversion" in w for w in sts.warnings)
    # The streaming STT model rides the catalog, not /v1/models.
    scribe = by_id["scribe_v2_realtime"]
    assert ModelCategory.TRANSCRIPTION in scribe.categories
    assert scribe.classification_source == "keycall_catalog"


def test_underscoped_key_reads_as_permission_not_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": {
            "type": "authentication_error",
            "message": "The API key you used is missing the permission models_read to execute this operation.",
            "status": "missing_permissions",
        }})

    client = make_client(handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    client.close()
    assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
    assert "elevenlabs.io/app/settings/api-keys" in excinfo.value.message


# --- voices ------------------------------------------------------------------


def test_list_voices_live_endpoint():
    client = make_client(standard_handler)
    voices = client.list_voices()
    client.close()
    assert [v.id for v in voices] == ["CwhRBWXzGAHq8TQ4Fs17", "EXAVITQu4vr4xnSDxMaL"]
    assert voices[0].name == "Roger"
    assert voices[0].provider == "elevenlabs"
    assert voices[0].description == "premade"


def test_list_voices_catalog_backed_makes_no_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("catalog-backed voices must not touch the network")

    for provider, expected_count in (("openai", 13), ("gemini", 30)):
        client = KeyCall(
            provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
        )
        voices = client.list_voices()
        client.close()
        assert len(voices) == expected_count
        assert all(v.provider == provider for v in voices)


def test_openai_voice_model_scoping_is_recorded():
    client = KeyCall(
        provider="openai", api_key=CANARY,
        httpx_transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    voices = {v.id: v for v in client.list_voices()}
    client.close()
    assert voices["marin"].models == ("gpt-4o-mini-tts",)
    assert voices["alloy"].models is None


def test_list_voices_refused_where_speech_is_not_supported():
    client = KeyCall(
        provider="deepseek", api_key=CANARY,
        httpx_transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.list_voices()
    client.close()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


# --- speech generation -------------------------------------------------------


def test_speech_without_voice_refuses_naming_the_choices():
    client = make_client(standard_handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="eleven_flash_v2_5", text="Hello.")
    client.close()
    assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE
    assert "Roger" in excinfo.value.message
    assert "CwhRBWXzGAHq8TQ4Fs17" in excinfo.value.message


def test_speech_routes_voice_in_path_and_returns_audio():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"ID3fake", headers={"content-type": "audio/mpeg"})

    client = make_client(handler)
    result = client.generate_speech(
        model="eleven_flash_v2_5", text="Hello.", voice="CwhRBWXzGAHq8TQ4Fs17"
    )
    client.close()
    assert "/v1/text-to-speech/CwhRBWXzGAHq8TQ4Fs17" in captured["url"]
    assert CANARY not in captured["url"]
    assert captured["body"] == {"text": "Hello.", "model_id": "eleven_flash_v2_5"}
    part = result.parts[0]
    assert part.media_type == "audio/mpeg"
    assert base64.b64decode(part.base64_data) == b"ID3fake"


def test_speech_validation_error_list_shape_reads_as_sentence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [
            {"type": "missing", "loc": ["body", "text"], "msg": "Field required", "input": None}
        ]})

    client = make_client(handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_speech(model="eleven_flash_v2_5", text="Hi.", voice="v-1234567890")
    client.close()
    assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE
    assert "body.text: Field required" in excinfo.value.message


def test_text_generation_refused():
    from keycall import Message, TextInput

    client = make_client(standard_handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="eleven_flash_v2_5",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    client.close()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


# --- streaming transcription -------------------------------------------------


def plan_for(sample_rate=16000, model=None):
    from keycall._registry import resolve_provider
    from keycall._types import TranscriptionConfig
    from keycall.adapters._elevenlabs import ElevenLabsAdapter

    adapter = ElevenLabsAdapter(resolve_provider("elevenlabs"))
    return adapter.transcription_plan(TranscriptionConfig(sample_rate=sample_rate, model=model))


def test_transcription_plan_query_params_and_model():
    path, _translator = plan_for(model="scribe_v2_realtime")
    assert "audio_format=pcm_16000" in path
    assert "commit_strategy=vad" in path
    assert "include_timestamps=true" in path
    assert "model_id=scribe_v2_realtime" in path
    assert CANARY not in path


def test_transcription_plan_rejects_unsupported_sample_rate():
    with pytest.raises(KeyCallError) as excinfo:
        plan_for(sample_rate=11025)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "16000" in excinfo.value.message


def test_translator_encodes_audio_as_json_message():
    _, translator = plan_for()
    frame = translator.encode_audio(b"\x01\x02")
    message = json.loads(frame)
    assert message == {
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(b"\x01\x02").decode(),
        "sample_rate": 16000,
        "commit": False,
    }
    (finish,) = translator.finish_messages()
    assert json.loads(finish)["commit"] is True


def test_translator_event_mapping_and_session_over():
    from keycall import FinalTranscript, InterimTranscript, TranscriptionSessionStarted

    _, translator = plan_for()
    (started,) = translator.events_for_frame(
        json.dumps({"message_type": "session_started", "session_id": "s-1"})
    )
    assert isinstance(started, TranscriptionSessionStarted)
    assert started.provider_session_id == "s-1"

    (interim,) = translator.events_for_frame(
        json.dumps({"message_type": "partial_transcript", "text": "Hel"})
    )
    assert isinstance(interim, InterimTranscript)

    # The text-only committed frame yields nothing; its timestamped twin
    # carries the final, with seconds converted to ms and spacing dropped.
    assert translator.events_for_frame(
        json.dumps({"message_type": "committed_transcript", "text": "Hello."})
    ) == []
    (final,) = translator.events_for_frame(json.dumps({
        "message_type": "committed_transcript_with_timestamps",
        "text": "Hello.",
        "words": [
            {"text": "Hello.", "start": 0.059, "end": 0.4, "type": "word", "logprob": -0.63},
            {"text": " ", "start": 0.4, "end": 0.41, "type": "spacing"},
        ],
    }))
    assert isinstance(final, FinalTranscript)
    assert final.words[0].start_ms == pytest.approx(59.0)
    assert final.words[0].end_ms == pytest.approx(400.0)
    assert final.words[0].confidence == pytest.approx(-0.63)
    assert len(final.words) == 1
    # Mid-session finals never end the session; the one answering finish does.
    assert translator.session_over is False
    translator.finish_messages()
    translator.events_for_frame(json.dumps({
        "message_type": "committed_transcript_with_timestamps", "text": "Bye.", "words": [],
    }))
    assert translator.session_over is True


def test_translator_auth_error_event_raises_typed():
    _, translator = plan_for()
    with pytest.raises(KeyCallError) as excinfo:
        translator.events_for_frame(
            json.dumps({"message_type": "auth_error", "error": "You must be authenticated"})
        )
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY


def test_session_sends_text_frames_and_ends_after_finish():
    from keycall._transcription import TranscriptionSession
    from keycall._types import TranscriptionSessionEnded

    _, translator = plan_for()
    sent: list[tuple[str, object]] = []

    class FakeWire:
        close_reason = None

        def __init__(self) -> None:
            self.queue = []

        def send(self, message):
            sent.append(("text", message))

        def send_bytes(self, payload):
            sent.append(("bytes", payload))

        def receive(self, timeout):
            return self.queue.pop(0) if self.queue else None

    class FakeCM:
        def __init__(self, wire):
            self.wire = wire

        def __enter__(self):
            return self.wire

        def __exit__(self, *exc):
            return None

    wire = FakeWire()

    class FakeTransport:
        def realtime_connect(self, path):
            return FakeCM(wire)

    session = TranscriptionSession(FakeTransport(), path="/ws", translator=translator)
    with session:
        session.send_audio(b"\x00\x01")
        assert sent[-1][0] == "text", "elevenlabs audio must ride a JSON text frame"
        session.finish()
        wire.queue = [json.dumps({
            "message_type": "committed_transcript_with_timestamps", "text": "Hi.", "words": [],
        })]
        events = list(session.events())
    assert isinstance(events[-1], TranscriptionSessionEnded)
    assert events[-1].reason == "client finished"
    assert events[-1].audio_duration_seconds is None
