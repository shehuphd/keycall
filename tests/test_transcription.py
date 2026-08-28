"""Streaming transcription: frame translation, session lifecycle, gates.

Fixture frames mirror the live sessions captured 2026-08-28: AssemblyAI's
Begin/Turn/Termination vocabulary with integer-ms word timings, and
Deepgram's Results/Metadata vocabulary with float-second timings and the
is_final/speech_final split.
"""

import json

import httpx
import pytest

from keycall import (
    ErrorCode,
    KeyCall,
    KeyCallError,
    Message,
    ModelCategory,
    TextInput,
)
from keycall._transcription import TranscriptionSession
from keycall._types import TranscriptionConfig
from keycall.adapters._stt import AssemblyAITranslator, DeepgramTranslator

CANARY = "sk-canary-transcription-key"


def make_client(provider, handler=None, **kwargs):
    transport = httpx.MockTransport(handler) if handler else None
    return KeyCall(provider=provider, api_key=CANARY, httpx_transport=transport, **kwargs)


def kinds(events):
    return [event.kind for event in events]


# --- gates ------------------------------------------------------------------


def test_transcribe_stream_rejected_for_llm_providers():
    client = make_client("openai")
    with pytest.raises(KeyCallError) as excinfo:
        client.transcribe_stream()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "assemblyai" in excinfo.value.message
    assert "deepgram" in excinfo.value.message
    client.close()


def test_generate_text_rejected_for_stt_providers():
    client = make_client("deepgram")
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="nova-3",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "transcribe_stream" in excinfo.value.message
    client.close()


def test_sample_rate_floor_enforced():
    with pytest.raises(ValueError):
        TranscriptionConfig(sample_rate=4000)


# --- model listing: catalog behind a credential check -----------------------


def test_stt_model_listing_validates_credential_and_reads_catalog():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"transcripts": []})

    client = make_client("assemblyai", handler)
    discovery = client.list_models(categories={ModelCategory.TRANSCRIPTION}, refresh=True)
    client.close()
    assert captured["path"] == "/v2/transcript"
    assert captured["auth"] == CANARY  # bare key, no Bearer prefix
    assert [m.id for m in discovery.models] == ["universal-3-5-pro"]
    assert discovery.models[0].classification_source == "keycall_catalog"


def test_deepgram_uses_token_auth_scheme():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"subject": "x"})

    client = make_client("deepgram", handler)
    discovery = client.list_models(categories={ModelCategory.TRANSCRIPTION}, refresh=True)
    client.close()
    assert captured["auth"] == f"Token {CANARY}"
    assert [m.id for m in discovery.models] == ["nova-3", "nova-2"]


def test_bad_stt_key_is_typed_invalid_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Authentication error"})

    client = make_client("assemblyai", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(categories={ModelCategory.TRANSCRIPTION}, refresh=True)
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY
    client.close()


# --- transcription_plan: paths carry config, never the credential -----------


def test_assemblyai_plan_full_wss_url_with_sample_rate():
    client = make_client("assemblyai")
    path, translator = client._adapter.transcription_plan(
        TranscriptionConfig(sample_rate=44100)
    )
    client.close()
    assert path == "wss://streaming.assemblyai.com/v3/ws?sample_rate=44100"
    assert isinstance(translator, AssemblyAITranslator)
    assert CANARY not in path


def test_assemblyai_plan_optional_model():
    client = make_client("assemblyai")
    path, _ = client._adapter.transcription_plan(
        TranscriptionConfig(model="universal-3-5-pro")
    )
    client.close()
    assert "speech_model=universal-3-5-pro" in path


def test_deepgram_plan_host_rooted_with_stream_params():
    client = make_client("deepgram")
    path, translator = client._adapter.transcription_plan(TranscriptionConfig())
    client.close()
    assert path.startswith("/v1/listen?")
    assert "encoding=linear16" in path
    assert "sample_rate=16000" in path
    assert "interim_results=true" in path
    assert "punctuate=true" in path
    assert isinstance(translator, DeepgramTranslator)
    assert CANARY not in path


def test_deepgram_plan_optional_model():
    client = make_client("deepgram")
    path, _ = client._adapter.transcription_plan(TranscriptionConfig(model="nova-3"))
    client.close()
    assert "model=nova-3" in path


# --- AssemblyAI frame translation -------------------------------------------


ASSEMBLYAI_BEGIN = json.dumps(
    {"type": "Begin", "id": "sess-1", "expires_at": 1787957471}
)
ASSEMBLYAI_INTERIM = json.dumps(
    {
        "type": "Turn",
        "turn_order": 0,
        "end_of_turn": False,
        "transcript": "Hello, this",
        "end_of_turn_confidence": 0.0,
        "utterance": "",
    }
)
ASSEMBLYAI_FINAL = json.dumps(
    {
        "type": "Turn",
        "turn_order": 0,
        "end_of_turn": True,
        "transcript": "Hello, this is a test.",
        "end_of_turn_confidence": 1.0,
        "utterance": "Hello, this is a test.",
        "words": [
            {"start": 0, "end": 451, "text": "Hello,", "confidence": 0.95, "word_is_final": True},
            {"start": 466, "end": 767, "text": "this", "confidence": 0.99, "word_is_final": True},
        ],
    }
)
ASSEMBLYAI_TERMINATION = json.dumps(
    {"type": "Termination", "audio_duration_seconds": 9, "session_duration_seconds": 13}
)


def test_assemblyai_begin_becomes_session_started():
    (event,) = AssemblyAITranslator().events_for_frame(ASSEMBLYAI_BEGIN)
    assert event.kind == "session_started"
    assert event.provider_session_id == "sess-1"


def test_assemblyai_growing_turn_is_interim():
    (event,) = AssemblyAITranslator().events_for_frame(ASSEMBLYAI_INTERIM)
    assert event.kind == "interim_transcript"
    assert event.text == "Hello, this"
    assert event.channel is None


def test_assemblyai_end_of_turn_is_final_with_ms_words():
    (event,) = AssemblyAITranslator().events_for_frame(ASSEMBLYAI_FINAL)
    assert event.kind == "final_transcript"
    assert event.text == "Hello, this is a test."
    assert event.utterance_end is True
    assert event.confidence is None  # AssemblyAI scores per-word only
    first = event.words[0]
    assert (first.text, first.start_ms, first.end_ms) == ("Hello,", 0.0, 451.0)
    assert first.confidence == 0.95


def test_assemblyai_termination_holds_duration_for_session_end():
    translator = AssemblyAITranslator()
    assert translator.events_for_frame(ASSEMBLYAI_TERMINATION) == []
    assert translator.audio_duration_seconds == 9.0


def test_assemblyai_speech_started_is_plumbing():
    frame = json.dumps({"type": "SpeechStarted", "timestamp": 0, "confidence": 0.8})
    assert AssemblyAITranslator().events_for_frame(frame) == []


def test_assemblyai_unknown_frame_surfaces_bounded():
    (event,) = AssemblyAITranslator().events_for_frame(json.dumps({"type": "Mystery"}))
    assert event.kind == "unknown"
    assert event.provider_kind == "Mystery"


# --- Deepgram frame translation ---------------------------------------------


def deepgram_results(*, transcript, is_final, speech_final, words=()):
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {
                "alternatives": [
                    {"transcript": transcript, "confidence": 0.98, "words": list(words)}
                ]
            },
        }
    )


def test_deepgram_interim_results():
    (event,) = DeepgramTranslator().events_for_frame(
        deepgram_results(transcript="Hello. This is", is_final=False, speech_final=False)
    )
    assert event.kind == "interim_transcript"
    assert event.text == "Hello. This is"
    assert event.channel == 0


def test_deepgram_final_converts_seconds_to_ms_and_prefers_punctuated():
    (event,) = DeepgramTranslator().events_for_frame(
        deepgram_results(
            transcript="Hello. This is a test.",
            is_final=True,
            speech_final=True,
            words=[
                {
                    "word": "hello",
                    "start": 0.1178125,
                    "end": 0.35343748,
                    "confidence": 0.9,
                    "punctuated_word": "Hello.",
                }
            ],
        )
    )
    assert event.kind == "final_transcript"
    assert event.utterance_end is True
    assert event.confidence == 0.98
    assert event.channel == 0
    word = event.words[0]
    assert word.text == "Hello."  # punctuated form wins
    assert word.start_ms == pytest.approx(117.8125)
    assert word.end_ms == pytest.approx(353.43748)


def test_deepgram_is_final_without_speech_final_keeps_utterance_open():
    (event,) = DeepgramTranslator().events_for_frame(
        deepgram_results(transcript="partial stretch", is_final=True, speech_final=False)
    )
    assert event.kind == "final_transcript"
    assert event.utterance_end is False


def test_deepgram_finalized_silence_is_no_event():
    assert (
        DeepgramTranslator().events_for_frame(
            deepgram_results(transcript="", is_final=True, speech_final=True)
        )
        == []
    )


def test_deepgram_metadata_holds_duration_for_session_end():
    translator = DeepgramTranslator()
    frame = json.dumps({"type": "Metadata", "request_id": "r-1", "duration": 9.370625})
    assert translator.events_for_frame(frame) == []
    assert translator.audio_duration_seconds == pytest.approx(9.370625)


def test_deepgram_unknown_frame_surfaces_bounded():
    (event,) = DeepgramTranslator().events_for_frame(json.dumps({"type": "Mystery"}))
    assert event.kind == "unknown"
    assert event.provider_kind == "Mystery"


# --- session lifecycle over a stub wire -------------------------------------


class _StubWire:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent_text = []
        self.sent_bytes = []
        self.close_reason = "1000: done"

    def send(self, message):
        self.sent_text.append(message)

    def send_bytes(self, data):
        self.sent_bytes.append(data)

    def receive(self, timeout=None):
        return self._frames.pop(0) if self._frames else None


class _StubConnect:
    def __init__(self, wire):
        self._wire = wire

    def __enter__(self):
        return self._wire

    def __exit__(self, *exc_info):
        return None


class _StubTransport:
    def __init__(self, wire):
        self._wire = wire
        self.connected_path = None

    def realtime_connect(self, path):
        self.connected_path = path
        return _StubConnect(self._wire)


def test_session_lifecycle_audio_finish_and_terminal_event():
    wire = _StubWire([ASSEMBLYAI_BEGIN, ASSEMBLYAI_INTERIM, ASSEMBLYAI_FINAL, ASSEMBLYAI_TERMINATION])
    transport = _StubTransport(wire)
    translator = AssemblyAITranslator()

    with TranscriptionSession(transport, path="wss://x/ws", translator=translator) as session:
        session.send_audio(b"\x00\x01")
        session.finish()
        events = list(session.events())

    assert transport.connected_path == "wss://x/ws"
    assert wire.sent_bytes == [b"\x00\x01"]  # audio rides binary frames
    assert wire.sent_text == [json.dumps({"type": "Terminate"})]
    assert kinds(events) == [
        "session_started",
        "interim_transcript",
        "final_transcript",
        "session_ended",
    ]
    ended = events[-1]
    assert ended.audio_duration_seconds == 9.0
    assert ended.reason == "1000: done"


def test_session_dropped_connection_ends_without_duration():
    # No Termination frame arrived: the duration stays None, the reason
    # carries the close, and the session still ends in good order.
    wire = _StubWire([ASSEMBLYAI_BEGIN])
    wire.close_reason = "1006: abnormal closure"
    session = TranscriptionSession(
        _StubTransport(wire), path="wss://x/ws", translator=AssemblyAITranslator()
    )
    with session:
        events = list(session.events())
    assert kinds(events) == ["session_started", "session_ended"]
    assert events[-1].audio_duration_seconds is None
    assert events[-1].reason == "1006: abnormal closure"


def test_session_requires_context_manager():
    session = TranscriptionSession(
        _StubTransport(_StubWire([])), path="wss://x/ws", translator=AssemblyAITranslator()
    )
    with pytest.raises(RuntimeError):
        session.send_audio(b"\x00")
