"""Realtime sessions: dialect translation, gating, and the session layer.

Frame fixtures mirror live captures from 2026-08-14: OpenAI's GA
Realtime API, xAI's pre-GA Grok Voice dialect, and Gemini's
BidiGenerateContent (including its binary frames and thought parts).
No test opens a socket; the wire is faked at the transport seam.
"""

import base64
import contextlib
import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, RealtimeConfig
from keycall._realtime import RealtimeSession
from keycall.adapters._realtime import GeminiRealtimeTranslator, OpenAIRealtimeTranslator

CANARY = "sk-canary-realtime-key"


def make_client(provider, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("realtime tests must not make HTTP requests")

    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


# --- gating -----------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "perplexity", "moonshot"])
def test_realtime_refused_where_no_realtime_api_exists(provider):
    client = make_client(provider)
    with pytest.raises(KeyCallError) as excinfo:
        client.realtime(model="some-model")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    for supported in ("gemini", "openai", "xai"):
        assert supported in excinfo.value.message


def test_realtime_refused_for_custom_targets():
    client = KeyCall(
        provider="my-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.realtime(model="m")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_the_model_rides_the_path_and_the_key_never_does():
    client = make_client("openai")
    path, _ = client._adapter.realtime_plan(RealtimeConfig(model="gpt-realtime"))
    assert path == "/v1/realtime?model=gpt-realtime"
    assert CANARY not in path

    client = make_client("gemini")
    path, _ = client._adapter.realtime_plan(RealtimeConfig(model="gemini-x"))
    assert "BidiGenerateContent" in path
    assert CANARY not in path


# --- the OpenAI/xAI dialect -------------------------------------------------


def openai_translator(**config):
    return OpenAIRealtimeTranslator(
        RealtimeConfig(model="gpt-realtime", **config), provider="openai", ga_session=True
    )


def xai_translator(**config):
    return OpenAIRealtimeTranslator(
        RealtimeConfig(model="grok-voice-latest", **config), provider="xai", ga_session=False
    )


def test_ga_setup_places_voice_under_audio_output():
    (message,) = openai_translator(voice="marin", instructions="Be brief.").setup_messages()
    session = json.loads(message)["session"]
    assert session["type"] == "realtime"
    assert session["audio"] == {"output": {"voice": "marin"}}
    assert session["instructions"] == "Be brief."


def test_pre_ga_setup_places_voice_at_the_top_level():
    (message,) = xai_translator(voice="xai_ara").setup_messages()
    session = json.loads(message)["session"]
    assert session["voice"] == "xai_ara"
    assert "type" not in session


def test_provider_config_merges_into_the_session():
    (message,) = openai_translator(provider_config={"tracing": "auto"}).setup_messages()
    assert json.loads(message)["session"]["tracing"] == "auto"


def test_a_user_text_turn_is_item_create_plus_response_create():
    first, second = openai_translator().user_text_messages("hi")
    assert json.loads(first)["type"] == "conversation.item.create"
    assert json.loads(first)["item"]["content"] == [{"type": "input_text", "text": "hi"}]
    assert json.loads(second)["type"] == "response.create"


def test_audio_chunks_append_and_the_turn_ends_with_commit():
    (chunk,) = openai_translator().audio_chunk_messages(b"\x01\x02")
    assert json.loads(chunk) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"\x01\x02").decode(),
    }
    commit, respond = openai_translator().end_audio_messages()
    assert json.loads(commit)["type"] == "input_audio_buffer.commit"
    assert json.loads(respond)["type"] == "response.create"


def test_openai_frames_translate_to_normalized_events():
    t = openai_translator()

    (started,) = t.events_for_frame(
        json.dumps({"type": "session.created", "session": {"id": "sess_1"}})
    )
    assert started.kind == "session_started"
    assert started.provider_session_id == "sess_1"

    audio = base64.b64encode(b"pcm-bytes").decode()
    (delta,) = t.events_for_frame(
        json.dumps({"type": "response.output_audio.delta", "delta": audio})
    )
    assert delta.kind == "audio_delta"
    assert delta.data == b"pcm-bytes"

    (words,) = t.events_for_frame(
        json.dumps({"type": "response.output_audio_transcript.delta", "delta": "Ray"})
    )
    assert words.kind == "transcript_delta" and words.text == "Ray"

    (text,) = t.events_for_frame(
        json.dumps({"type": "response.output_text.delta", "delta": "Blue"})
    )
    assert text.kind == "transcript_delta" and text.text == "Blue"

    (done,) = t.events_for_frame(
        json.dumps(
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 17, "output_tokens": 8, "total_tokens": 25},
                },
            }
        )
    )
    assert done.kind == "turn_complete"
    assert done.usage.total_tokens == 25


def test_a_cancelled_response_is_an_interruption_not_a_turn():
    t = openai_translator()
    (event,) = t.events_for_frame(
        json.dumps({"type": "response.done", "response": {"status": "cancelled"}})
    )
    assert event.kind == "interrupted"
    (event,) = t.events_for_frame(json.dumps({"type": "input_audio_buffer.speech_started"}))
    assert event.kind == "interrupted"


def test_plumbing_frames_yield_nothing_and_unknown_frames_stay_bounded():
    t = xai_translator()
    assert t.events_for_frame(json.dumps({"type": "ping"})) == []
    assert t.events_for_frame(json.dumps({"type": "conversation.created"})) == []
    (unknown,) = t.events_for_frame(json.dumps({"type": "shiny.new.event", "blob": "x" * 9000}))
    assert unknown.kind == "unknown"
    assert unknown.provider_kind == "shiny.new.event"


def test_an_error_frame_raises_a_typed_error():
    with pytest.raises(KeyCallError) as excinfo:
        openai_translator().events_for_frame(
            json.dumps({"type": "error", "error": {"message": "bad session"}})
        )
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "bad session" in excinfo.value.message


# --- the Gemini dialect -----------------------------------------------------


def gemini_translator(**config):
    return GeminiRealtimeTranslator(
        RealtimeConfig(model="gemini-2.5-flash-native-audio-latest", **config),
        provider="gemini",
    )


def test_gemini_setup_names_the_model_and_asks_for_transcription():
    (message,) = gemini_translator(voice="Puck", instructions="Be brief.").setup_messages()
    setup = json.loads(message)["setup"]
    assert setup["model"] == "models/gemini-2.5-flash-native-audio-latest"
    assert setup["outputAudioTranscription"] == {}
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    voice = setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice == {"voiceName": "Puck"}
    assert setup["systemInstruction"] == {"parts": [{"text": "Be brief."}]}


def test_gemini_setup_keeps_an_explicit_models_prefix():
    translator = GeminiRealtimeTranslator(
        RealtimeConfig(model="models/gemini-x"), provider="gemini"
    )
    (message,) = translator.setup_messages()
    assert json.loads(message)["setup"]["model"] == "models/gemini-x"


def test_gemini_frames_translate_including_binary_ones():
    t = gemini_translator()

    # setupComplete arrives as a binary WebSocket frame (observed live).
    (started,) = t.events_for_frame(json.dumps({"setupComplete": {}}).encode())
    assert started.kind == "session_started"

    audio = base64.b64encode(b"pcm").decode()
    events = t.events_for_frame(
        json.dumps(
            {
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {"inlineData": {"mimeType": "audio/pcm", "data": audio}},
                            {"text": "planning the answer", "thought": True},
                        ]
                    }
                }
            }
        )
    )
    # The thought part is reasoning, not speech: audio only.
    assert [e.kind for e in events] == ["audio_delta"]
    assert events[0].data == b"pcm"

    (words,) = t.events_for_frame(
        json.dumps({"serverContent": {"outputTranscription": {"text": "Rayleigh"}}})
    )
    assert words.kind == "transcript_delta" and words.text == "Rayleigh"

    (done,) = t.events_for_frame(
        json.dumps(
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {
                    "promptTokenCount": 387,
                    "responseTokenCount": 51,
                    "totalTokenCount": 438,
                    "thoughtsTokenCount": 157,
                },
            }
        )
    )
    assert done.kind == "turn_complete"
    assert done.usage.total_tokens == 438
    assert done.usage.reasoning_tokens == 157

    (cut,) = t.events_for_frame(json.dumps({"serverContent": {"interrupted": True}}))
    assert cut.kind == "interrupted"

    assert t.events_for_frame(json.dumps({"serverContent": {"generationComplete": True}})) == []


def test_gemini_audio_rides_realtime_input_with_a_pcm_mime():
    (chunk,) = gemini_translator().audio_chunk_messages(b"\x00\x01")
    audio = json.loads(chunk)["realtimeInput"]["audio"]
    assert audio["mimeType"] == "audio/pcm;rate=16000"
    assert base64.b64decode(audio["data"]) == b"\x00\x01"
    (end,) = gemini_translator().end_audio_messages()
    assert json.loads(end) == {"realtimeInput": {"audioStreamEnd": True}}


# --- the session layer ------------------------------------------------------


class FakeWire:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.close_reason = "1000"

    def send(self, message):
        self.sent.append(message)

    def receive(self, timeout=None):
        if self.frames:
            return self.frames.pop(0)
        return None


class FakeTransport:
    def __init__(self, wire):
        self.wire = wire

    @contextlib.contextmanager
    def realtime_connect(self, path):
        yield self.wire


def session_over(wire, provider="openai", **config_kwargs):
    config = RealtimeConfig(model="gpt-realtime", **config_kwargs)
    translator = OpenAIRealtimeTranslator(config, provider=provider, ga_session=True)
    return RealtimeSession(
        FakeTransport(wire),
        path="/v1/realtime?model=gpt-realtime",
        translator=translator,
        provider=provider,
        config=config,
    )


def test_a_session_configures_streams_events_and_reports_the_close():
    wire = FakeWire(
        [
            json.dumps({"type": "session.created", "session": {"id": "s1"}}),
            json.dumps({"type": "response.output_audio_transcript.delta", "delta": "Hi"}),
            json.dumps(
                {"type": "response.done", "response": {"status": "completed", "usage": {}}}
            ),
        ]
    )
    with session_over(wire, instructions="Be brief.") as session:
        session.send_text("hello")
        kinds = [event.kind for event in session.events()]

    # Setup went first, then the text turn (item + response.create).
    assert json.loads(wire.sent[0])["type"] == "session.update"
    assert [json.loads(m)["type"] for m in wire.sent[1:]] == [
        "conversation.item.create",
        "response.create",
    ]
    assert kinds == ["session_started", "transcript_delta", "turn_complete", "session_ended"]


def test_provider_config_use_is_reported_with_a_warning():
    wire = FakeWire([])
    with pytest.warns(UserWarning, match="provider_config"), session_over(
        wire, provider_config={"tracing": "auto"}
    ):
        pass


def test_a_session_outside_its_context_refuses():
    session = session_over(FakeWire([]))
    with pytest.raises(RuntimeError):
        session.send_text("hi")
    with pytest.raises(RuntimeError):
        next(session.events())


def test_the_realtime_url_is_host_rooted_and_headers_carry_the_key():
    client = make_client("gemini")
    transport = client._transport
    url = transport._realtime_url("/ws/google.ai.generativelanguage.v1beta.Bidi")
    # The base URL's /v1beta prefix must not stack onto the WS path.
    assert url == "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.Bidi"
    headers = transport._realtime_headers()
    assert headers["x-goog-api-key"] == CANARY
    assert "Content-Type" not in headers
