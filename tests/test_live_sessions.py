"""Live sessions (gpt-live full duplex): dialect translation, gating, and
the session layer.

The wire dialect these fixtures exercise is PROVISIONAL: gpt-live shipped
2026-09-10 and KeyCall has not yet run a live probe against
``v1/live/sessions`` (that needs a funded, gpt-live-1-entitled key, and
the release gate is all-or-nothing over every live target). The frame
names below track OpenAI's published docs; the normalized ``LiveEvent``
taxonomy they map to is the stable surface. No test opens a socket; the
wire is faked at the transport seam, as in the realtime tests.
"""

import base64
import contextlib
import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, LiveConfig
from keycall._live import LiveSession
from keycall.adapters._live import OpenAILiveTranslator

CANARY = "sk-canary-live-key"


def make_client(provider, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("live tests must not make HTTP requests")

    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


# --- gating -----------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "xai", "deepseek", "perplexity"])
def test_live_refused_where_no_live_api_exists(provider):
    client = make_client(provider)
    with pytest.raises(KeyCallError) as excinfo:
        client.live(model="some-model")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "openai" in excinfo.value.message


def test_live_refused_for_custom_targets():
    client = KeyCall(
        provider="my-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.live(model="m")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_the_model_rides_the_session_config_not_the_path():
    client = make_client("openai")
    path, _ = client._adapter.live_plan(LiveConfig(model="gpt-live-1"))
    assert path == "/v1/live/sessions"
    assert "gpt-live-1" not in path
    assert CANARY not in path


def test_openai_live_builds_a_session_object():
    client = make_client("openai")
    session = client.live(model="gpt-live-1")
    assert isinstance(session, LiveSession)


def test_an_empty_model_is_refused():
    with pytest.raises(ValueError):
        LiveConfig(model="")


# --- the gpt-live dialect ---------------------------------------------------


def live_translator(**config):
    return OpenAILiveTranslator(LiveConfig(model="gpt-live-1", **config), provider="openai")


def test_setup_configures_a_live_session_with_voice_under_audio_output():
    (message,) = live_translator(voice="marin", instructions="Be brief.").setup_messages()
    frame = json.loads(message)
    assert frame["type"] == "session.update"
    session = frame["session"]
    assert session["type"] == "live"
    assert session["model"] == "gpt-live-1"
    assert session["instructions"] == "Be brief."
    assert session["audio"] == {"output": {"voice": "marin"}}


def test_setup_delegates_reasoning_to_the_backend_responses_model():
    (message,) = live_translator(
        backend_model="gpt-5.1",
        backend_tools=[{"type": "web_search"}],
    ).setup_messages()
    backend = json.loads(message)["session"]["backend"]
    assert backend["type"] == "responses"
    assert backend["model"] == "gpt-5.1"
    assert backend["tools"] == [{"type": "web_search"}]


def test_setup_omits_the_backend_block_when_nothing_is_delegated():
    (message,) = live_translator().setup_messages()
    assert "backend" not in json.loads(message)["session"]


def test_provider_config_merges_into_the_session():
    (message,) = live_translator(provider_config={"tracing": "auto"}).setup_messages()
    assert json.loads(message)["session"]["tracing"] == "auto"


def test_a_user_text_turn_is_item_create_plus_response_create():
    first, second = live_translator().user_text_messages("hi")
    assert json.loads(first)["type"] == "conversation.item.create"
    assert json.loads(first)["item"]["content"] == [{"type": "input_text", "text": "hi"}]
    assert json.loads(second)["type"] == "response.create"


def test_audio_chunks_append_and_the_turn_ends_with_commit():
    (chunk,) = live_translator().audio_chunk_messages(b"\x01\x02")
    assert json.loads(chunk) == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"\x01\x02").decode(),
    }
    (commit,) = live_translator().end_audio_messages()
    assert json.loads(commit)["type"] == "input_audio_buffer.commit"


def test_frames_translate_to_normalized_events():
    t = live_translator()

    (started,) = t.events_for_frame(
        json.dumps({"type": "session.created", "session": {"id": "sess_1"}})
    )
    assert started.kind == "session_started"
    assert started.provider_session_id == "sess_1"

    (interim,) = t.events_for_frame(
        json.dumps({"type": "input_audio_transcription.delta", "delta": "hel"})
    )
    assert interim.kind == "input_transcript_delta" and interim.text == "hel"

    (final,) = t.events_for_frame(
        json.dumps({"type": "input_audio_transcription.completed", "transcript": "hello"})
    )
    assert final.kind == "input_transcript_final" and final.text == "hello"

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


def test_a_cancelled_response_and_barge_in_are_interruptions():
    t = live_translator()
    (event,) = t.events_for_frame(
        json.dumps({"type": "response.done", "response": {"status": "cancelled"}})
    )
    assert event.kind == "interrupted"
    (event,) = t.events_for_frame(json.dumps({"type": "input_audio_buffer.speech_started"}))
    assert event.kind == "interrupted"


def test_the_billed_duration_is_captured_from_the_turn_usage():
    t = live_translator()
    assert t.billed_seconds is None
    t.events_for_frame(
        json.dumps(
            {
                "type": "response.done",
                "response": {"status": "completed", "usage": {"billed_seconds": 12.5}},
            }
        )
    )
    assert t.billed_seconds == 12.5


def test_the_billed_duration_is_captured_from_a_session_end_frame():
    t = live_translator()
    assert t.events_for_frame(
        json.dumps({"type": "session.ended", "session": {"duration_seconds": 42}})
    ) == []
    assert t.billed_seconds == 42.0


def test_plumbing_frames_yield_nothing_and_unknown_frames_stay_bounded():
    t = live_translator()
    assert t.events_for_frame(json.dumps({"type": "ping"})) == []
    assert t.events_for_frame(json.dumps({"type": "response.created"})) == []
    (unknown,) = t.events_for_frame(json.dumps({"type": "shiny.new.event", "blob": "x" * 9000}))
    assert unknown.kind == "unknown"
    assert unknown.provider_kind == "shiny.new.event"


def test_an_error_frame_raises_a_typed_error():
    with pytest.raises(KeyCallError) as excinfo:
        live_translator().events_for_frame(
            json.dumps({"type": "error", "error": {"message": "bad session"}})
        )
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "bad session" in excinfo.value.message


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
    config = LiveConfig(model="gpt-live-1", **config_kwargs)
    translator = OpenAILiveTranslator(config, provider=provider)
    return LiveSession(
        FakeTransport(wire),
        path="/v1/live/sessions",
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
                {
                    "type": "response.done",
                    "response": {"status": "completed", "usage": {"billed_seconds": 3.0}},
                }
            ),
            json.dumps({"type": "session.ended", "session": {"billed_seconds": 3.0}}),
        ]
    )
    with session_over(wire, instructions="Be brief.") as session:
        session.send_text("hello")
        events = list(session.events())

    # Setup went first, then the text turn (item + response.create).
    assert json.loads(wire.sent[0])["type"] == "session.update"
    assert [json.loads(m)["type"] for m in wire.sent[1:]] == [
        "conversation.item.create",
        "response.create",
    ]
    kinds = [event.kind for event in events]
    assert kinds == ["session_started", "transcript_delta", "turn_complete", "session_ended"]
    ended = events[-1]
    assert ended.reason == "1000"
    assert ended.billed_seconds == 3.0


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
