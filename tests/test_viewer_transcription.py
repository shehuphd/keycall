"""The viewer's transcription bridge: browser <-> TranscriptionSession
translation, tested against a fake session the same way
test_viewer_realtime.py tests the realtime bridge. The WebSocket framing
underneath is shared with that bridge and tested there."""

from __future__ import annotations

import base64
import json
import threading

from keycall import ErrorCode, KeyCallError
from keycall.viewer._transcription_bridge import run_transcription_bridge


class _FakeWire:
    """``release``, when given, is set the moment the queued frames run
    out, so a paired _FakeSession can hold its own events back until the
    main loop has finished consuming every frame, instead of the two
    threads racing to decide which runs first."""

    def __init__(self, incoming: list[str], release: threading.Event | None = None) -> None:
        self._incoming = list(incoming)
        self._release = release
        self.sent: list[dict] = []
        self.closed = False

    def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def recv(self) -> str | None:
        if not self._incoming:
            if self._release is not None:
                self._release.set()
            return None
        return self._incoming.pop(0)

    def close(self) -> None:
        self.closed = True


class _Event:
    def __init__(self, kind: str, **fields):
        self.kind = kind
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeSession:
    def __init__(self, events: list[_Event], wait_for: threading.Event | None = None) -> None:
        self._events = events
        self._wait_for = wait_for
        self.sent_audio: list[bytes] = []
        self.finishes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    def finish(self) -> None:
        self.finishes += 1

    def events(self, *, timeout=None):
        if self._wait_for is not None:
            # Bounded rather than an unconditional wait: a test whose main
            # loop never exhausts the wire (it broke out earlier, on an
            # error) would otherwise block here forever.
            self._wait_for.wait(timeout=0.3)
        yield from self._events


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.calls: list[dict] = []

    def transcribe_stream(self, *, model=None, sample_rate=16000):
        self.calls.append({"model": model, "sample_rate": sample_rate})
        return self._session


def test_bridge_relays_every_event_kind_as_the_matching_json_frame():
    events = [
        _Event("session_started", provider_session_id="sess_1"),
        _Event("interim_transcript", text="the quick", channel=None),
        _Event(
            "final_transcript",
            text="The quick brown fox.",
            utterance_end=True,
            confidence=0.97,
            channel=0,
        ),
        _Event("unknown", provider_kind="some.frame"),
        _Event("session_ended", reason="server closed", audio_duration_seconds=3.0),
    ]
    session = _FakeSession(events)
    client = _FakeClient(session)
    wire = _FakeWire([])

    run_transcription_bridge(client, wire, model="nova-3", sample_rate=16000)

    assert client.calls == [{"model": "nova-3", "sample_rate": 16000}]
    kinds = [msg["type"] for msg in wire.sent]
    assert kinds == [
        "session_started", "interim_transcript", "final_transcript", "unknown", "session_ended",
    ]
    assert wire.sent[0]["provider_session_id"] == "sess_1"
    assert wire.sent[1] == {"type": "interim_transcript", "text": "the quick", "channel": None}
    assert wire.sent[2] == {
        "type": "final_transcript",
        "text": "The quick brown fox.",
        "utterance_end": True,
        "confidence": 0.97,
        "channel": 0,
    }
    assert wire.sent[3]["provider_kind"] == "some.frame"
    assert wire.sent[4] == {
        "type": "session_ended",
        "reason": "server closed",
        "audio_duration_seconds": 3.0,
    }
    assert wire.closed


def test_bridge_applies_incoming_audio_and_finish():
    release = threading.Event()
    session = _FakeSession(
        [_Event("session_ended", reason=None, audio_duration_seconds=None)], wait_for=release
    )
    client = _FakeClient(session)
    wire = _FakeWire(
        [
            json.dumps({"type": "audio_chunk", "pcm_base64": base64.b64encode(b"\x00\x01").decode()}),
            json.dumps({"type": "finish"}),
        ],
        release=release,
    )

    run_transcription_bridge(client, wire, model=None, sample_rate=16000)

    assert client.calls == [{"model": None, "sample_rate": 16000}]
    assert session.sent_audio == [b"\x00\x01"]
    assert session.finishes == 1


def test_bridge_ignores_a_frame_it_cannot_parse():
    session = _FakeSession([_Event("session_ended", reason=None, audio_duration_seconds=None)])
    client = _FakeClient(session)
    wire = _FakeWire(["not json", json.dumps(["not", "a", "dict"]), json.dumps({"type": "?"})])

    run_transcription_bridge(client, wire, model=None, sample_rate=16000)

    assert session.sent_audio == []
    assert session.finishes == 0


def test_bridge_reports_a_setup_failure_as_an_error_frame():
    class _RefusingClient:
        def transcribe_stream(self, *, model=None, sample_rate=16000):
            raise KeyCallError(
                "provider 'openai' has no streaming transcription API",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider="openai",
                operation="streaming_transcription",
            )

    wire = _FakeWire([])
    run_transcription_bridge(_RefusingClient(), wire, model=None, sample_rate=16000)

    assert len(wire.sent) == 1
    assert wire.sent[0] == {
        "type": "error",
        "code": "unsupported_operation",
        "message": "provider 'openai' has no streaming transcription API",
        "retryable": False,
    }
    assert wire.closed


def test_bridge_reports_a_mid_session_error_and_stops():
    class _FailingSession(_FakeSession):
        def send_audio(self, pcm: bytes) -> None:
            raise KeyCallError(
                "wire closed", code=ErrorCode.NETWORK_ERROR, operation="streaming_transcription"
            )

    # Without this, the provider pump's thread can set `stop` before the
    # main loop's first check of it even runs, skipping the one frame the
    # assertion below depends on — the same race test_viewer_realtime.py
    # documents.
    release = threading.Event()
    session = _FailingSession([], wait_for=release)
    client = _FakeClient(session)
    wire = _FakeWire(
        [json.dumps({"type": "audio_chunk", "pcm_base64": base64.b64encode(b"\x00").decode()})],
        release=release,
    )

    run_transcription_bridge(client, wire, model=None, sample_rate=16000)

    assert wire.sent == [
        {"type": "error", "code": "network_error", "message": "wire closed", "retryable": False}
    ]
    assert wire.closed
