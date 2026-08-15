"""The viewer's realtime bridge: WebSocket framing and the browser<->
RealtimeSession translation. No live socket here: _ws.py's framing is
tested against in-memory buffers, and _realtime_bridge.py's translation
is tested against a fake session, matching how test_reasoning_effort.py
tests adapter logic without a network call."""

from __future__ import annotations

import base64
import io
import json
import threading

from keycall import ErrorCode, KeyCallError
from keycall.viewer._realtime_bridge import run_bridge
from keycall.viewer._ws import WebSocketConnection, accept_key

# --- _ws.py: RFC 6455 framing ------------------------------------------


def test_accept_key_matches_the_rfc6455_worked_example():
    # The exact key and expected accept value from RFC 6455 section 1.3.
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def _client_frame(payload: bytes) -> bytes:
    """A masked text frame, the shape every browser sends."""
    mask = b"\x01\x02\x03\x04"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    length = len(masked)
    if length < 126:
        header = bytes([0x81, 0x80 | length])
    else:
        header = bytes([0x81, 0x80 | 126]) + length.to_bytes(2, "big")
    return header + mask + masked


def test_recv_unmasks_a_client_text_frame():
    rfile = io.BytesIO(_client_frame(b"hello"))
    wfile = io.BytesIO()
    conn = WebSocketConnection(rfile, wfile)
    assert conn.recv() == "hello"


def test_recv_handles_a_frame_longer_than_125_bytes():
    text = "x" * 500
    rfile = io.BytesIO(_client_frame(text.encode()))
    conn = WebSocketConnection(rfile, io.BytesIO())
    assert conn.recv() == text


def test_recv_answers_none_on_a_close_frame():
    close_frame = bytes([0x88, 0x80]) + b"\x00\x00\x00\x00"
    conn = WebSocketConnection(io.BytesIO(close_frame), io.BytesIO())
    assert conn.recv() is None


def test_recv_answers_none_when_the_socket_just_ends():
    conn = WebSocketConnection(io.BytesIO(b""), io.BytesIO())
    assert conn.recv() is None


def test_recv_answers_a_ping_with_a_pong_and_keeps_reading():
    ping = bytes([0x89, 0x84]) + b"\x00\x00\x00\x00" + b"ping"
    frames = ping + _client_frame(b"after")
    wfile = io.BytesIO()
    conn = WebSocketConnection(io.BytesIO(frames), wfile)
    assert conn.recv() == "after"
    sent = wfile.getvalue()
    assert sent[0] == 0x8A  # pong opcode, fin bit set


def test_send_text_frame_is_unmasked_with_a_correct_length_prefix():
    wfile = io.BytesIO()
    WebSocketConnection(io.BytesIO(b""), wfile).send_text("hi")
    sent = wfile.getvalue()
    assert sent[0] == 0x81  # fin + text opcode
    assert sent[1] == 2  # unmasked, length 2
    assert sent[2:] == b"hi"


# --- _realtime_bridge.py: browser <-> session translation ---------------


class _FakeWire:
    """``release``, when given, is set the moment the queued frames run
    out, so a paired _FakeSession can hold its own events back until
    the main loop has finished consuming every frame, instead
    of the two threads racing to decide which runs first."""

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


class _Usage:
    def __init__(self, input_tokens=3, output_tokens=4, total_tokens=7):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class _FakeSession:
    def __init__(self, events: list[_Event], wait_for: threading.Event | None = None) -> None:
        self._events = events
        self._wait_for = wait_for
        self.sent_text: list[str] = []
        self.sent_audio: list[bytes] = []
        self.ended_turns = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    def end_audio_turn(self) -> None:
        self.ended_turns += 1

    def events(self, *, timeout=None):
        if self._wait_for is not None:
            # Bounded rather than an unconditional wait: a test whose
            # main loop never exhausts the wire (it broke out earlier,
            # on an error) would otherwise block here forever.
            self._wait_for.wait(timeout=0.3)
        yield from self._events


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.realtime_calls: list[dict] = []

    def realtime(self, *, model, voice=None, instructions=None):
        self.realtime_calls.append({"model": model, "voice": voice, "instructions": instructions})
        return self._session


def test_bridge_relays_every_event_kind_as_the_matching_json_frame():
    events = [
        _Event("session_started", provider_session_id="sess_1"),
        _Event("transcript_delta", text="hi there"),
        _Event("audio_delta", data=b"\x01\x00\x02\x00"),
        _Event("interrupted"),
        _Event("turn_complete", usage=_Usage()),
        _Event("unknown", provider_kind="some.frame"),
        _Event("session_ended", reason="server closed"),
    ]
    session = _FakeSession(events)
    client = _FakeClient(session)
    wire = _FakeWire([])

    run_bridge(client, wire, model="gpt-realtime", voice="alloy", instructions="be concise")

    assert client.realtime_calls == [
        {"model": "gpt-realtime", "voice": "alloy", "instructions": "be concise"}
    ]
    kinds = [msg["type"] for msg in wire.sent]
    assert kinds == [
        "session_started", "transcript_delta", "audio_delta", "interrupted",
        "turn_complete", "unknown", "session_ended",
    ]
    assert wire.sent[0]["provider_session_id"] == "sess_1"
    assert wire.sent[1]["text"] == "hi there"
    assert base64.b64decode(wire.sent[2]["pcm_base64"]) == b"\x01\x00\x02\x00"
    assert wire.sent[4]["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    assert wire.sent[5]["provider_kind"] == "some.frame"
    assert wire.sent[6]["reason"] == "server closed"
    assert wire.closed


def test_bridge_applies_incoming_text_turn_and_end_audio_turn():
    release = threading.Event()
    session = _FakeSession([_Event("session_ended", reason=None)], wait_for=release)
    client = _FakeClient(session)
    wire = _FakeWire(
        [
            json.dumps({"type": "text_turn", "text": "hello there"}),
            json.dumps({"type": "audio_chunk", "pcm_base64": base64.b64encode(b"\x00\x01").decode()}),
            json.dumps({"type": "end_audio_turn"}),
        ],
        release=release,
    )

    run_bridge(client, wire, model="gpt-realtime", voice=None, instructions=None)

    assert session.sent_text == ["hello there"]
    assert session.sent_audio == [b"\x00\x01"]
    assert session.ended_turns == 1


def test_bridge_ignores_a_frame_it_cannot_parse():
    session = _FakeSession([_Event("session_ended", reason=None)])
    client = _FakeClient(session)
    wire = _FakeWire(["not json", json.dumps(["not", "a", "dict"])])

    run_bridge(client, wire, model="gpt-realtime", voice=None, instructions=None)

    assert session.sent_text == []
    assert session.sent_audio == []


def test_bridge_reports_a_setup_failure_as_an_error_frame():
    class _RefusingClient:
        def realtime(self, *, model, voice=None, instructions=None):
            raise KeyCallError(
                "provider 'deepseek' has no realtime API",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider="deepseek",
                operation="realtime",
            )

    wire = _FakeWire([])
    run_bridge(_RefusingClient(), wire, model="m", voice=None, instructions=None)

    assert len(wire.sent) == 1
    assert wire.sent[0] == {
        "type": "error",
        "code": "unsupported_operation",
        "message": "provider 'deepseek' has no realtime API",
        "retryable": False,
    }
    assert wire.closed


def test_bridge_reports_a_mid_session_error_and_stops():
    class _FailingSession(_FakeSession):
        def send_text(self, text: str) -> None:
            raise KeyCallError("wire closed", code=ErrorCode.NETWORK_ERROR, operation="realtime")

    # Without this, pump_provider's thread can set `stop` before the main
    # loop's first check of it even runs (this raced in practice, not just
    # in theory), skipping the one frame the assertion below depends on.
    release = threading.Event()
    session = _FailingSession([], wait_for=release)
    client = _FakeClient(session)
    wire = _FakeWire([json.dumps({"type": "text_turn", "text": "hi"})], release=release)

    run_bridge(client, wire, model="m", voice=None, instructions=None)

    assert wire.sent == [
        {"type": "error", "code": "network_error", "message": "wire closed", "retryable": False}
    ]
    assert wire.closed
