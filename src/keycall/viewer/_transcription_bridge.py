"""Bridges a browser WebSocket connection to a KeyCall streaming
transcription session.

The browser speaks a small JSON control protocol, one frame each way:

  up:    {"type": "audio_chunk", "pcm_base64": "..."}
         {"type": "finish"}
  down:  {"type": "session_started", "provider_session_id": "..." | null}
         {"type": "interim_transcript", "text": "...", "channel": 0 | null}
         {"type": "final_transcript", "text": "...", "utterance_end": bool,
          "confidence": 0.98 | null, "channel": 0 | null}
         {"type": "session_ended", "reason": "..." | null,
          "audio_duration_seconds": 3.0 | null}
         {"type": "error", "code": "...", "message": "...", "retryable": bool}

Audio rides base64 inside JSON text frames for the same reason the
realtime bridge's does: sessions are short and local, and a second,
binary framing would cost more code than the encoding overhead it saves.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
from typing import Any, Protocol

from .._client import KeyCall
from .._errors import KeyCallError

__all__ = ["run_transcription_bridge"]


class Wire(Protocol):
    """What run_transcription_bridge needs from a browser connection.
    _ws.WebSocketConnection satisfies this; tests substitute a fake."""

    def send_text(self, text: str) -> None: ...
    def recv(self) -> str | None: ...
    def close(self) -> None: ...


def _event_json(event: Any) -> dict[str, Any]:
    if event.kind == "interim_transcript":
        return {"type": "interim_transcript", "text": event.text, "channel": event.channel}
    if event.kind == "final_transcript":
        return {
            "type": "final_transcript",
            "text": event.text,
            "utterance_end": event.utterance_end,
            "confidence": event.confidence,
            "channel": event.channel,
        }
    if event.kind == "session_started":
        return {"type": "session_started", "provider_session_id": event.provider_session_id}
    if event.kind == "session_ended":
        return {
            "type": "session_ended",
            "reason": event.reason,
            "audio_duration_seconds": event.audio_duration_seconds,
        }
    return {"type": "unknown", "provider_kind": event.provider_kind}


def _error_json(error: KeyCallError) -> dict[str, Any]:
    return {
        "type": "error",
        "code": error.code.value,
        "message": error.message,
        "retryable": error.retryable,
    }


def _handle_incoming(session: Any, frame: str) -> None:
    """Apply one browser control frame to the session. A frame this
    bridge doesn't recognize, or can't parse, is dropped rather than
    ending the connection over it."""
    try:
        message = json.loads(frame)
    except ValueError:
        return
    if not isinstance(message, dict):
        return
    kind = message.get("type")
    if kind == "audio_chunk":
        encoded = message.get("pcm_base64")
        if isinstance(encoded, str):
            try:
                session.send_audio(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError):
                pass
    elif kind == "finish":
        session.finish()


def run_transcription_bridge(
    client: KeyCall,
    wire: Wire,
    *,
    model: str | None,
    sample_rate: int,
) -> None:
    """Run until the browser closes the socket, the provider ends the
    session, or either side errors. Blocks the calling thread for the
    life of the session, same as the realtime bridge: the caller runs
    this on its own connection thread, one per open tab."""
    try:
        with client.transcribe_stream(model=model, sample_rate=sample_rate) as session:
            stop = threading.Event()

            def pump_provider() -> None:
                try:
                    for event in session.events(timeout=None):
                        wire.send_text(json.dumps(_event_json(event)))
                        if event.kind == "session_ended":
                            break
                except KeyCallError as error:
                    wire.send_text(json.dumps(_error_json(error)))
                finally:
                    stop.set()

            provider_thread = threading.Thread(target=pump_provider, daemon=True)
            provider_thread.start()
            try:
                while not stop.is_set():
                    frame = wire.recv()
                    if frame is None:
                        break
                    try:
                        _handle_incoming(session, frame)
                    except KeyCallError as error:
                        wire.send_text(json.dumps(_error_json(error)))
                        break
            finally:
                stop.set()
                provider_thread.join(timeout=5)
    except KeyCallError as error:
        wire.send_text(json.dumps(_error_json(error)))
    finally:
        wire.close()
