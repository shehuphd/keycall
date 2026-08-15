"""Bridges a browser WebSocket connection to a KeyCall realtime session.

The browser speaks a small JSON control protocol, one frame each way:

  up:    {"type": "text_turn", "text": "..."}
         {"type": "audio_chunk", "pcm_base64": "..."}
         {"type": "end_audio_turn"}
  down:  {"type": "session_started", "provider_session_id": "..." | null}
         {"type": "audio_delta", "pcm_base64": "..."}
         {"type": "transcript_delta", "text": "..."}
         {"type": "turn_complete", "usage": {...}}
         {"type": "interrupted"}
         {"type": "session_ended", "reason": "..." | null}
         {"type": "error", "code": "...", "message": "...", "retryable": bool}

Audio rides the same text-frame protocol as everything else, base64
inside the JSON envelope, rather than a second binary framing: sessions
are short and local, so a parallel binary framing on top would cost more
code than the encoding overhead it would save.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
from typing import Any, Protocol

from .._client import KeyCall
from .._errors import KeyCallError

__all__ = ["run_bridge"]


class Wire(Protocol):
    """What run_bridge needs from a browser connection. _ws.WebSocketConnection
    satisfies this; tests substitute a fake."""

    def send_text(self, text: str) -> None: ...
    def recv(self) -> str | None: ...
    def close(self) -> None: ...


def _event_json(event: Any) -> dict[str, Any]:
    if event.kind == "audio_delta":
        return {"type": "audio_delta", "pcm_base64": base64.b64encode(event.data).decode("ascii")}
    if event.kind == "transcript_delta":
        return {"type": "transcript_delta", "text": event.text}
    if event.kind == "turn_complete":
        usage = event.usage
        return {
            "type": "turn_complete",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
    if event.kind == "session_started":
        return {"type": "session_started", "provider_session_id": event.provider_session_id}
    if event.kind == "interrupted":
        return {"type": "interrupted"}
    if event.kind == "session_ended":
        return {"type": "session_ended", "reason": event.reason}
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
    if kind == "text_turn":
        text = message.get("text")
        if isinstance(text, str) and text:
            session.send_text(text)
    elif kind == "audio_chunk":
        encoded = message.get("pcm_base64")
        if isinstance(encoded, str):
            try:
                session.send_audio(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError):
                pass
    elif kind == "end_audio_turn":
        session.end_audio_turn()


def run_bridge(
    client: KeyCall,
    wire: Wire,
    *,
    model: str,
    voice: str | None,
    instructions: str | None,
) -> None:
    """Run until the browser closes the socket, the provider ends the
    session, or either side errors. Blocks the calling thread for the
    life of the session: the caller is expected to run this on its own
    connection thread, one per open Realtime tab."""
    try:
        with client.realtime(model=model, voice=voice, instructions=instructions) as session:
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
