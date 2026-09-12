"""Live protocol translator for OpenAI's gpt-live (``v1/live/sessions``).

PARTIALLY PROBED. gpt-live shipped 2026-09-10 on its own full-duplex
WebSocket endpoint. A first live probe against it ran 2026-09-12 (on a
funded, gpt-live-1-entitled key): the endpoint is reachable and entitled,
the socket connects and authenticates, the model id is accepted, and the
opening handshake is corrected here from that probe. The first client
frame is ``session.start`` (not the Realtime API's ``session.update``,
which the endpoint rejects with "The first Live event must be
session.start"), and reasoning delegation rides ``delegation.responses``
(not a ``backend`` block), matching OpenAI's published config shape. The
inbound event names and the audio-buffer frames are not yet probe-
confirmed (the first probe only opened and sent a text turn); they stay
best-guesses against OpenAI's docs and the Realtime conventions, each a
single-place edit once a later probe reads them. The normalized
``LiveEvent`` taxonomy this produces is the stable surface a caller sees;
only the strings mapping to it move.

The translator turns provider frames into normalized events and caller
actions into provider messages. It never sees the credential; connection
and auth live in the transport.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .._errors import ErrorCode, KeyCallError
from .._types import (
    LiveAudioDelta,
    LiveConfig,
    LiveEvent,
    LiveInputTranscriptDelta,
    LiveInputTranscriptFinal,
    LiveInterrupted,
    LiveSessionStarted,
    LiveTranscriptDelta,
    LiveTurnComplete,
    UnknownLiveEvent,
    Usage,
)

# Frame types that are session plumbing, not caller-visible events.
# PROVISIONAL, confirm at probe.
_LIVE_PLUMBING = frozenset(
    {
        "session.updated",
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_stopped",
        "ping",
        "rate_limits.updated",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _decode_frame(payload: str | bytes, *, provider: str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        frame = json.loads(payload)
    except ValueError:
        raise KeyCallError(
            "live frame was not valid JSON",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=provider,
            operation="live",
        ) from None
    if not isinstance(frame, dict):
        raise KeyCallError(
            "live frame was not a JSON object",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=provider,
            operation="live",
        )
    return frame


class OpenAILiveTranslator:
    """The gpt-live full-duplex dialect (PROVISIONAL, see module docstring)."""

    def __init__(self, config: LiveConfig, *, provider: str) -> None:
        self._config = config
        self._provider = provider
        # Read on the session-ended event: the voice loop is billed per
        # second, and the provider reports the elapsed duration near close.
        self.billed_seconds: float | None = None

    def setup_messages(self) -> tuple[str, ...]:
        session: dict[str, Any] = {"model": self._config.model}
        if self._config.instructions is not None:
            session["instructions"] = self._config.instructions
        audio: dict[str, Any] = {}
        if self._config.voice is not None:
            audio["output"] = {"voice": self._config.voice}
        # The caller's own audio is transcribed only when the session asks
        # for it: the model's output transcript rides its audio for free,
        # but the input transcript is a separate opt-in that bills for the
        # extra recognition. Without this the LiveInputTranscript* events
        # never arrive. Not yet probe-confirmed (the first probe only sent
        # a text turn); confirm the audio.input shape at a later probe.
        if self._config.input_transcription:
            audio["input"] = {"transcription": {}}
        if audio:
            session["audio"] = audio
        # Responses delegation: gpt-live hands reasoning and tool use to a
        # separate backend Responses model, billed separately. The config
        # rides delegation.responses (OpenAI's published shape), not a
        # backend block.
        responses: dict[str, Any] = {}
        if self._config.backend_model is not None:
            responses["model"] = self._config.backend_model
        if self._config.backend_tools:
            responses["tools"] = [dict(tool) for tool in self._config.backend_tools]
        if responses:
            session["delegation"] = {"type": "responses", "responses": responses}
        if self._config.provider_config is not None:
            session.update(self._config.provider_config)
        # The first client frame on v1/live/sessions must be session.start
        # carrying the config (probe-confirmed 2026-09-12: session.update,
        # the Realtime opener, is rejected outright).
        return (json.dumps({"type": "session.start", "session": session}),)

    def user_text_messages(self, text: str) -> tuple[str, ...]:
        # gpt-live creates the input item on the delegated Responses
        # conversation: response.item.create, not the Realtime API's
        # conversation.item.create (per OpenAI's docs; not yet probe-
        # confirmed).
        return (
            json.dumps(
                {
                    "type": "response.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            ),
            json.dumps({"type": "response.create"}),
        )

    def audio_chunk_messages(self, pcm: bytes) -> tuple[str, ...]:
        encoded = base64.b64encode(pcm).decode()
        return (json.dumps({"type": "input_audio_buffer.append", "audio": encoded}),)

    def end_audio_messages(self) -> tuple[str, ...]:
        return (json.dumps({"type": "input_audio_buffer.commit"}),)

    def _record_duration(self, container: dict[str, Any]) -> None:
        for key in ("billed_seconds", "duration_seconds", "seconds"):
            value = container.get(key)
            if isinstance(value, (int, float)):
                self.billed_seconds = float(value)
                return

    def events_for_frame(self, payload: str | bytes) -> list[LiveEvent]:
        frame = _decode_frame(payload, provider=self._provider)
        frame_type = str(frame.get("type", ""))
        if frame_type in ("session.started", "session.created"):
            session = _as_dict(frame.get("session"))
            session_id = session.get("id")
            return [
                LiveSessionStarted(
                    provider_session_id=str(session_id) if session_id else None
                )
            ]
        if frame_type in (
            "input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.delta",
        ):
            return [LiveInputTranscriptDelta(text=str(frame.get("delta", "")))]
        if frame_type in (
            "input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.completed",
        ):
            return [LiveInputTranscriptFinal(text=str(frame.get("transcript", "")))]
        if frame_type == "response.output_audio.delta":
            return [LiveAudioDelta(data=base64.b64decode(frame.get("delta", "")))]
        if frame_type == "response.output_audio_transcript.delta":
            return [LiveTranscriptDelta(text=str(frame.get("delta", "")))]
        if frame_type == "input_audio_buffer.speech_started":
            return [LiveInterrupted()]
        if frame_type == "response.done":
            response = _as_dict(frame.get("response"))
            if str(response.get("status", "")) == "cancelled":
                return [LiveInterrupted()]
            usage_raw = _as_dict(response.get("usage"))
            self._record_duration(usage_raw)
            return [
                LiveTurnComplete(
                    usage=Usage(
                        input_tokens=usage_raw.get("input_tokens"),
                        output_tokens=usage_raw.get("output_tokens"),
                        total_tokens=usage_raw.get("total_tokens"),
                    )
                )
            ]
        if frame_type in ("session.done", "session.ended"):
            # The session's own final usage arrives before the socket
            # closes; capture the billed duration so LiveSessionEnded can
            # carry it. The ended event itself is emitted on socket close.
            self._record_duration(_as_dict(frame.get("session")) or frame)
            return []
        if frame_type == "error":
            error = _as_dict(frame.get("error"))
            raise KeyCallError(
                f"provider reported a live error: {str(error.get('message', ''))[:300]}",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._provider,
                operation="live",
            )
        if frame_type in _LIVE_PLUMBING:
            return []
        return [UnknownLiveEvent(provider_kind=frame_type[:100] or "unnamed")]
