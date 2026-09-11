"""Live protocol translator for OpenAI's gpt-live (``v1/live/sessions``).

PROVISIONAL WIRE. gpt-live shipped 2026-09-10 on its own full-duplex
WebSocket endpoint, and KeyCall has not yet run a live probe against it
(the probe needs a funded, gpt-live-1-entitled key, and the release gate
is all-or-nothing over every live target). The session-config and event
names below are built against OpenAI's published docs and the Realtime
API's conventions; each is a single-place edit once the probe records the
endpoint's own vocabulary. The normalized ``LiveEvent`` taxonomy this
produces is the stable surface a caller sees; only the strings mapping to
it are provisional.

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
        session: dict[str, Any] = {"type": "live", "model": self._config.model}
        if self._config.instructions is not None:
            session["instructions"] = self._config.instructions
        audio: dict[str, Any] = {}
        if self._config.voice is not None:
            audio["output"] = {"voice": self._config.voice}
        # The caller's own audio is transcribed only when the session asks
        # for it: the model's output transcript rides its audio for free,
        # but the input transcript is a separate opt-in that bills for the
        # extra recognition. Without this the LiveInputTranscript* events
        # never arrive. PROVISIONAL wire, confirm the shape at probe.
        if self._config.input_transcription:
            audio["input"] = {"transcription": {}}
        if audio:
            session["audio"] = audio
        # Responses delegation: gpt-live hands reasoning and tool use to a
        # separate backend model, billed separately.
        backend: dict[str, Any] = {}
        if self._config.backend_model is not None:
            backend["model"] = self._config.backend_model
        if self._config.backend_tools:
            backend["tools"] = [dict(tool) for tool in self._config.backend_tools]
        if backend:
            session["backend"] = {"type": "responses", **backend}
        if self._config.provider_config is not None:
            session.update(self._config.provider_config)
        return (json.dumps({"type": "session.update", "session": session}),)

    def user_text_messages(self, text: str) -> tuple[str, ...]:
        return (
            json.dumps(
                {
                    "type": "conversation.item.create",
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
