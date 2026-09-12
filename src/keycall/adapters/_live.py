"""Live protocol translator for OpenAI's gpt-live (``v1/live/sessions``).

PARTIALLY PROBED. gpt-live shipped 2026-09-10 on its own full-duplex
WebSocket endpoint. A first live probe against it ran 2026-09-12 (on a
funded, gpt-live-1-entitled key): the endpoint is reachable and entitled,
the socket connects and authenticates, the model id is accepted, and the
opening handshake is corrected here from that probe. The first client
frame is ``session.start`` (not the Realtime API's ``session.update``,
which the endpoint rejects with "The first Live event must be
session.start"), and reasoning delegation rides ``delegation.responses``
(not a ``backend`` block), matching OpenAI's published config shape.

Probe round 2 (2026-09-12) tapped the raw frames: the delegated backend's
Responses stream arrives nested inside a ``response.event`` envelope (one
backend event per envelope, under ``.event``), so this translator unwraps
it and maps the inner Responses types, the backend's ``response.output_
text.delta`` being the interviewer's words. That probe voiced nothing on a
text-injected turn. Rounds 3 and 4 established that gpt-live has no
``output_modalities`` key at all (rejected both nested under a per-response
``response`` wrapper and as a top-level session field), so none is sent:
audio output is governed by the ``audio.output`` block, and ``response.
create`` carries no arguments. A text turn stays text-only under this
wire, so voicing is expected on an audio-input turn (``send_audio``).
Probe round 5 had the endpoint enumerate its whole client-event
allowlist: caller audio is ``session.input_audio.append`` (not the
Realtime API's ``input_audio_buffer.append``) and there is no commit verb,
so ``end_audio_turn`` uses ``session.input_audio.mute``.

Probe round 6 ran a full voiced turn and captured the inbound side. On the
audio path the model's output is top-level ``session.*`` frames, not the
``response.event`` envelope (which is the text path): ``session.output_
audio.delta`` is the voice, ``session.output_transcript.delta`` the model's
own words, and ``session.input_transcript.delta`` the caller's words (which
stream without asking). There is no turn-complete or session-ended frame;
billing is incremental through ``session.usage.updated`` (its
``usage.seconds`` is the cumulative elapsed cost), surfaced as
``LiveUsageUpdated`` and carried onto ``LiveSessionEnded`` at socket close.
The session is ended deliberately with ``session.close`` on context exit. The normalized ``LiveEvent`` taxonomy this produces is the stable
surface a caller sees; only the strings mapping to it move.

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
    LiveUsageUpdated,
    UnknownLiveEvent,
    Usage,
)

# Top-level (session-layer) frame types that are plumbing, not caller-
# visible events. Probe round 2 (2026-09-12) confirmed session.delegation.
# created arrives at the top level when the backend delegation opens.
_LIVE_PLUMBING = frozenset(
    {
        "session.updated",
        "session.delegation.created",
        "session.input_audio.muted",
        "session.input_audio.unmuted",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_stopped",
        "ping",
        "rate_limits.updated",
    }
)

# Event types inside the delegated Responses stream (unwrapped from a
# response.event envelope) that are generation lifecycle, not caller-
# visible. Observed in probe round 2 (2026-09-12).
_RESPONSES_PLUMBING = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
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
        # gpt-live has no ``output_modalities`` key: probe rounds 3 and 4
        # (2026-09-12) had it rejected both nested under a per-response
        # ``response`` wrapper and as a top-level session field, each
        # erroring the whole session. Audio output is governed by the
        # ``audio.output`` block alone (the voice above). A text-injected
        # turn came back text-only under this wire (round 2), so voicing is
        # expected on an audio-input turn, which is the path still to be
        # probed; no modality field is sent.
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
        # response.create takes no arguments here: the output modality is
        # set once on the session config (see setup_messages). gpt-live
        # rejects the Realtime API's ``response`` wrapper on this frame
        # (probe round 3, 2026-09-12).
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
        # gpt-live's own namespace, not the Realtime API's
        # input_audio_buffer.*: the client-event allowlist the endpoint
        # enumerated (probe round 5, 2026-09-12) names session.input_audio.
        # append for caller audio.
        encoded = base64.b64encode(pcm).decode()
        return (json.dumps({"type": "session.input_audio.append", "audio": encoded}),)

    def end_audio_messages(self) -> tuple[str, ...]:
        # No commit verb exists in gpt-live's vocabulary. The model
        # endpoints the caller's turn itself (server VAD); this closes the
        # turn by hand with session.input_audio.mute, the allowlist's
        # end-of-input signal (probe round 5, 2026-09-12).
        return (json.dumps({"type": "session.input_audio.mute"}),)

    def close_messages(self) -> tuple[str, ...]:
        # Close the session gracefully on the way out. gpt-live keeps the
        # socket open on its own timer and sends no session-ended frame, so
        # this is how the session is ended deliberately (session.close is on
        # the client-event allowlist, probe round 5).
        return (json.dumps({"type": "session.close"}),)

    def _record_duration(self, container: dict[str, Any]) -> None:
        for key in ("billed_seconds", "duration_seconds", "seconds"):
            value = container.get(key)
            if isinstance(value, (int, float)):
                self.billed_seconds = float(value)
                return

    def _responses_stream_events(self, inner: dict[str, Any]) -> list[LiveEvent]:
        """Map one event from the delegated Responses stream, unwrapped from
        a response.event envelope. The backend model's text output is the
        interviewer's words (there is no separate transcript frame), so it
        becomes a transcript_delta; response.completed closes the turn and
        carries the usage."""
        inner_type = str(inner.get("type", ""))
        if inner_type == "response.output_text.delta":
            return [LiveTranscriptDelta(text=str(inner.get("delta", "")))]
        if inner_type == "response.output_audio.delta":
            return [LiveAudioDelta(data=base64.b64decode(inner.get("delta", "")))]
        if inner_type == "response.output_audio_transcript.delta":
            return [LiveTranscriptDelta(text=str(inner.get("delta", "")))]
        if inner_type == "response.completed":
            response = _as_dict(inner.get("response"))
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
        if inner_type in _RESPONSES_PLUMBING:
            return []
        # Nested unknowns keep the envelope prefix so a reader can tell a
        # backend-stream frame apart from a top-level one.
        return [UnknownLiveEvent(provider_kind=f"response.event/{inner_type}"[:100])]

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
        # gpt-live's own audio-turn output frames, all top-level session.*
        # (probe round 6, 2026-09-12): the model's voice, its own words, and
        # the caller's transcribed words.
        if frame_type == "session.output_audio.delta":
            return [LiveAudioDelta(data=base64.b64decode(frame.get("delta", "")))]
        if frame_type == "session.output_transcript.delta":
            return [LiveTranscriptDelta(text=str(frame.get("delta", "")))]
        if frame_type == "session.input_transcript.delta":
            return [LiveInputTranscriptDelta(text=str(frame.get("delta", "")))]
        if frame_type == "session.usage.updated":
            # Cumulative billing, reported through the session rather than on
            # close; record the elapsed seconds so LiveSessionEnded can carry
            # the last value, and surface it live.
            usage_raw = _as_dict(frame.get("usage"))
            self._record_duration(usage_raw)
            return [LiveUsageUpdated(billed_seconds=self.billed_seconds)]
        if frame_type == "response.event":
            # An envelope carrying one event of the delegated Responses
            # stream under .event, seen on the text-input path (probe round
            # 2). The audio path uses the session.* frames above instead.
            return self._responses_stream_events(_as_dict(frame.get("event")))
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
        if frame_type in ("session.done", "session.ended"):
            # If a close/ended frame ever arrives, capture any final billed
            # duration; the ended event itself is emitted on socket close.
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
