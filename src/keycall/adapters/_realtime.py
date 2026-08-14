"""Realtime protocol translators.

Two wire dialects cover the three providers that offer a realtime
surface (all captured live 2026-08-14):

- OpenAI and xAI speak the Realtime API over
  ``wss://<host>/v1/realtime?model=...``: JSON event frames named
  ``session.created``, ``response.output_audio.delta``,
  ``response.output_audio_transcript.delta``, ``response.done``, and so
  on. xAI's dialect predates OpenAI's GA session shape — its session
  object keys ``modalities`` where GA keys ``output_modalities`` — and
  Grok Voice is voice-only: a text-modality update is echoed back as
  accepted, but the answer still arrives as audio plus transcript.
- Gemini speaks ``BidiGenerateContent``: a ``setup`` frame opens the
  session (the API key rides the ``x-goog-api-key`` header, never the
  URL), ``clientContent`` and ``realtimeInput`` frames go up, and
  ``serverContent`` frames come down — some as binary WebSocket frames
  carrying JSON, which the session layer decodes before translation.
  Current bidi models are audio-only (a TEXT response modality is
  refused), so the words come from ``outputTranscription``.

Translators turn provider frames into the normalized RealtimeEvent
taxonomy and caller actions into provider messages. They never see the
credential; connection and auth live in the transport.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .._errors import ErrorCode, KeyCallError
from .._types import (
    RealtimeAudioDelta,
    RealtimeConfig,
    RealtimeEvent,
    RealtimeInterrupted,
    RealtimeSessionStarted,
    RealtimeTranscriptDelta,
    RealtimeTurnComplete,
    UnknownRealtimeEvent,
    Usage,
)

# Frame types that are session plumbing, not caller-visible events.
_OPENAI_PLUMBING = frozenset(
    {
        "session.updated",
        "conversation.created",
        "conversation.item.created",
        "conversation.item.added",
        "conversation.item.done",
        "ping",
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "input_audio_buffer.committed",
        "input_audio_buffer.cleared",
        "input_audio_buffer.speech_stopped",
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
            "realtime frame was not valid JSON",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=provider,
            operation="realtime",
        ) from None
    if not isinstance(frame, dict):
        raise KeyCallError(
            "realtime frame was not a JSON object",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=provider,
            operation="realtime",
        )
    return frame


class OpenAIRealtimeTranslator:
    """The Realtime API dialect, shared by OpenAI (GA session shape) and
    xAI (pre-GA session shape, voice-only)."""

    def __init__(self, config: RealtimeConfig, *, provider: str, ga_session: bool) -> None:
        self._config = config
        self._provider = provider
        self._ga_session = ga_session

    def setup_messages(self) -> tuple[str, ...]:
        session: dict[str, Any] = {}
        if self._ga_session:
            session["type"] = "realtime"
            if self._config.voice is not None:
                session["audio"] = {"output": {"voice": self._config.voice}}
        elif self._config.voice is not None:
            session["voice"] = self._config.voice
        if self._config.instructions is not None:
            session["instructions"] = self._config.instructions
        if self._config.provider_config is not None:
            session.update(self._config.provider_config)
        if not session:
            return ()
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
        return (
            json.dumps({"type": "input_audio_buffer.commit"}),
            json.dumps({"type": "response.create"}),
        )

    def events_for_frame(self, payload: str | bytes) -> list[RealtimeEvent]:
        frame = _decode_frame(payload, provider=self._provider)
        frame_type = str(frame.get("type", ""))
        if frame_type == "session.created":
            session = frame.get("session")
            session_id = session.get("id") if isinstance(session, dict) else None
            return [
                RealtimeSessionStarted(
                    provider_session_id=str(session_id) if session_id else None
                )
            ]
        if frame_type == "response.output_audio.delta":
            return [RealtimeAudioDelta(data=base64.b64decode(frame.get("delta", "")))]
        if frame_type in (
            "response.output_text.delta",
            "response.output_audio_transcript.delta",
        ):
            return [RealtimeTranscriptDelta(text=str(frame.get("delta", "")))]
        if frame_type == "input_audio_buffer.speech_started":
            # The provider's voice-activity detection heard the caller
            # start talking; any answer in flight is being cut off.
            return [RealtimeInterrupted()]
        if frame_type == "response.done":
            response = _as_dict(frame.get("response"))
            if str(response.get("status", "")) == "cancelled":
                return [RealtimeInterrupted()]
            usage_raw = _as_dict(response.get("usage"))
            return [
                RealtimeTurnComplete(
                    usage=Usage(
                        input_tokens=usage_raw.get("input_tokens"),
                        output_tokens=usage_raw.get("output_tokens"),
                        total_tokens=usage_raw.get("total_tokens"),
                    )
                )
            ]
        if frame_type == "error":
            error = _as_dict(frame.get("error"))
            raise KeyCallError(
                f"provider reported a realtime error: {str(error.get('message', ''))[:300]}",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._provider,
                operation="realtime",
            )
        if frame_type in _OPENAI_PLUMBING:
            return []
        return [UnknownRealtimeEvent(provider_kind=frame_type[:100] or "unnamed")]


class GeminiRealtimeTranslator:
    """The BidiGenerateContent dialect."""

    # Gemini Live takes caller audio as 16 kHz 16-bit PCM and answers at
    # 24 kHz (both documented and observed 2026-08-14).
    _INPUT_MIME = "audio/pcm;rate=16000"

    def __init__(self, config: RealtimeConfig, *, provider: str) -> None:
        self._config = config
        self._provider = provider

    def setup_messages(self) -> tuple[str, ...]:
        model = self._config.model
        if not model.startswith("models/"):
            model = f"models/{model}"
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if self._config.voice is not None:
            generation_config["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._config.voice}}
            }
        setup: dict[str, Any] = {
            "model": model,
            "generationConfig": generation_config,
            # Without this the words exist only as audio; with it the
            # provider transcribes its own speech.
            "outputAudioTranscription": {},
        }
        if self._config.instructions is not None:
            setup["systemInstruction"] = {"parts": [{"text": self._config.instructions}]}
        if self._config.provider_config is not None:
            setup.update(self._config.provider_config)
        return (json.dumps({"setup": setup}),)

    def user_text_messages(self, text: str) -> tuple[str, ...]:
        return (
            json.dumps(
                {
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": text}]}],
                        "turnComplete": True,
                    }
                }
            ),
        )

    def audio_chunk_messages(self, pcm: bytes) -> tuple[str, ...]:
        encoded = base64.b64encode(pcm).decode()
        return (
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {"data": encoded, "mimeType": self._INPUT_MIME}
                    }
                }
            ),
        )

    def end_audio_messages(self) -> tuple[str, ...]:
        # Gemini's server-side voice-activity detection decides when the
        # caller's turn ended; this only tells it the stream ran dry.
        return (json.dumps({"realtimeInput": {"audioStreamEnd": True}}),)

    def events_for_frame(self, payload: str | bytes) -> list[RealtimeEvent]:
        frame = _decode_frame(payload, provider=self._provider)
        if "setupComplete" in frame:
            return [RealtimeSessionStarted()]
        content = frame.get("serverContent")
        if not isinstance(content, dict):
            names = ", ".join(sorted(frame.keys()))
            return [UnknownRealtimeEvent(provider_kind=names[:100] or "unnamed")]
        events: list[RealtimeEvent] = []
        if content.get("interrupted"):
            events.append(RealtimeInterrupted())
        model_turn = content.get("modelTurn")
        if isinstance(model_turn, dict):
            for part in model_turn.get("parts", []):
                if not isinstance(part, dict):
                    continue
                blob = part.get("inlineData")
                if isinstance(blob, dict) and blob.get("data"):
                    events.append(
                        RealtimeAudioDelta(data=base64.b64decode(blob["data"]))
                    )
                if part.get("text") and not part.get("thought"):
                    # Thought parts are the model's reasoning, flagged
                    # `thought: true` (observed live 2026-08-14); they are
                    # not words being spoken and stay out of the transcript.
                    events.append(RealtimeTranscriptDelta(text=str(part["text"])))
        transcription = content.get("outputTranscription")
        if isinstance(transcription, dict) and transcription.get("text"):
            events.append(RealtimeTranscriptDelta(text=str(transcription["text"])))
        if content.get("turnComplete"):
            usage_raw = _as_dict(frame.get("usageMetadata"))
            prompt = usage_raw.get("promptTokenCount")
            answer = usage_raw.get("responseTokenCount")
            events.append(
                RealtimeTurnComplete(
                    usage=Usage(
                        input_tokens=prompt,
                        output_tokens=answer,
                        total_tokens=usage_raw.get("totalTokenCount"),
                        reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                    )
                )
            )
        if not events:
            # generationComplete and friends: plumbing between turns.
            return []
        return events
