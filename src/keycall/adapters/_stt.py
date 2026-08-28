"""Speech-to-text adapters: AssemblyAI and Deepgram streaming transcription.

Both providers speak a WebSocket dialect: the caller pushes raw 16-bit
mono PCM as binary frames and reads JSON transcript frames back. The
dialects differ in every particular — AssemblyAI finalizes whole turns
(a growing transcript on one turn_order, closed by end_of_turn, timings
in integer ms), Deepgram finalizes stretches (is_final, with the
utterance boundary a separate speech_final flag, timings in float
seconds) — so each has its own translator, and the normalized event set
is the contract. All frame behavior live-verified 2026-08-28.

Neither provider is an LLM vendor: text generation and every other LLM
operation refuse with a typed error, and the model list is maintained
catalog data behind a credential-validating GET, the same pattern
Perplexity's non-discoverable model list uses.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import (
    FinalTranscript,
    InterimTranscript,
    InvocationResult,
    Model,
    TextGenerationRequest,
    TranscriptionConfig,
    TranscriptionEvent,
    TranscriptionSessionStarted,
    TranscriptWord,
    UnknownTranscriptionEvent,
)
from ._base import ProviderAdapter


class _SttAdapter(ProviderAdapter):
    """Shared base for STT-only providers: catalog model listing behind a
    credential check, and clean refusals for every LLM operation."""

    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"])

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        # The response itself is discarded: reaching a 2xx proves the
        # credential works, which is all this call can honestly establish.
        models = [
            Model(
                id=str(entry["id"]),
                provider=self.resolved.provider,
                categories=frozenset(
                    ModelCategory(category) for category in entry.get("categories", [])
                ),
                classification_source="keycall_catalog",
                warnings=(
                    (
                        f"{self.resolved.provider} models are not API-discoverable; "
                        "list is maintained by KeyCall"
                    ),
                ),
            )
            for entry in self.resolved.catalog_models
        ]
        return models, None

    def _refuse_llm_operation(self) -> KeyCallError:
        return KeyCallError(
            f"provider {self.resolved.provider!r} is a speech-to-text service "
            "with no text-generation API; use transcribe_stream",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.TEXT_GENERATION.value,
        )

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        raise self._refuse_llm_operation()

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Any,
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        raise self._refuse_llm_operation()


class AssemblyAITranslator:
    """AssemblyAI Universal Streaming v3 frames → normalized events.

    Interims arrive as Turn frames with end_of_turn false, the transcript
    growing in place; the final Turn for the same turn_order carries
    end_of_turn true with per-word timings (integer ms) and per-word
    confidence — no overall confidence exists, so FinalTranscript's stays
    None. SpeechStarted frames are voice-activity notices with no
    portable meaning and are skipped. Termination carries
    audio_duration_seconds, held for the session's ending event."""

    def __init__(self) -> None:
        self.audio_duration_seconds: float | None = None

    def finish_messages(self) -> tuple[str, ...]:
        return (json.dumps({"type": "Terminate"}),)

    def events_for_frame(self, payload: str | bytes) -> list[TranscriptionEvent]:
        if isinstance(payload, bytes):
            return [UnknownTranscriptionEvent(provider_kind="binary")]
        try:
            frame = json.loads(payload)
        except ValueError:
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        if not isinstance(frame, dict):
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        kind = str(frame.get("type", ""))
        if kind == "Begin":
            return [
                TranscriptionSessionStarted(
                    provider_session_id=str(frame["id"]) if frame.get("id") else None
                )
            ]
        if kind == "Turn":
            text = str(frame.get("transcript", ""))
            if not frame.get("end_of_turn"):
                return [InterimTranscript(text=text)] if text else []
            words = tuple(
                TranscriptWord(
                    text=str(word.get("text", "")),
                    start_ms=float(word.get("start", 0)),
                    end_ms=float(word.get("end", 0)),
                    confidence=word.get("confidence"),
                )
                for word in frame.get("words", [])
                if isinstance(word, dict)
            )
            return [FinalTranscript(text=text, words=words, utterance_end=True)]
        if kind == "Termination":
            duration = frame.get("audio_duration_seconds")
            if duration is not None:
                self.audio_duration_seconds = float(duration)
            return []
        if kind == "SpeechStarted":
            return []  # voice-activity notice, no portable meaning
        return [UnknownTranscriptionEvent(provider_kind=kind or "?")]


class DeepgramTranslator:
    """Deepgram /v1/listen frames → normalized events.

    Results frames carry is_final (this text will not change) separately
    from speech_final (the speaker also finished the utterance) — both
    surface, as FinalTranscript.utterance_end. Word timings are float
    seconds, normalized to ms; punctuated_word is preferred over the raw
    word when present, matching the punctuate=true the session requests.
    The terminal Metadata frame carries duration, held for the session's
    ending event. No session-begin frame exists."""

    def __init__(self) -> None:
        self.audio_duration_seconds: float | None = None

    def finish_messages(self) -> tuple[str, ...]:
        return (json.dumps({"type": "CloseStream"}),)

    def events_for_frame(self, payload: str | bytes) -> list[TranscriptionEvent]:
        if isinstance(payload, bytes):
            return [UnknownTranscriptionEvent(provider_kind="binary")]
        try:
            frame = json.loads(payload)
        except ValueError:
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        if not isinstance(frame, dict):
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        kind = str(frame.get("type", ""))
        if kind == "Results":
            channel_index = frame.get("channel_index")
            channel = (
                int(channel_index[0])
                if isinstance(channel_index, list) and channel_index
                else None
            )
            alternatives = (frame.get("channel") or {}).get("alternatives") or [{}]
            alt = alternatives[0] if isinstance(alternatives[0], dict) else {}
            text = str(alt.get("transcript", ""))
            if not frame.get("is_final"):
                return [InterimTranscript(text=text, channel=channel)] if text else []
            if not text:
                return []  # silence finalized: nothing was said
            words = tuple(
                TranscriptWord(
                    text=str(word.get("punctuated_word") or word.get("word", "")),
                    start_ms=float(word.get("start", 0)) * 1000.0,
                    end_ms=float(word.get("end", 0)) * 1000.0,
                    confidence=word.get("confidence"),
                )
                for word in alt.get("words", [])
                if isinstance(word, dict)
            )
            return [
                FinalTranscript(
                    text=text,
                    words=words,
                    utterance_end=bool(frame.get("speech_final")),
                    confidence=alt.get("confidence"),
                    channel=channel,
                )
            ]
        if kind == "Metadata":
            duration = frame.get("duration")
            if duration is not None:
                self.audio_duration_seconds = float(duration)
            return []
        if kind in ("SpeechStarted", "UtteranceEnd"):
            # Voice-activity notices; the utterance boundary itself is
            # normalized as FinalTranscript.utterance_end (speech_final).
            return []
        return [UnknownTranscriptionEvent(provider_kind=kind or "?")]


class AssemblyAIAdapter(_SttAdapter):
    def transcription_plan(self, config: TranscriptionConfig) -> tuple[str, Any]:
        path = self.resolved.operations["streaming_transcription"]["path"]
        path += f"?sample_rate={config.sample_rate}"
        if config.model is not None:
            path += f"&speech_model={quote(config.model, safe='')}"
        return path, AssemblyAITranslator()


class DeepgramAdapter(_SttAdapter):
    def transcription_plan(self, config: TranscriptionConfig) -> tuple[str, Any]:
        path = self.resolved.operations["streaming_transcription"]["path"]
        path += (
            f"?encoding=linear16&sample_rate={config.sample_rate}&channels=1"
            "&interim_results=true&endpointing=300&utterance_end_ms=1000&punctuate=true"
        )
        if config.model is not None:
            path += f"&model={quote(config.model, safe='')}"
        return path, DeepgramTranslator()
