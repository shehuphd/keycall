"""ElevenLabs adapter: speech generation, voices, realtime transcription.

Every operation speaks ElevenLabs's own wire, all live-verified
2026-08-31. `GET /v1/models` is authenticated and lists TTS and
voice-conversion models with capability booleans, so classification
comes from provider metadata; STT models are absent from it and ride
the catalog instead. Speech routes by voice id in the URL path —
`POST /v1/text-to-speech/{voice_id}` — so a voice is required and the
client refuses a voiceless request pre-flight, naming the account's
voices. Realtime transcription sends audio as JSON messages (base64
PCM inside `input_audio_chunk`), not the binary frames AssemblyAI and
Deepgram take, and the server never closes after a commit: the
translator flags the session over once the post-finish final arrives,
since no close or billed-duration frame will ever come.

Errors need two extra readings the shared translator doesn't cover:
the body's `detail` is usually an object (whose `status` distinguishes
`invalid_api_key` from `missing_permissions` — a valid key without a
required read scope) but arrives as a pydantic-style list on
validation failures.
"""

from __future__ import annotations

import base64
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
    Usage,
    Voice,
)
from ._base import ProviderAdapter

# Sample rates the realtime endpoint's pcm_{rate} audio formats accept
# (verified against the session config 2026-08-31).
_REALTIME_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})


class ElevenLabsTranslator:
    """Realtime STT frames → normalized events.

    Interims arrive as partial_transcript, finals as committed_transcript
    (text only) followed by committed_transcript_with_timestamps carrying
    per-word seconds-based timings, logprob confidence, and speaker ids —
    only the timestamped final is surfaced, so one commit yields one
    FinalTranscript. Word entries typed "spacing" are separators, not
    words, and are dropped. A bad key opens the socket and then sends an
    auth_error event before the server closes; that surfaces as a typed
    error rather than a silent session end. No frame reports billed
    duration, and the server holds the socket open after a commit, so
    the translator marks the session over once the final that answers
    finish() arrives."""

    def __init__(self, sample_rate: int) -> None:
        self.audio_duration_seconds: float | None = None
        self.session_over = False
        self._sample_rate = sample_rate
        self._finish_requested = False

    def encode_audio(self, pcm: bytes) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode("ascii"),
                "sample_rate": self._sample_rate,
                "commit": False,
            }
        )

    def finish_messages(self) -> tuple[str, ...]:
        self._finish_requested = True
        return (
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "sample_rate": self._sample_rate,
                    "commit": True,
                }
            ),
        )

    def events_for_frame(self, payload: str | bytes) -> list[TranscriptionEvent]:
        if isinstance(payload, bytes):
            return [UnknownTranscriptionEvent(provider_kind="binary")]
        try:
            frame = json.loads(payload)
        except ValueError:
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        if not isinstance(frame, dict):
            return [UnknownTranscriptionEvent(provider_kind="malformed")]
        kind = str(frame.get("message_type", ""))
        if kind == "session_started":
            session_id = frame.get("session_id")
            return [
                TranscriptionSessionStarted(
                    provider_session_id=str(session_id) if session_id else None
                )
            ]
        if kind == "partial_transcript":
            text = str(frame.get("text", ""))
            return [InterimTranscript(text=text)] if text else []
        if kind == "committed_transcript":
            # The timestamped twin of this frame follows and carries
            # everything this one does plus the words; surfacing both
            # would double every final.
            return []
        if kind == "committed_transcript_with_timestamps":
            text = str(frame.get("text", ""))
            words = tuple(
                TranscriptWord(
                    text=str(word.get("text", "")),
                    start_ms=float(word.get("start", 0)) * 1000.0,
                    end_ms=float(word.get("end", 0)) * 1000.0,
                    confidence=word.get("logprob"),
                )
                for word in frame.get("words", [])
                if isinstance(word, dict) and word.get("type") == "word"
            )
            if self._finish_requested:
                self.session_over = True
            if not text:
                return []
            return [FinalTranscript(text=text, words=words, utterance_end=True)]
        if kind == "auth_error":
            raise KeyCallError(
                str(frame.get("error") or "authentication failed"),
                code=ErrorCode.INVALID_API_KEY,
                provider="elevenlabs",
            )
        if kind in ("quota_exceeded", "rate_limited"):
            raise KeyCallError(
                str(frame.get("error") or kind),
                code=ErrorCode.RATE_LIMITED,
                retryable=True,
                provider="elevenlabs",
            )
        if kind == "input_error":
            return [UnknownTranscriptionEvent(provider_kind="input_error")]
        return [UnknownTranscriptionEvent(provider_kind=kind or "?")]


class ElevenLabsAdapter(ProviderAdapter):
    requires_voice = True

    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"])

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        models: list[Model] = []
        entries = payload if isinstance(payload, list) else []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("model_id"):
                continue
            categories: set[ModelCategory] = set()
            warnings: tuple[str, ...] = ()
            if entry.get("can_do_text_to_speech"):
                categories.add(ModelCategory.SPEECH_GENERATION)
            elif entry.get("can_do_voice_conversion"):
                categories = {ModelCategory.UNKNOWN}
                warnings = ("voice-conversion model; no KeyCall operation drives it",)
            else:
                categories = {ModelCategory.UNKNOWN}
            models.append(
                Model(
                    id=str(entry["model_id"]),
                    provider=self.resolved.provider,
                    categories=frozenset(categories),
                    classification_source="provider_metadata",
                    warnings=warnings,
                )
            )
        # STT models don't appear in /v1/models; the streaming set is
        # maintained catalog data, the same as AssemblyAI's and Deepgram's.
        for entry in self.resolved.catalog_models:
            models.append(
                Model(
                    id=str(entry["id"]),
                    provider=self.resolved.provider,
                    categories=frozenset(
                        ModelCategory(category) for category in entry.get("categories", [])
                    ),
                    classification_source="keycall_catalog",
                    warnings=(
                        (
                            "streaming transcription models are not API-discoverable; "
                            "list is maintained by KeyCall"
                        ),
                    ),
                )
            )
        return models, None

    # --- speech generation ---

    def build_speech_spec(self, request: Any) -> RequestSpec:
        if not request.voice:
            # The client's pre-flight check answers this with the actual
            # voice list; reaching here without one is its bug, not the
            # caller's, so the message stays plain.
            raise KeyCallError(
                "elevenlabs routes speech by voice; pass voice=<voice_id>",
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=Operation.SPEECH_GENERATION.value,
            )
        op = self.resolved.operations["speech_generation"]
        path = op["path"].replace("{voice_id}", quote(str(request.voice), safe=""))
        return RequestSpec(
            method=op["method"],
            path=path,
            json_body={"text": request.text, "model_id": request.model},
        )

    def parse_speech_response(
        self,
        payload: Any,
        *,
        headers: Any,
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        # The endpoint answers with the audio bytes themselves, not a JSON
        # envelope (verified live 2026-08-31), the same as OpenAI's TTS route.
        if not isinstance(payload, bytes) or not payload:
            raise KeyCallError(
                "provider returned no audio for a speech-generation request",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
            )
        media_type = str(headers.get("content-type", "audio/mpeg")).split(";")[0]
        return self.speech_result(
            base64_data=base64.b64encode(payload).decode("ascii"),
            media_type=media_type or "audio/mpeg",
            usage=Usage(),
            model=model,
            round_trip_duration_ms=round_trip_duration_ms,
        )

    # --- voices ---

    def build_voices_spec(self) -> RequestSpec:
        op = self.resolved.operations["list_voices"]
        return RequestSpec(method=op["method"], path=op["path"])

    def parse_voices_response(self, payload: Any) -> tuple[Voice, ...]:
        entries = payload.get("voices") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise KeyCallError(
                "provider returned no voice list",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
            )
        voices = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("voice_id"):
                continue
            category = entry.get("category")
            voices.append(
                Voice(
                    provider=self.resolved.provider,
                    id=str(entry["voice_id"]),
                    name=str(entry.get("name") or entry["voice_id"]),
                    description=str(category) if category else None,
                )
            )
        return tuple(voices)

    # --- streaming transcription ---

    def transcription_plan(self, config: TranscriptionConfig) -> tuple[str, Any]:
        if config.sample_rate not in _REALTIME_SAMPLE_RATES:
            supported = ", ".join(str(rate) for rate in sorted(_REALTIME_SAMPLE_RATES))
            raise KeyCallError(
                f"elevenlabs streaming transcription takes sample rates "
                f"{supported}; got {config.sample_rate}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.STREAMING_TRANSCRIPTION.value,
            )
        path = self.resolved.operations["streaming_transcription"]["path"]
        path += (
            f"?audio_format=pcm_{config.sample_rate}"
            "&commit_strategy=vad&include_timestamps=true"
        )
        if config.model is not None:
            path += f"&model_id={quote(config.model, safe='')}"
        return path, ElevenLabsTranslator(config.sample_rate)

    # --- refusals for LLM operations ---

    def _refuse_llm_operation(self) -> KeyCallError:
        return KeyCallError(
            f"provider {self.resolved.provider!r} is a speech service "
            "with no text-generation API; use generate_speech or "
            "transcribe_stream",
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

    # --- error translation ---

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            message = str(detail.get("message", ""))
            status = str(detail.get("status", ""))
            if status == "missing_permissions":
                # The key is valid but missing a read scope — a different
                # fix from a wrong key, so say so instead of "invalid".
                return (
                    ErrorCode.PERMISSION_DENIED,
                    False,
                    (message or "the key is missing a permission")
                    + " (edit the key's permissions at "
                    "https://elevenlabs.io/app/settings/api-keys)",
                )
            if status:
                message = f"{message} ({status})" if message else status
            if message:
                if status_code == 401:
                    return ErrorCode.INVALID_API_KEY, False, message
                if status_code == 400:
                    return ErrorCode.MODEL_NOT_SUITABLE, False, message
        elif isinstance(detail, list):
            # Validation failures arrive pydantic-style: a list of
            # {msg, loc} entries rather than the object above.
            parts = [
                f"{'.'.join(str(loc) for loc in entry.get('loc', []))}: {entry.get('msg', '')}"
                for entry in detail
                if isinstance(entry, dict)
            ]
            if parts:
                return ErrorCode.MODEL_NOT_SUITABLE, False, "; ".join(parts)
        return super().translate_error(status_code, payload)
