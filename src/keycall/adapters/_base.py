"""Adapter interface: pure request-building and response-parsing.

Adapters never perform I/O and never see the credential. The client drives
the page loop and hands specs to the transport layer, which is the single
place credentials are revealed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._registry import ResolvedProvider, providers_with
from .._sanitize import safe_request_id
from .._transport import DownloadPlan, RequestSpec
from .._types import (
    AudioOutput,
    Citation,
    CodeExecutionOutput,
    EmbeddingOutput,
    ImageOutput,
    InvocationResult,
    Model,
    OutputPart,
    StreamEvent,
    TextGenerationRequest,
    TextOutput,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallComplete,
    ToolCallStarted,
    Usage,
    VideoJob,
    VideoOutput,
    Voice,
)


def parse_tool_arguments(raw: Any, *, provider: str) -> Mapping[str, Any]:
    """Providers that send arguments as a JSON string (OpenAI, the compat
    family) get parsed here; malformed argument JSON from a provider is a
    typed error, never a silently dropped call."""
    if isinstance(raw, Mapping):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        parsed = None
    if not isinstance(parsed, dict):
        raise KeyCallError(
            "provider sent malformed tool-call arguments",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=provider,
            operation=Operation.TEXT_GENERATION.value,
        )
    return parsed


# Magic bytes for the formats every image-capable provider in the v1 set
# accepts. Sniffing beats trusting a caller-supplied media_type: Anthropic
# and Gemini both require an accurate one and reject a mismatch, and a
# caller passing bytes usually has no reason to know the format.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


_AUDIO_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
)

_FILE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
)


def media_type_for(part: Any, *, kind: str, provider: str) -> str:
    """The media type for a part sent as bytes, read from the content
    rather than from a caller's label: providers reject a mismatch, and the
    bytes are the better evidence. A declared media_type covers formats
    KeyCall doesn't recognize; an unidentifiable one raises rather than
    being sent with a guess."""
    data = part.data or b""
    table = {
        "image": _IMAGE_SIGNATURES,
        "audio": _AUDIO_SIGNATURES,
        "file": _FILE_SIGNATURES,
    }[kind]
    for signature, media_type in table:
        if data.startswith(signature):
            return media_type
    if kind == "image" and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if kind == "audio" and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if part.media_type:
        return str(part.media_type)
    raise KeyCallError(
        f"could not identify the {kind} format from its content; pass "
        "media_type on the part",
        code=ErrorCode.UNSUPPORTED_OPERATION,
        provider=provider,
        operation=Operation.TEXT_GENERATION.value,
    )


def image_media_type(part: Any, *, provider: str) -> str:
    """The media type for an image sent as bytes: sniffed from the content,
    falling back to a declared media_type only for formats we don't
    recognize. An unidentifiable image is a typed error, not a guess that
    the provider will reject with something less clear."""
    return media_type_for(part, kind="image", provider=provider)


def released_at(entry: Mapping[str, Any]) -> datetime | None:
    """When a list entry says the model appeared, from whichever field the
    provider uses: a unix `created` (OpenAI, Moonshot) or an ISO
    `created_at` (Anthropic). Returns None when the provider reports
    nothing or reports something unparseable, because a bad timestamp
    would reorder the walk on invented evidence, and no timestamp at all
    is handled by the caller.

    Always tz-aware and in UTC, so values from the two formats sort against
    each other rather than raising on a naive/aware comparison."""
    raw = entry.get("created")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    iso = entry.get("created_at")
    if isinstance(iso, str) and iso:
        try:
            parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


# What each provider calls the input-token ceiling in a model-list entry.
# Three of the six report one and they agree on the meaning while disagreeing
# on the name, which is the ordinary case for a normalizing layer: read every
# spelling, expose one field, leave it None where a provider says nothing.
_CONTEXT_LIMIT_FIELDS = (
    "inputTokenLimit",  # Gemini
    "max_input_tokens",  # Anthropic
    "context_length",  # Moonshot
)


def context_limit(entry: Mapping[str, Any]) -> int | None:
    """The largest input the model accepts, in tokens, as the provider
    reports it. None where the provider reports nothing (OpenAI, DeepSeek)
    or reports something unusable.

    Deliberately never inferred from a bundled table or a sibling model:
    a caller budgets against this number, and an invented one is worse than
    an absent one it can see and handle."""
    for field in _CONTEXT_LIMIT_FIELDS:
        raw = entry.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        # A zero or negative ceiling is a provider bug, not a usable limit.
        if value > 0:
            return value
    return None


def dedupe_citations(citations: Sequence[Citation]) -> tuple[Citation, ...]:
    """Drop citations that repeat one already present in full, keeping the
    first occurrence and the provider's order.

    Deliberately not "one entry per URL". Providers differ in what a
    citation means: Anthropic attaches distinct ``cited_text`` per claim, so
    the same source legitimately appears several times and collapsing by URL
    would destroy the attribution. OpenAI sends url and title with no
    excerpt, so a source cited for three claims arrives as three identical
    records carrying no information the first doesn't. Only the second kind
    is removed. Before this rule KeyCall collapsed by URL on Perplexity and
    on streamed Gemini but not on non-streamed Gemini, so the same request
    returned different citations depending on the path.
    """
    seen: set[tuple[str, str | None, str | None]] = set()
    unique: list[Citation] = []
    for citation in citations:
        identity = (citation.url, citation.title, citation.cited_text)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(citation)
    return tuple(unique)


def _schema_has_key(schema: Any, key: str) -> bool:
    """Whether `key` appears anywhere in a JSON Schema tree, at any nesting
    depth: a schema generator puts a keyword like `additionalProperties` on
    every object level it emits, and `$defs`/`definitions` can nest one
    again inside a referenced sub-schema."""
    if isinstance(schema, dict):
        if key in schema:
            return True
        return any(_schema_has_key(value, key) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_key(item, key) for item in schema)
    return False


class InbandStreamError(Exception):
    """A provider error event received mid-stream. Carries the raw provider
    message; the client scrubs it before it can reach a KeyCallError."""

    def __init__(self, code: ErrorCode, retryable: bool, raw_message: str) -> None:
        super().__init__(raw_message)
        self.code = code
        self.retryable = retryable
        self.raw_message = raw_message


class StreamAssembler(ABC):
    """Per-call accumulator: translates raw SSE pairs into typed events and
    builds the final InvocationResult. Fed by the client's TextStream; the
    response headers are set on it once the stream opens."""

    def __init__(self, resolved: ResolvedProvider, request: TextGenerationRequest) -> None:
        self.resolved = resolved
        self.request = request
        self.response_headers: Mapping[str, str] = {}
        self.saw_terminal = False
        self.model: str = request.model
        self.finish_reason: str | None = None
        self.usage = Usage()
        self.usage_reported = False
        self.provider_request_id: str | None = None
        self.citations: list[Citation] = []
        self.warnings: list[str] = []
        self.tool_calls: list[ToolCall] = []
        self.code_executions: list[CodeExecutionOutput] = []
        self._text: list[str] = []
        # Provider key (block index, item id, choice index) -> in-flight call.
        self._pending_calls: dict[Any, dict[str, Any]] = {}

    @abstractmethod
    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        """Translate one SSE pair into zero or more typed events. Raises
        InbandStreamError for provider error events, KeyCallError for
        malformed stream data."""

    def on_close(self) -> list[StreamEvent]:
        """Called when the server closes the stream. Adapters whose
        protocol has no terminal event (Gemini) decide here whether the
        close was a completion; the default treats close as no signal."""
        return []

    def _parse_data(self, data: str) -> Any:
        try:
            return json.loads(data)
        except ValueError:
            raise KeyCallError(
                "provider sent a non-JSON stream event",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            ) from None

    def append_text(self, text: str) -> None:
        self._text.append(text)

    @property
    def text(self) -> str:
        return "".join(self._text)

    # --- streamed tool calls ---
    #
    # Three of the four protocols announce a call, stream its arguments as
    # JSON fragments, then close it; only the keys differ (Anthropic block
    # index, OpenAI item id, compat tool-call index). These helpers hold
    # that shared shape so each adapter only maps its own event names.

    def begin_tool_call(
        self, key: Any, *, call_id: str, name: str, opaque: str | None = None
    ) -> ToolCallStarted:
        self._pending_calls[key] = {
            "id": call_id,
            "name": name,
            "opaque": opaque,
            "fragments": [],
        }
        return ToolCallStarted(id=call_id, name=name)

    def append_tool_arguments(self, key: Any, fragment: str) -> list[StreamEvent]:
        pending = self._pending_calls.get(key)
        if pending is None or not fragment:
            return []
        pending["fragments"].append(fragment)
        return [ToolCallArgumentsDelta(id=pending["id"], fragment=fragment)]

    def complete_tool_call(self, key: Any, *, arguments: Any = None) -> list[StreamEvent]:
        """Close an in-flight call and record it. ``arguments`` overrides the
        accumulated fragments for providers that also send the finished
        argument string (OpenAI)."""
        pending = self._pending_calls.pop(key, None)
        if pending is None:
            return []
        raw = arguments if arguments is not None else "".join(pending["fragments"])
        call = ToolCall(
            id=pending["id"],
            name=pending["name"],
            arguments=parse_tool_arguments(raw, provider=self.resolved.provider),
            opaque=pending["opaque"],
        )
        self.tool_calls.append(call)
        return [ToolCallComplete(tool_call=call)]

    def flush_tool_calls(self) -> list[StreamEvent]:
        """Close every still-open call, in the order they were announced.
        Providers that never mark a call finished (the compat family closes
        the whole message instead) land here."""
        events: list[StreamEvent] = []
        for key in list(self._pending_calls):
            events.extend(self.complete_tool_call(key))
        return events

    def record_tool_call(self, call: ToolCall) -> list[StreamEvent]:
        """Record a call that arrived whole (Gemini), with no fragments to
        accumulate. Still reported as started-then-complete so callers can
        treat every provider the same way."""
        self.tool_calls.append(call)
        return [
            ToolCallStarted(id=call.id, name=call.name),
            ToolCallComplete(tool_call=call),
        ]

    def record_code_execution(self, output: CodeExecutionOutput) -> None:
        """Record a code_interpreter/code_execution run that arrived whole.
        No StreamEvent of its own — unlike a tool call, there is nothing
        for a caller to act on mid-stream; it only needs to appear once,
        correctly, in the final result's parts."""
        self.code_executions.append(output)

    def finalize(self, *, round_trip_duration_ms: float) -> InvocationResult:
        if self.provider_request_id is None and self.resolved.provider_request_id_header:
            self.provider_request_id = safe_request_id(
                self.response_headers.get(self.resolved.provider_request_id_header)
            )
        if not self.usage_reported:
            self.warnings.append("provider reported no usage information")
        parts: list[OutputPart] = []
        if self._text:
            parts.append(TextOutput(text=self.text))
        parts.extend(self.tool_calls)
        parts.extend(self.code_executions)
        return InvocationResult(
            provider=self.resolved.provider,
            model=self.model,
            operation=Operation.TEXT_GENERATION,
            parts=tuple(parts),
            usage=self.usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=self.provider_request_id,
            finish_reason=self.finish_reason,
            citations=dedupe_citations(self.citations),
            warnings=tuple(self.warnings),
        )


class ProviderAdapter(ABC):
    """One instance per resolved provider profile. Stateless and pure."""

    def __init__(self, resolved: ResolvedProvider) -> None:
        self.resolved = resolved

    # --- model listing (client drives the page loop) ---

    @abstractmethod
    def initial_list_request(self) -> RequestSpec: ...

    @abstractmethod
    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        """Return this page's normalized models and the next page's spec,
        or None when there are no more pages. Must tolerate unknown fields
        and classify conservatively."""

    # --- text generation ---

    @abstractmethod
    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec: ...

    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        """Streaming variant of build_generation_spec. Raises for adapters
        that haven't implemented streaming rather than guessing a flag."""
        raise KeyCallError(
            f"streaming is not implemented for provider {self.resolved.provider!r}",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.TEXT_GENERATION.value,
        )

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        raise KeyCallError(
            f"streaming is not implemented for provider {self.resolved.provider!r}",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.TEXT_GENERATION.value,
        )

    @abstractmethod
    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        """Decode into the common envelope. Raw provider objects must not
        leak through the public API."""

    # --- embeddings ---

    def build_embedding_spec(self, request: Any) -> RequestSpec:
        """Adapters without a verified embeddings endpoint refuse here
        rather than guessing a path."""
        raise KeyCallError(
            f"provider {self.resolved.provider!r} has no embeddings API; "
            "embeddings are supported on: "
            + ", ".join(sorted(providers_with("embeddings"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.EMBEDDING.value,
        )

    def parse_embedding_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
        expected: int,
    ) -> InvocationResult:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} has no embeddings API",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.EMBEDDING.value,
        )

    def embedding_result(
        self,
        vectors: list[tuple[float, ...]],
        *,
        usage: Usage,
        model: str,
        round_trip_duration_ms: float,
        provider_request_id: str | None = None,
        expected: int,
    ) -> InvocationResult:
        """One EmbeddingOutput per input, in the order they were sent. A
        provider returning a different count is a typed error: silently
        misaligning vectors with inputs would corrupt a caller's index."""
        if len(vectors) != expected:
            raise KeyCallError(
                f"provider returned {len(vectors)} embeddings for {expected} "
                "inputs; the vectors cannot be matched to their inputs",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.EMBEDDING.value,
            )
        return InvocationResult(
            provider=self.resolved.provider,
            model=model,
            operation=Operation.EMBEDDING,
            parts=tuple(EmbeddingOutput(values=vector) for vector in vectors),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=provider_request_id,
        )

    # --- image generation ---

    def build_image_spec(self, request: Any) -> RequestSpec:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate images; "
            "image generation is supported on: "
            + ", ".join(sorted(providers_with("image_generation"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.IMAGE_GENERATION.value,
        )

    def parse_image_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate images",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.IMAGE_GENERATION.value,
        )

    def image_result(
        self,
        images: list[tuple[str, str]],
        *,
        usage: Usage,
        model: str,
        round_trip_duration_ms: float,
        provider_request_id: str | None = None,
        warnings: tuple[str, ...] = (),
    ) -> InvocationResult:
        """images is (base64_data, media_type) per picture. A response with
        no image is a typed error: a caller asking for a picture and getting
        an empty result would otherwise have to guess why."""
        if not images:
            raise KeyCallError(
                "provider returned no image for an image-generation request",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.IMAGE_GENERATION.value,
            )
        return InvocationResult(
            provider=self.resolved.provider,
            model=model,
            operation=Operation.IMAGE_GENERATION,
            parts=tuple(
                ImageOutput(base64_data=data, media_type=media_type)
                for data, media_type in images
            ),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=provider_request_id,
            warnings=warnings,
        )

    # --- speech generation ---

    def build_speech_spec(self, request: Any) -> RequestSpec:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate speech; "
            "speech generation is supported on: "
            + ", ".join(sorted(providers_with("speech_generation"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    def parse_speech_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate speech",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    def speech_result(
        self,
        *,
        base64_data: str,
        media_type: str,
        usage: Usage,
        model: str,
        round_trip_duration_ms: float,
        provider_request_id: str | None = None,
        warnings: tuple[str, ...] = (),
    ) -> InvocationResult:
        """One AudioOutput. Unlike images, no provider offers an `n` for
        this operation — a TTS call always produces one clip, so
        the parameter list takes a single clip rather than a list of
        them, matching what the operation returns."""
        return InvocationResult(
            provider=self.resolved.provider,
            model=model,
            operation=Operation.SPEECH_GENERATION,
            parts=(AudioOutput(base64_data=base64_data, media_type=media_type),),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=provider_request_id,
            warnings=warnings,
        )

    # --- voices ---
    #
    # A provider whose voice set is a fixed named list carries it in the
    # catalog (resolved.catalog_voices) and the client answers list_voices
    # without the network; a provider with a live voices endpoint
    # implements the two hooks below instead. A voice is required before
    # any speech request can be built when requires_voice is True — the
    # client enforces it pre-flight so the refusal can name the choices.

    requires_voice: bool = False

    def build_voices_spec(self) -> RequestSpec:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate speech, so "
            "it has no voices; speech generation is supported on: "
            + ", ".join(sorted(providers_with("speech_generation"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    def parse_voices_response(self, payload: Any) -> tuple[Voice, ...]:
        raise KeyCallError(
            f"provider {self.resolved.provider!r} has no live voices endpoint",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    # --- video generation ---
    #
    # Three-phase job lifecycle, unlike every synchronous operation above:
    # start a render, poll its status, download the finished file. Both
    # supporting providers (Gemini's Veo, xAI's Grok Imagine) converged on
    # this shape independently, so the adapter interface mirrors it
    # directly rather than pretending video answers in one round trip.

    def _video_gate(self) -> KeyCallError:
        return KeyCallError(
            f"provider {self.resolved.provider!r} cannot generate video; "
            "video generation is supported on: "
            + ", ".join(sorted(providers_with("video_generation"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.VIDEO_GENERATION.value,
        )

    def build_video_start_spec(self, request: Any) -> RequestSpec:
        raise self._video_gate()

    def parse_video_start(self, payload: Any, *, model: str) -> VideoJob:
        raise self._video_gate()

    def build_video_status_spec(self, job: VideoJob) -> RequestSpec:
        raise self._video_gate()

    def parse_video_status(self, payload: Any, *, job: VideoJob) -> VideoJob:
        raise self._video_gate()

    def video_download_plan(self, job: VideoJob) -> DownloadPlan:
        raise self._video_gate()

    def server_tool_continuation(
        self, request: Any, result: Any
    ) -> Any | None:
        """When a provider executes a tool server-side but still requires
        the caller to echo the call back before it will answer (Moonshot's
        $web_search), the follow-up request that performs the echo. None
        everywhere else, which is every other provider."""
        return None

    def realtime_plan(self, config: Any) -> tuple[str, Any]:
        """The WebSocket path (host-rooted, unlike request paths, which
        stack on the base URL's own prefix) and the frame translator for
        a realtime session. Providers without a realtime API refuse here,
        before any connection."""
        raise KeyCallError(
            f"provider {self.resolved.provider!r} has no realtime API; "
            "realtime is supported on: "
            + ", ".join(sorted(providers_with("realtime"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation="realtime",
        )

    def transcription_plan(self, config: Any) -> tuple[str, Any]:
        """The WebSocket path and frame translator for a streaming
        transcription session. Providers without one refuse here, before
        any connection."""
        raise KeyCallError(
            f"provider {self.resolved.provider!r} has no streaming "
            "transcription API; transcribe_stream is supported on: "
            + ", ".join(sorted(providers_with("streaming_transcription"))),
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.STREAMING_TRANSCRIPTION.value,
        )

    def video_result(
        self,
        *,
        base64_data: str,
        media_type: str,
        url: str | None,
        model: str,
        round_trip_duration_ms: float,
        usage: Usage | None = None,
        warnings: tuple[str, ...] = (),
    ) -> InvocationResult:
        """One VideoOutput per render: neither provider offers an `n` for
        video the way image generation does. ``url`` is the provider's
        own download location, carried so a caller who prefers streaming
        the file elsewhere can, for as long as the provider keeps it
        alive."""
        return InvocationResult(
            provider=self.resolved.provider,
            model=model,
            operation=Operation.VIDEO_GENERATION,
            parts=(
                VideoOutput(base64_data=base64_data, media_type=media_type, url=url),
            ),
            usage=usage or Usage(),
            round_trip_duration_ms=round_trip_duration_ms,
            warnings=warnings,
        )

    # --- error translation (transport calls this, then scrubs) ---

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        """Map a provider error response to (code, retryable, message).
        The returned message is scrubbed by the transport before use.
        Default covers the common OpenAI-shaped error body."""
        message = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", ""))
            elif isinstance(error, str):
                message = error
        if status_code == 401:
            return ErrorCode.INVALID_API_KEY, False, message or "invalid API key"
        if status_code == 403:
            return ErrorCode.PERMISSION_DENIED, False, message or "permission denied"
        if status_code == 402:
            # The credential is valid and the account is not entitled to
            # use it: unpaid balance, exhausted credit, billing hold. Not a
            # malformed response, which is where an unmapped status lands
            # and which sends a caller looking for a bug in their request.
            # The provider's own message carries the actionable detail.
            return (
                ErrorCode.PERMISSION_DENIED,
                False,
                message or "payment required: the account cannot make calls until billing is settled",
            )
        if status_code == 404:
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "not found"
        if status_code == 429:
            return ErrorCode.RATE_LIMITED, True, message or "rate limited"
        if status_code == 400 and "is not supported with" in message:
            # A capability the provider has but this model doesn't, e.g.
            # OpenAI's web_search tool on gpt-3.5-turbo. The request is
            # well formed, so calling it a malformed response points the
            # caller at their own JSON instead of at the model choice.
            guidance = (
                f"{message} Choose a model that supports it, or turn the "
                "feature off for this one."
            )
            return ErrorCode.MODEL_NOT_SUITABLE, False, guidance
        if status_code >= 500:
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider server error"
        return (
            ErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
            message or f"unexpected status {status_code}",
        )

    # --- shared helpers ---

    def validate_generation_request(self, request: TextGenerationRequest) -> None:
        """Pre-flight checks that mirror what the provider would reject:
        part types and placement, capability gates, and sampling params
        against models with maintained evidence that they reject them."""
        from .._capabilities import TOOL_CALLING_PROVIDERS, sampling_violation
        from .._types import AudioInput, FileInput, ImageInput, TextInput, ToolCall, ToolResult

        for message in request.messages:
            for part in message.content:
                if isinstance(part, TextInput):
                    continue
                if isinstance(part, ToolCall):
                    if message.role != "assistant":
                        raise KeyCallError(
                            "ToolCall parts belong in assistant messages (the model's "
                            f"turn), not role {message.role!r}",
                            code=ErrorCode.UNSUPPORTED_OPERATION,
                            operation=Operation.TEXT_GENERATION.value,
                        )
                    continue
                if isinstance(part, ToolResult):
                    if message.role != "user":
                        raise KeyCallError(
                            "ToolResult parts belong in user messages (the caller's "
                            f"turn), not role {message.role!r}",
                            code=ErrorCode.UNSUPPORTED_OPERATION,
                            operation=Operation.TEXT_GENERATION.value,
                        )
                    continue
                media_kind = {
                    ImageInput: "image",
                    AudioInput: "audio",
                    FileInput: "file",
                }.get(type(part))
                if media_kind is not None:
                    self._check_media_part(part, role=message.role, kind=media_kind)
                    continue
                raise KeyCallError(
                    f"{type(part).__name__} is not implemented for text "
                    "generation; keycall accepts TextInput, ImageInput, "
                    "AudioInput, FileInput, ToolCall, and ToolResult parts",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if request.tools:
            if not self.resolved.capabilities.tool_calling:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} does not support tool "
                    "calling; tools are supported on: "
                    + ", ".join(sorted(TOOL_CALLING_PROVIDERS)),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
            if any(tool.input_schema is None for tool in request.tools):
                from .._capabilities import CUSTOM_TOOL_PROVIDERS

                if not self.resolved.capabilities.custom_tool:
                    raise KeyCallError(
                        f"provider {self.resolved.provider!r} has no custom "
                        "(freeform) tool convention; a Tool with "
                        "input_schema=None is supported on: "
                        + ", ".join(sorted(CUSTOM_TOOL_PROVIDERS)),
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        provider=self.resolved.provider,
                        operation=Operation.TEXT_GENERATION.value,
                    )
            if any(tool.defer_loading for tool in request.tools):
                from .._capabilities import TOOL_SEARCH_PROVIDERS

                if not self.resolved.capabilities.tool_search:
                    raise KeyCallError(
                        f"provider {self.resolved.provider!r} has no tool-search "
                        "convention; defer_loading=True is supported on: "
                        + ", ".join(sorted(TOOL_SEARCH_PROVIDERS)),
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        provider=self.resolved.provider,
                        operation=Operation.TEXT_GENERATION.value,
                    )
            if request.response_schema is not None and self.resolved.provider == "anthropic":
                # Schema enforcement on Anthropic is itself a forced tool
                # call; combining it with caller tools is mechanically
                # impossible in one turn.
                raise KeyCallError(
                    "anthropic cannot combine tools with response_schema: "
                    "schema enforcement forces its own tool call, excluding "
                    "the caller's tools in the same turn",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if self.resolved.provider == "anthropic":
            for message in request.messages:
                for part in message.content:
                    if (
                        isinstance(part, TextInput)
                        and part.cacheable
                        and part.cache_ttl_seconds not in (300, 3600)
                    ):
                        raise KeyCallError(
                            "anthropic's cache_control ttl accepts only 300 "
                            "(\"5m\") or 3600 (\"1h\") seconds, not "
                            f"{part.cache_ttl_seconds}",
                            code=ErrorCode.UNSUPPORTED_OPERATION,
                            provider=self.resolved.provider,
                            operation=Operation.TEXT_GENERATION.value,
                        )
        violation = sampling_violation(
            self.resolved,
            request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if violation is not None:
            raise KeyCallError(
                violation,
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if request.web_search:
            from .._capabilities import WEB_SEARCH_PROVIDERS

            if not self.resolved.capabilities.web_search:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} has no native web search "
                    "tool; web_search is supported on: "
                    + ", ".join(sorted(WEB_SEARCH_PROVIDERS)),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if request.apply_patch:
            from .._capabilities import APPLY_PATCH_PROVIDERS

            if not self.resolved.capabilities.apply_patch:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} has no apply_patch tool; "
                    "apply_patch is supported on: " + ", ".join(sorted(APPLY_PATCH_PROVIDERS)),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
            if any(tool.name == "apply_patch" for tool in request.tools):
                # "apply_patch" is the name KeyCall reserves for the
                # provider-owned tool's own ToolCall/ToolResult parts; a
                # caller-defined tool with the same name would be
                # indistinguishable from it on replay.
                raise KeyCallError(
                    "a caller-defined tool cannot be named 'apply_patch' "
                    "while apply_patch=True — that name is reserved for "
                    "the apply_patch tool's own ToolCall/ToolResult parts",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if request.code_interpreter:
            from .._capabilities import CODE_INTERPRETER_PROVIDERS

            if not self.resolved.capabilities.code_interpreter:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} has no code interpreter "
                    "tool; code_interpreter is supported on: "
                    + ", ".join(sorted(CODE_INTERPRETER_PROVIDERS)),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if request.reasoning_effort is not None:
            from .._registry import providers_with

            if not self.resolved.capabilities.reasoning_effort:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} has no live-verified native "
                    "reasoning-effort control; reasoning_effort is supported on: "
                    + ", ".join(sorted(providers_with("reasoning_effort"))),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
            # "minimal" is narrower than the reasoning_effort capability
            # flag above: OpenAI's Responses API is the only place it's
            # live-verified. Every other reasoning-capable provider maps
            # the value straight through to its own native control, so
            # "minimal" would reach the wire as a level that control does
            # not define, refused live rather than caught here.
            if request.reasoning_effort == "minimal" and self.resolved.provider != "openai":
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} does not support the "
                    "'minimal' reasoning effort; only openai does. Use 'low', "
                    "'medium', or 'high' instead",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if (
            request.web_search
            and request.response_schema is not None
            and self.resolved.provider == "anthropic"
        ):
            # Not a guess: Anthropic's tool_choice={"type":"tool",...}, the
            # only mechanism KeyCall has for schema enforcement here, forces
            # the model to call exactly that tool and nothing else in the
            # same turn — mechanically incompatible with also invoking the
            # server-side web_search tool. This is an API constraint,
            # not a live-probed guess.
            raise KeyCallError(
                "anthropic cannot combine web_search with response_schema: "
                "forcing the structured-output tool prevents the model "
                "from also calling web_search in the same turn",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if (
            request.response_schema is not None
            and self.resolved.provider == "gemini"
            and _schema_has_key(request.response_schema, "additionalProperties")
        ):
            # Live-verified 2026-08-08: gemini's schema dialect rejects the
            # key with a 400 wherever it appears, not only at the top
            # level — the opposite of OpenAI's strict mode, which requires
            # it on every object level. One schema object cannot satisfy
            # both, and a schema generator that defaults to including it
            # (Pydantic's model_json_schema(), for one) will trip this on
            # every nested object unless it is stripped first. Caught here
            # rather than silently stripped: KeyCall doesn't know whether
            # the same schema object is about to be reused for a provider
            # that requires the key.
            raise KeyCallError(
                "gemini rejects any 'additionalProperties' key in "
                "response_schema, at any nesting depth. Strip it from the "
                "schema before calling gemini with it",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )

    def _check_media_part(self, part: Any, *, role: str, kind: str) -> None:
        """Media support splits by form, not just by provider: several
        providers read raw bytes and refuse to fetch a URL. Each refusal
        names the form that does work, or the providers that do."""
        capabilities = self.resolved.capabilities
        provider = self.resolved.provider
        takes_bytes = getattr(capabilities, f"{kind}_input_bytes")
        takes_url = getattr(capabilities, f"{kind}_input_url")
        noun = {"image": "image", "audio": "audio", "file": "file"}[kind]
        if role != "user":
            raise KeyCallError(
                f"{type(part).__name__} belongs in user messages, not role {role!r}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if not takes_bytes and not takes_url:
            supporting = sorted(providers_with(f"{kind}_input"))
            detail = (
                f" {noun.capitalize()} input is supported on: " + ", ".join(supporting)
                if supporting
                else ""
            )
            raise KeyCallError(
                f"provider {provider!r} does not accept {noun} input.{detail}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if part.url is not None and not takes_url:
            raise KeyCallError(
                f"provider {provider!r} does not fetch {noun} URLs; read the "
                f"{noun} and pass data=... instead. KeyCall will not fetch it "
                "for you: an adapter that made its own network request could "
                "be pointed anywhere by caller data",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if part.data is not None and not takes_bytes:
            raise KeyCallError(
                f"provider {provider!r} does not accept {noun} bytes; pass "
                "url=... instead",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if part.data is not None:
            media_type_for(part, kind=kind, provider=provider)

    def parse_tool_arguments(self, raw: Any) -> Mapping[str, Any]:
        return parse_tool_arguments(raw, provider=self.resolved.provider)

    @staticmethod
    def tool_result_text(content: Any) -> str:
        """ToolResult.content as the string most providers want."""
        return content if isinstance(content, str) else json.dumps(content)

    @staticmethod
    def sampling_fields(request: TextGenerationRequest) -> dict[str, float]:
        """temperature/top_p body fields, omitted when unset (the OpenAI-shaped
        field names, which Anthropic shares)."""
        fields: dict[str, float] = {}
        if request.temperature is not None:
            fields["temperature"] = request.temperature
        if request.top_p is not None:
            fields["top_p"] = request.top_p
        return fields
