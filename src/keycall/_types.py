"""Public data records: content parts, messages, requests, results.

Directional input/output content types with stable ``kind`` discriminators.
All records are keyword-only and frozen; sequences are accepted as any
Sequence and normalized to tuples internally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from ._enums import ModelCategory, Operation

__all__ = [
    "AliasFact",
    "AudioInput",
    "AudioOutput",
    "Citation",
    "CitationFound",
    "CodeExecutionOutput",
    "EmbeddingOutput",
    "EmbeddingRequest",
    "FileInput",
    "FileOutput",
    "FinalTranscript",
    "ImageGenerationRequest",
    "ImageInput",
    "ImageOutput",
    "InputPart",
    "InterimTranscript",
    "InvocationResult",
    "Message",
    "MessageRole",
    "Model",
    "ModelDiscovery",
    "OutputPart",
    "ReasoningDelta",
    "SpeechGenerationRequest",
    "StreamEvent",
    "StreamFinish",
    "StreamStart",
    "TextDelta",
    "TextGenerationRequest",
    "TextInput",
    "TextOutput",
    "Tool",
    "ToolCall",
    "ToolCallArgumentsDelta",
    "ToolCallComplete",
    "ToolCallStarted",
    "ToolResult",
    "TranscriptOutput",
    "TranscriptWord",
    "TranscriptionConfig",
    "TranscriptionEvent",
    "TranscriptionSessionEnded",
    "TranscriptionSessionStarted",
    "UnknownOutput",
    "UnknownStreamEvent",
    "UnknownTranscriptionEvent",
    "Usage",
    "VideoGenerationRequest",
    "VideoJob",
    "VideoJobStatus",
    "VideoOutput",
    "Voice",
]

MessageRole = Literal["system", "user", "assistant"]
_VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant")


# --- input parts -----------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TextInput:
    """``cacheable=True`` marks this block as a stable prefix boundary: on
    Anthropic and OpenAI (the two providers whose caching requires a caller
    marker rather than happening automatically), KeyCall sets that
    provider's own breakpoint here. Every other provider ignores the flag
    and keeps caching automatically, the same as it already does with no
    marker at all. ``cache_ttl_seconds`` only means something on Anthropic,
    which offers only two tiers (300s/"5m", 3600s/"1h"); any other value
    is refused before the network call rather than silently rounded, since
    the 1-hour tier is billed at double the base input rate."""

    text: str
    kind: Literal["text"] = "text"
    cacheable: bool = False
    cache_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput:
    """A picture for the model to look at, as bytes or a URL. Support is
    per-provider and per-form: OpenAI, Anthropic, and Perplexity take
    either, Gemini and Moonshot take bytes only, and DeepSeek takes
    neither. A form the provider can't accept is refused with
    UNSUPPORTED_OPERATION before any network call, so an unsupported
    attachment costs nothing. `media_type` is a hint; KeyCall sniffs the
    bytes and trusts what it finds over what the caller claimed.
    """

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    kind: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("ImageInput requires exactly one of url or data")


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioInput:
    """A sound file for the model to listen to. Gemini is the only provider
    that accepts one today, and only as bytes; every other provider refuses
    it with UNSUPPORTED_OPERATION before any network call. Sending audio to
    a text model is a different thing from transcription, which is its own
    model category KeyCall doesn't call.
    """

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    kind: Literal["audio"] = "audio"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("AudioInput requires exactly one of url or data")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileInput:
    """A document for the model to read, typically a PDF. OpenAI,
    Anthropic, and Gemini accept one as bytes; no provider accepts a URL,
    and DeepSeek, Perplexity, and Moonshot accept neither, so those are
    refused with UNSUPPORTED_OPERATION before any network call. `filename`
    is passed through where the provider shows it to the model, which is
    why a document keeps the name it had on disk.
    """

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    filename: str | None = None
    kind: Literal["file"] = "file"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("FileInput requires exactly one of url or data")


# --- tool calling ----------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Tool:
    """A caller-defined tool the model may request. KeyCall never executes
    tools; it normalizes the request/response wire shapes.

    ``input_schema=None`` declares a custom (freeform) tool instead of an
    ordinary JSON-Schema one: the model's call arrives as a plain string
    rather than parsed arguments, carried in the resulting ToolCall's
    ``arguments["input"]``. OpenAI-only (live-verified 2026-08-22); other
    providers raise UNSUPPORTED_OPERATION for a tool declared this way.

    ``defer_loading=True`` keeps this tool's definition out of the
    model's context until it searches for and finds it — a request-size
    optimization for callers registering many tools, not a behavior
    change to the tool itself. KeyCall sends the provider's tool-search
    tool automatically whenever any Tool in a request sets this; a
    discovered tool's call and reply are ordinary ToolCall/ToolResult
    parts, identical to a non-deferred tool's. OpenAI and Anthropic only
    (live-verified 2026-08-22); other providers raise
    UNSUPPORTED_OPERATION when this is set."""

    name: str
    description: str
    input_schema: Mapping[str, Any] | None
    defer_loading: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool.name must be a non-empty string")
        if self.input_schema is not None and (
            not isinstance(self.input_schema, Mapping) or "type" not in self.input_schema
        ):
            raise ValueError("Tool.input_schema must be a JSON Schema object with a 'type' key")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """The model's request to invoke a tool. Appears in result parts and is
    replayed inside an assistant message. ``opaque`` is adapter-owned echo
    data the provider requires back verbatim (e.g. Gemini's thought
    signature); callers must pass it through unmodified and never interpret
    it."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    opaque: str | None = None
    kind: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult:
    """The caller's answer to a ToolCall, sent inside a user message.
    ``content`` may be a string or a JSON-serializable mapping; adapters
    convert to each provider's required form."""

    tool_call_id: str
    name: str
    content: str | Mapping[str, Any]
    kind: Literal["tool_result"] = "tool_result"


InputPart = TextInput | ImageInput | AudioInput | FileInput | ToolCall | ToolResult
_INPUT_PART_TYPES = (TextInput, ImageInput, AudioInput, FileInput, ToolCall, ToolResult)


# --- output parts ----------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TextOutput:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageOutput:
    url: str | None = None
    base64_data: str | None = None
    media_type: str | None = None
    kind: Literal["image"] = "image"


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeExecutionOutput:
    """Provider-hosted code that ran during generation, and what it
    produced. Covers OpenAI's code_interpreter, Gemini's code_execution,
    xAI's code_interpreter, and Anthropic's bash_code_execution — the
    "code ran, produced a text result" case common to all four. A
    provider-generated file (e.g. a saved plot) surfaces separately: as an
    ImageOutput when the provider returns image bytes inline (Gemini), or
    not at all when the provider only returns an opaque file reference
    needing a further authenticated download (OpenAI, xAI, Anthropic) —
    that download path is not yet built. No ``opaque`` echo field: unlike
    a ToolCall, a caller never replays this back to the provider."""

    code: str
    output: str
    language: str | None = None
    kind: Literal["code_execution"] = "code_execution"


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioOutput:
    url: str | None = None
    base64_data: str | None = None
    media_type: str | None = None
    kind: Literal["audio"] = "audio"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptOutput:
    text: str
    kind: Literal["transcript"] = "transcript"


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingOutput:
    values: tuple[float, ...]
    kind: Literal["embedding"] = "embedding"


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoOutput:
    url: str | None = None
    base64_data: str | None = None
    media_type: str | None = None
    kind: Literal["video"] = "video"


@dataclass(frozen=True, slots=True, kw_only=True)
class FileOutput:
    url: str | None = None
    media_type: str | None = None
    filename: str | None = None
    kind: Literal["file"] = "file"


@dataclass(frozen=True, slots=True, kw_only=True)
class UnknownOutput:
    """A content type KeyCall doesn't recognize yet. Bounded, sanitized
    metadata only — never an unrestricted raw provider payload."""

    provider_kind: str
    kind: Literal["unknown"] = "unknown"


OutputPart = (
    TextOutput
    | ImageOutput
    | AudioOutput
    | TranscriptOutput
    | EmbeddingOutput
    | VideoOutput
    | FileOutput
    | ToolCall
    | CodeExecutionOutput
    | UnknownOutput
)


# --- messages and requests -------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    role: MessageRole
    content: Sequence[InputPart]

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got {self.role!r}")
        parts = tuple(self.content)
        if not parts:
            raise ValueError("Message.content must not be empty")
        for part in parts:
            if not isinstance(part, _INPUT_PART_TYPES):
                raise TypeError(
                    f"Message.content accepts typed input parts only, got {type(part).__name__}"
                )
        object.__setattr__(self, "content", parts)


@dataclass(frozen=True, slots=True, kw_only=True)
class TextGenerationRequest:
    """Carries no provider and no credential — those are client identity."""

    model: str
    messages: Sequence[Message]
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    web_search: bool = False
    """Enable the provider's native web search/retrieval tool. A bare
    boolean, not a general tool-calling surface — every provider that
    supports this takes it as one on/off switch, and adding configurable
    tool schemas is a bigger primitive KeyCall doesn't need until a
    caller needs more than search. Providers without a native search tool
    (DeepSeek, Moonshot, custom targets) raise UNSUPPORTED_OPERATION rather
    than silently ignoring the request. Perplexity's Sonar always searches
    regardless of this flag; setting it False there is a no-op, warned."""
    apply_patch: bool = False
    """Enable OpenAI's ``apply_patch`` tool: the model proposes file
    create/update/delete operations (V4A-format diffs), the caller executes
    them and replies with the outcome — the same round trip as a caller
    tool, but the schema is fixed and provider-owned, so there is nothing
    to declare in ``tools``. Calls and replies are ordinary ToolCall
    and ToolResult parts with ``name == "apply_patch"``; a caller-defined
    tool also named "apply_patch" is rejected rather than silently
    colliding with it. OpenAI-only (live-verified 2026-08-22); other
    providers raise UNSUPPORTED_OPERATION."""
    code_interpreter: bool = False
    """Enable the provider's hosted code-execution tool: the model writes
    and runs code server-side and the run's code and output come back as
    CodeExecutionOutput parts. Provider-run, not caller-run — unlike
    apply_patch and caller-defined tools, there is nothing for the caller
    to execute or reply to. A provider-generated file surfaces as an
    ImageOutput only when the provider returns its bytes inline (Gemini);
    OpenAI, xAI, and Anthropic instead return an opaque file reference
    that needs a further authenticated download KeyCall does not yet
    perform, so a code run that only produces a file (no text answer)
    currently loses that file. Supported on OpenAI, Gemini, xAI, and
    Anthropic; other providers raise UNSUPPORTED_OPERATION."""
    tools: Sequence[Tool] = ()
    """Caller-defined tools the model may request. KeyCall normalizes the
    definitions, the model's ToolCall parts, and ToolResult replies across
    providers; executing a tool and looping is the caller's job."""
    tool_choice: str | None = None
    """One of "auto" (default when tools are present), "required", or
    "none". Forcing a specific named tool is not yet supported: the named
    variants are unverified on most providers. Some providers reject
    "required" for some models (DeepSeek thinking models return 400); the
    provider's typed error is surfaced rather than KeyCall maintaining a
    rejection matrix."""
    response_schema: Mapping[str, Any] | None = None
    """A JSON Schema object the response must conform to. Enforced
    provider-side where the provider supports it (OpenAI, Anthropic,
    Gemini, Moonshot, Perplexity — see _capabilities.SCHEMA_ENFORCING_PROVIDERS);
    elsewhere KeyCall requests generic valid-JSON mode instead and adds a
    result warning, rather than claiming enforcement it can't deliver.
    result.text carries the JSON as a string on every provider, so callers
    parse the same way regardless of which mechanism produced it."""
    reasoning_effort: str | None = None
    """How hard a reasoning-capable model should think, in the provider's
    own vocabulary (commonly "low" / "medium" / "high"; OpenAI also takes
    "minimal"). Mapped to each provider's native control — KeyCall never
    converts the value, so a word the provider rejects comes back as the
    provider's own error naming what it takes. Reasoning spend was
    measured at 20x the answer's tokens on some models, which is what
    this exists to rein in. Providers without a live-verified native
    control refuse rather than silently ignoring the request."""

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ValueError("model must be a non-empty string")
        msgs = tuple(self.messages)
        if not msgs:
            raise ValueError("messages must not be empty")
        for message in msgs:
            if not isinstance(message, Message):
                raise TypeError(
                    "messages accepts Message objects only — no dicts or strings "
                    f"(got {type(message).__name__})"
                )
        object.__setattr__(self, "messages", msgs)
        tools = tuple(self.tools)
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(f"tools accepts Tool objects only (got {type(tool).__name__})")
        object.__setattr__(self, "tools", tools)
        if self.tool_choice is not None:
            if self.tool_choice not in ("auto", "required", "none"):
                raise ValueError("tool_choice must be one of 'auto', 'required', 'none'")
            if not tools:
                raise ValueError("tool_choice requires tools")
        if self.response_schema is not None and (
            not isinstance(self.response_schema, Mapping) or "type" not in self.response_schema
        ):
            raise ValueError("response_schema must be a JSON Schema object with a 'type' key")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 (exclusive) and 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingRequest:
    """Carries no provider and no credential — those are client identity.
    Inputs are embedded in order, and the result's parts come back in that
    same order so a caller can zip them against the inputs."""

    model: str
    inputs: Sequence[str]

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("EmbeddingRequest.inputs must not be empty")
        for value in inputs:
            if not isinstance(value, str):
                raise TypeError(
                    f"inputs accepts strings only (got {type(value).__name__})"
                )
            if not value:
                raise ValueError("inputs must not contain an empty string")
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageGenerationRequest:
    """Carries no provider and no credential — those are client identity.
    Deliberately just a model and a prompt: size and count are supported on
    OpenAI and ignored by Gemini's image models, and a parameter that
    silently does nothing on half the providers is worse than none."""

    model: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("ImageGenerationRequest.prompt must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechGenerationRequest:
    """Carries no provider and no credential — those are client identity.

    `voice` is included, unlike `size`/`count` on `ImageGenerationRequest`:
    both providers that support this operation (OpenAI, Gemini) take a
    named voice as a plain string, so it is not a parameter that goes
    unused on half the providers — it is either honored or the whole
    operation is refused before the network, same as everything else here.

    Typed as optional because it is optional on most of these models, but
    not all: on OpenAI, `gpt-4o-mini-tts` defaults a voice when it's
    omitted, while `tts-1` and `tts-1-hd` reject the call outright with
    "voice field required" (both live-verified 2026-08-12 — the passing
    case on one model does not generalize to the family, which is
    precisely what happened writing this docstring the first time).
    Gemini defaults a voice on every model checked. KeyCall does not pick
    a voice on a caller's behalf when a provider requires one: that would
    be a value the caller never asked for, and the provider's own refusal
    already names what is missing."""

    model: str
    text: str
    voice: str | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("SpeechGenerationRequest.text must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoGenerationRequest:
    """Carries no provider and no credential — those are client identity.

    `duration_seconds` and `aspect_ratio` are sent when given and omitted
    when not, so each provider applies its own default rather than one
    KeyCall invented; both providers that support this operation take both
    parameters. A value the provider refuses comes back as the provider's
    own error, naming its accepted range."""

    model: str
    prompt: str
    duration_seconds: int | None = None
    aspect_ratio: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("VideoGenerationRequest.prompt must not be empty")


VideoJobStatus = Literal["running", "succeeded", "failed"]


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoJob:
    """A handle to a video render in progress. Plain data with no
    credential inside, so it can be stored and polled later, from another
    process if needed, through a client bound to the same provider.

    ``status`` is a closed three-value set; ``provider_status`` carries
    the provider's own word for the state verbatim (xAI's ``expired``
    becomes ``failed`` here, with the original preserved). ``job_id`` is
    the provider's identifier in whatever form it uses — a bare UUID on
    xAI, an operation path on Gemini. On success ``video_url`` holds the
    provider's download location; on failure ``error_message`` holds the
    provider's own explanation, sanitized."""

    provider: str
    model: str
    job_id: str
    status: VideoJobStatus = "running"
    provider_status: str | None = None
    video_url: str | None = None
    error_message: str | None = None


# --- stream events ---------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamStart:
    """First event of a stream: the provider accepted the request."""

    model: str
    kind: Literal["stream_start"] = "stream_start"


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDelta:
    """An increment of generated text. With response_schema, deltas are
    fragments of the final JSON string."""

    text: str
    kind: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReasoningDelta:
    """An increment of a visible reasoning trace, on providers that stream
    one before the answer (DeepSeek, Moonshot, xAI). Exists so a reasoning
    model doesn't look hung: grok-4.6 was observed reasoning for 40
    seconds before its first answer token, and a consumer that renders
    only TextDelta shows nothing that whole time. The text is the
    provider's own visible trace, passed through as sent — display it,
    count it, or ignore it, but never mistake it for the answer."""

    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True, slots=True, kw_only=True)
class CitationFound:
    """A web-search source surfaced during the stream."""

    citation: Citation
    kind: Literal["citation"] = "citation"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallStarted:
    """The model began requesting a tool. The name is known; the arguments
    aren't yet. Never act on this event — wait for ToolCallComplete."""

    id: str
    name: str
    kind: Literal["tool_call_started"] = "tool_call_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallArgumentsDelta:
    """A fragment of a tool call's argument content, verbatim as the
    provider sent it — JSON arguments for a caller-defined tool, diff text
    for OpenAI's apply_patch. Fragments are individually meaningless: JSON
    ones split mid-token and only the concatenation parses. Useful for
    showing progress, not for parsing. Providers that send arguments whole
    (Gemini) emit none of these, so a stream can go from started to
    complete with no deltas in between."""

    id: str
    fragment: str
    kind: Literal["tool_call_arguments_delta"] = "tool_call_arguments_delta"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallComplete:
    """A fully assembled tool call, arguments parsed. This is the event to
    act on; the same ToolCall also appears in the final result's parts."""

    tool_call: ToolCall
    kind: Literal["tool_call_complete"] = "tool_call_complete"


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamFinish:
    """Last event of a completed stream. After this, ``stream.result()``
    returns the full InvocationResult."""

    finish_reason: str | None
    usage: Usage
    kind: Literal["stream_finish"] = "stream_finish"


@dataclass(frozen=True, slots=True, kw_only=True)
class UnknownStreamEvent:
    """A stream event KeyCall doesn't recognize yet. Bounded provider kind
    only — never a raw provider payload."""

    provider_kind: str
    kind: Literal["unknown"] = "unknown"


StreamEvent = (
    StreamStart
    | TextDelta
    | ReasoningDelta
    | CitationFound
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallComplete
    | StreamFinish
    | UnknownStreamEvent
)


# --- realtime events --------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeSessionStarted:
    """The provider accepted the session. On OpenAI and xAI this carries
    the provider's session id; Gemini's setup acknowledgement has none."""

    provider_session_id: str | None = None
    kind: Literal["session_started"] = "session_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeAudioDelta:
    """A chunk of generated audio, decoded to raw bytes. The encoding is
    the session's output format: 16-bit PCM unless the session was
    configured otherwise (OpenAI and xAI at 24 kHz, Gemini at 24 kHz)."""

    data: bytes
    kind: Literal["audio_delta"] = "audio_delta"


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeTranscriptDelta:
    """An increment of the words being spoken (or, in a text-modality
    session, the text answer itself). On voice-first providers this is the
    provider's own transcript of its audio; it can trail the audio deltas
    it describes."""

    text: str
    kind: Literal["transcript_delta"] = "transcript_delta"


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeTurnComplete:
    """The model finished a response turn. Token usage is reported where
    the provider reports it (OpenAI and Gemini; xAI answers none)."""

    usage: Usage
    kind: Literal["turn_complete"] = "turn_complete"


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeInterrupted:
    """The turn in progress was cut off — the caller (or the provider's
    own voice-activity detection) started a new turn before this one
    finished. Audio already emitted is not retracted."""

    kind: Literal["interrupted"] = "interrupted"


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeSessionEnded:
    """The connection closed. ``reason`` is the provider's scrubbed close
    message when it gave one."""

    reason: str | None = None
    kind: Literal["session_ended"] = "session_ended"


@dataclass(frozen=True, slots=True, kw_only=True)
class UnknownRealtimeEvent:
    """A realtime frame KeyCall doesn't recognize yet. Bounded provider
    kind only — never a raw provider payload."""

    provider_kind: str
    kind: Literal["unknown"] = "unknown"


RealtimeEvent = (
    RealtimeSessionStarted
    | RealtimeAudioDelta
    | RealtimeTranscriptDelta
    | RealtimeTurnComplete
    | RealtimeInterrupted
    | RealtimeSessionEnded
    | UnknownRealtimeEvent
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeConfig:
    """What a realtime session asks of the provider. The common core is
    normalized; ``provider_config`` is passed through verbatim into the
    provider's session-configuration message for everything KeyCall does
    not model, and its use is reported with a warning so a portability
    seam is never silent."""

    model: str
    voice: str | None = None
    instructions: str | None = None
    provider_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty string")


# --- streaming transcription events ----------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionSessionStarted:
    """The provider accepted the transcription session. AssemblyAI answers
    a Begin frame carrying its session id; Deepgram sends no
    acknowledgement frame at all, so no event of this kind arrives there —
    a successful connect is its only accept signal."""

    provider_session_id: str | None = None
    kind: Literal["session_started"] = "session_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptWord:
    """One recognized word inside a finalized transcript, with its timing
    in milliseconds from the start of the session's audio. ``confidence``
    is the provider's own 0-1 score where reported."""

    text: str
    start_ms: float
    end_ms: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InterimTranscript:
    """A provisional transcript of audio still being spoken. Superseded by
    later interims and by the FinalTranscript that closes the utterance —
    useful for live display, never for downstream processing. ``channel``
    is the provider's audio-channel index where it reports one (Deepgram);
    None on single-channel providers (AssemblyAI)."""

    text: str
    channel: int | None = None
    kind: Literal["interim_transcript"] = "interim_transcript"


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalTranscript:
    """A finalized stretch of transcript: this text will not change.
    ``utterance_end`` marks whether the speaker also finished the thought —
    Deepgram can finalize text mid-utterance (is_final without
    speech_final), in which case more finals belonging to the same
    utterance follow; AssemblyAI only finalizes whole turns, so it is
    always True there. ``confidence`` is the provider's overall score
    where reported (Deepgram); None where the provider only scores
    per-word (AssemblyAI — read ``words`` instead)."""

    text: str
    words: tuple[TranscriptWord, ...] = ()
    utterance_end: bool = True
    confidence: float | None = None
    channel: int | None = None
    kind: Literal["final_transcript"] = "final_transcript"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionSessionEnded:
    """The session closed. ``audio_duration_seconds`` is the provider's
    own count of billable audio processed — STT bills per second, not per
    token, so this is the usage figure for the session. Both AssemblyAI
    (Termination frame) and Deepgram (terminal Metadata frame) report it;
    None means the session ended without the provider's summary frame,
    e.g. a dropped connection."""

    reason: str | None = None
    audio_duration_seconds: float | None = None
    kind: Literal["session_ended"] = "session_ended"


@dataclass(frozen=True, slots=True, kw_only=True)
class UnknownTranscriptionEvent:
    """A transcription frame KeyCall doesn't recognize yet. Bounded
    provider kind only — never a raw provider payload."""

    provider_kind: str
    kind: Literal["unknown"] = "unknown"


TranscriptionEvent = (
    TranscriptionSessionStarted
    | InterimTranscript
    | FinalTranscript
    | TranscriptionSessionEnded
    | UnknownTranscriptionEvent
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionConfig:
    """What a streaming transcription session asks of the provider.
    ``model`` None means the provider's default streaming model.
    ``sample_rate`` is the rate of the 16-bit mono PCM audio the caller
    will send — the only audio form this surface takes; encode conversion
    is the caller's job."""

    model: str | None = None
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz")


# --- results ---------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Normalized usage. None means the provider didn't report a value;
    0 means the provider explicitly reported zero. Never conflate them."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    provider_units: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Voice:
    """One voice a provider can speak with. ``id`` is the exact value
    ``generate_speech(voice=...)`` sends; ``name`` is the human label
    (the same string where the provider makes no distinction).
    ``models`` scopes a voice to specific models where the provider's
    support varies that way (OpenAI's newest four voices are
    gpt-4o-mini-tts only); None means every speech model takes it.
    ``description`` carries whatever category or characteristic the
    provider reports, when it reports one."""

    provider: str
    id: str
    name: str
    description: str | None = None
    models: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AliasFact:
    """This model id is a rolling alias per its provider's recorded naming
    convention: a pointer the provider retargets over time, not a dated
    snapshot. ``maintained`` is the per-provider liveness fact — ``True``
    where the provider keeps the alias aimed at a live model (Gemini),
    ``False`` where the family was observed retired (OpenAI's
    ``-chat-latest``), ``None`` where the convention is recorded but
    liveness is unverified. ``verified`` dates the evidence; ``note``
    carries one sentence of it."""

    provider: str
    model_id: str
    convention: str
    maintained: bool | None
    verified: str
    note: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Model:
    id: str
    provider: str
    categories: frozenset[ModelCategory]
    display_name: str | None = None
    lifecycle: str | None = None
    # When the provider says this model appeared. Populated only where the
    # list endpoint reports it: OpenAI and Moonshot as a unix `created`,
    # Anthropic as an ISO `created_at`; Gemini and DeepSeek report nothing
    # and Perplexity has no list endpoint at all. Unlike a context window,
    # a missing value here costs a caller nothing, because the one consumer
    # (candidate ordering in verify) falls back to a different rule rather
    # than needing a number it can't get.
    released_at: datetime | None = None
    # Largest input the model accepts, in tokens, where the provider says.
    # Gemini, Anthropic, and Moonshot report it under three different
    # names; OpenAI and DeepSeek report nothing and Perplexity has no list
    # endpoint, so it is None there. Never inferred from a bundled table:
    # a caller budgets against this, and a wrong ceiling is worse than an
    # absent one it can branch on.
    context_limit: int | None = None
    capabilities: frozenset[str] = frozenset()
    classification_source: str = "unknown"
    warnings: tuple[str, ...] = ()
    # Present only when the id matches a recorded rolling-alias convention
    # for this provider. None covers both "dated/pinned id" and "no
    # convention recorded" — absent, never a guess.
    alias: AliasFact | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelDiscovery:
    provider: str
    models: tuple[Model, ...]
    categories: frozenset[ModelCategory]
    fetched_at: datetime
    from_cache: bool
    catalog_version: str
    catalog_stale: bool = False
    warnings: tuple[str, ...] = ()


def _without_tracking(url: str) -> str:
    """The URL with campaign-tracking parameters dropped.

    Deliberately narrow. Only keys beginning ``utm_`` go: they are a
    convention for attributing traffic and are ignored by the servers that
    receive them, so removing one can't change what the URL resolves to.
    Anything else — a document id, a page anchor, a signed parameter — is
    load-bearing somewhere, and guessing wrong would break the link rather
    than tidy it.

    Every surviving parameter is passed through byte for byte: its order,
    its case, and its exact percent-encoding. Parsing the query and
    re-encoding it would have been shorter, and wrong — that round trip
    rewrites ``%20`` as ``+`` and turns a valueless ``&flag`` into
    ``&flag=``. Both are usually equivalent and occasionally not, and a
    signed or opaque parameter is precisely where "usually" fails. So the
    query is split on ``&`` and filtered as text, and nothing that stays is
    ever re-serialized.
    """
    # Case-folded for the check only; the value tested is discarded and the
    # URL itself is never lowercased. Without this the fast path would skip
    # a provider that capitalizes the key.
    if "utm_" not in url.lower():
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        # Not parseable: hand it back rather than damage it.
        return url
    if not parts.query:
        return url
    segments = parts.query.split("&")
    kept = [
        segment
        for segment in segments
        # The key is everything before the first "=", or the whole segment
        # when there is no "=" at all.
        if not segment.split("=", 1)[0].lower().startswith("utm_")
    ]
    if len(kept) == len(segments):
        return url
    return urlunsplit(parts._replace(query="&".join(kept)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    """One normalized web-search source, across whichever shape the
    provider actually returned it in (OpenAI's text annotations, Anthropic's
    per-block citations, Gemini's grounding chunks, Perplexity's
    search_results).

    ``url`` is the provider's, with campaign-tracking parameters removed.
    OpenAI appends ``?utm_source=openai`` to every cited URL (verified
    2026-08-10; Anthropic, Gemini, and Perplexity append nothing), which
    attributes the click to OpenAI in the destination site's analytics and
    follows the link into whatever a caller renders, logs, or stores.
    Nothing about it identifies the source, so it is stripped here rather
    than at each of the nine places a citation is built — normalizing at
    the type is what stops a new adapter or a streamed event slipping past.
    There is no API option to suppress it: OpenAI's web_search tool exposes
    eight settings and none concerns tracking.

    Only the ``utm_*`` family is removed, because it never changes what a
    URL resolves to. Every other parameter is left exactly as it arrived,
    including Gemini's vertexaisearch.cloud.google.com redirect, which is
    the direct source by Google's own design and which KeyCall doesn't
    pre-resolve.
    """

    url: str
    title: str | None = None
    cited_text: str | None = None

    def __post_init__(self) -> None:
        cleaned = _without_tracking(self.url)
        if cleaned != self.url:
            object.__setattr__(self, "url", cleaned)


@dataclass(frozen=True, slots=True, kw_only=True)
class InvocationResult:
    provider: str
    model: str
    operation: Operation
    parts: tuple[OutputPart, ...]
    usage: Usage
    round_trip_duration_ms: float
    provider_request_id: str | None = None
    finish_reason: str | None = None
    citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def text(self) -> str | None:
        """Concatenated text output, or None if no text parts exist.
        Its presence must not imply every invocation produces text."""
        texts = [part.text for part in self.parts if isinstance(part, TextOutput)]
        return "".join(texts) if texts else None

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The model's tool-call requests, in order. Several per response
        is normal, not an edge case."""
        return tuple(part for part in self.parts if isinstance(part, ToolCall))

    @property
    def code_executions(self) -> tuple[CodeExecutionOutput, ...]:
        """Provider-hosted code runs, in order. Empty unless the request
        set code_interpreter=True."""
        return tuple(part for part in self.parts if isinstance(part, CodeExecutionOutput))

    def to_assistant_message(self) -> Message:
        """This response as an assistant Message for conversation replay:
        text parts become TextInput, ToolCall parts carry over (including
        their provider echo data). Non-text, non-tool parts are dropped."""
        content: list[InputPart] = []
        for part in self.parts:
            if isinstance(part, TextOutput):
                content.append(TextInput(text=part.text))
            elif isinstance(part, ToolCall):
                content.append(part)
        return Message(role="assistant", content=content)
