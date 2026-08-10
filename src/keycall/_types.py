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

from ._enums import ModelCategory, Operation

__all__ = [
    "AudioInput",
    "AudioOutput",
    "Citation",
    "CitationFound",
    "EmbeddingOutput",
    "EmbeddingRequest",
    "FileInput",
    "FileOutput",
    "ImageGenerationRequest",
    "ImageInput",
    "ImageOutput",
    "InputPart",
    "InvocationResult",
    "Message",
    "MessageRole",
    "Model",
    "ModelDiscovery",
    "OutputPart",
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
    "UnknownOutput",
    "UnknownStreamEvent",
    "Usage",
    "VideoOutput",
]

MessageRole = Literal["system", "user", "assistant"]
_VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant")


# --- input parts -----------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TextInput:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput:
    """Not yet accepted by text generation. The type is part of the
    directional content taxonomy and is validated here, but every adapter
    rejects it with UNSUPPORTED_OPERATION before any network call: no
    provider mapping has been built or verified. Present so the taxonomy is
    complete and so a caller can model content it will send later, not as a
    signal that image input works today.
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
    """Not yet accepted by text generation. The type is part of the
    directional content taxonomy and is validated here, but every adapter
    rejects it with UNSUPPORTED_OPERATION before any network call: no
    provider mapping has been built or verified. Present so the taxonomy is
    complete and so a caller can model content it will send later, not as a
    signal that audio input works today.
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
    """Not yet accepted by text generation. The type is part of the
    directional content taxonomy and is validated here, but every adapter
    rejects it with UNSUPPORTED_OPERATION before any network call: no
    provider mapping has been built or verified. Present so the taxonomy is
    complete and so a caller can model content it will send later, not as a
    signal that file input works today.
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
    tools; it normalizes the request/response wire shapes."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool.name must be a non-empty string")
        if not isinstance(self.input_schema, Mapping) or "type" not in self.input_schema:
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
    tool schemas is a bigger primitive KeyCall doesn't need until a real
    caller needs more than search. Providers without a native search tool
    (DeepSeek, Moonshot, custom targets) raise UNSUPPORTED_OPERATION rather
    than silently ignoring the request. Perplexity's Sonar always searches
    regardless of this flag; setting it False there is a no-op, warned."""
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
class CitationFound:
    """A web-search source surfaced during the stream."""

    citation: Citation
    kind: Literal["citation"] = "citation"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallStarted:
    """The model began requesting a tool. The name is known; the arguments
    are not yet. Never act on this event — wait for ToolCallComplete."""

    id: str
    name: str
    kind: Literal["tool_call_started"] = "tool_call_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallArgumentsDelta:
    """A fragment of a tool call's argument JSON, exactly as the provider
    sent it. Fragments are individually meaningless: they split mid-token
    and only the concatenation is valid JSON. Useful for showing progress,
    not for parsing. Providers that send arguments whole (Gemini) emit
    none of these, so a stream can go from started to complete with no
    deltas in between."""

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
    | CitationFound
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallComplete
    | StreamFinish
    | UnknownStreamEvent
)


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
class Model:
    id: str
    provider: str
    categories: frozenset[ModelCategory]
    display_name: str | None = None
    lifecycle: str | None = None
    context_limit: int | None = None
    capabilities: frozenset[str] = frozenset()
    classification_source: str = "unknown"
    warnings: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    """One normalized web-search source, across whichever shape the
    provider actually returned it in (OpenAI's text annotations, Anthropic's
    per-block citations, Gemini's grounding chunks, Perplexity's
    search_results). ``url`` is what the provider gave KeyCall — for
    Gemini this is a vertexaisearch.cloud.google.com redirect, not the
    direct source, by Google's own design; it resolves correctly when
    followed, KeyCall does not pre-resolve it."""

    url: str
    title: str | None = None
    cited_text: str | None = None


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
