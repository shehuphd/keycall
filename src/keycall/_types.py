"""Public data records: content parts, messages, requests, results.

Directional input/output content types with stable ``kind`` discriminators
(naming-final.md section 6). All records are keyword-only and frozen;
sequences are accepted as any Sequence and normalized to tuples internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ._enums import ModelCategory, Operation

__all__ = [
    "AudioInput",
    "AudioOutput",
    "Citation",
    "EmbeddingOutput",
    "FileInput",
    "FileOutput",
    "ImageInput",
    "ImageOutput",
    "InputPart",
    "InvocationResult",
    "Message",
    "MessageRole",
    "Model",
    "ModelDiscovery",
    "OutputPart",
    "TextGenerationRequest",
    "TextInput",
    "TextOutput",
    "TranscriptOutput",
    "UnknownOutput",
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
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    kind: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("ImageInput requires exactly one of url or data")


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioInput:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    kind: Literal["audio"] = "audio"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("AudioInput requires exactly one of url or data")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileInput:
    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    filename: str | None = None
    kind: Literal["file"] = "file"

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("FileInput requires exactly one of url or data")


InputPart = TextInput | ImageInput | AudioInput | FileInput
_INPUT_PART_TYPES = (TextInput, ImageInput, AudioInput, FileInput)


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
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 (exclusive) and 1")


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
