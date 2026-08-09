"""Adapter interface: pure request-building and response-parsing.

Adapters never perform I/O and never see the credential. The client drives
the page loop and hands specs to the transport layer, which is the single
place credentials are revealed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._registry import ResolvedProvider
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    Citation,
    InvocationResult,
    Model,
    StreamEvent,
    TextGenerationRequest,
    TextOutput,
    Usage,
)


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
        self._text: list[str] = []

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

    def finalize(self, *, round_trip_duration_ms: float) -> InvocationResult:
        if self.provider_request_id is None and self.resolved.provider_request_id_header:
            self.provider_request_id = safe_request_id(
                self.response_headers.get(self.resolved.provider_request_id_header)
            )
        if not self.usage_reported:
            self.warnings.append("provider reported no usage information")
        return InvocationResult(
            provider=self.resolved.provider,
            model=self.model,
            operation=Operation.TEXT_GENERATION,
            parts=(TextOutput(text=self.text),) if self._text else (),
            usage=self.usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=self.provider_request_id,
            finish_reason=self.finish_reason,
            citations=tuple(self.citations),
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
        if status_code == 404:
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "not found"
        if status_code == 429:
            return ErrorCode.RATE_LIMITED, True, message or "rate limited"
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
        from .._capabilities import TOOL_CALLING_PROVIDERS, rejects_sampling_params
        from .._types import TextInput, ToolCall, ToolResult

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
                raise KeyCallError(
                    f"text generation supports text and tool parts only, "
                    f"got {type(part).__name__}",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    operation=Operation.TEXT_GENERATION.value,
                )
        if request.tools:
            if (
                not self.resolved.is_custom
                and self.resolved.provider not in TOOL_CALLING_PROVIDERS
            ):
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} does not support tool "
                    "calling; tools are supported on: "
                    + ", ".join(sorted(TOOL_CALLING_PROVIDERS)),
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
        if (request.temperature is not None or request.top_p is not None) and (
            rejects_sampling_params(request.model)
        ):
            raise KeyCallError(
                f"model {request.model!r} rejects temperature/top_p; remove the "
                "sampling parameters for this model",
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        if request.web_search:
            from .._capabilities import WEB_SEARCH_PROVIDERS

            if self.resolved.provider not in WEB_SEARCH_PROVIDERS:
                raise KeyCallError(
                    f"provider {self.resolved.provider!r} has no native web search "
                    "tool; web_search is supported on: "
                    + ", ".join(sorted(WEB_SEARCH_PROVIDERS)),
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
            # server-side web_search tool. This is a real API constraint,
            # not a live-probed guess.
            raise KeyCallError(
                "anthropic cannot combine web_search with response_schema: "
                "forcing the structured-output tool prevents the model "
                "from also calling web_search in the same turn",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )

    def parse_tool_arguments(self, raw: Any) -> Mapping[str, Any]:
        """Providers that send arguments as a JSON string (OpenAI, the
        compat family) get parsed here; malformed argument JSON from a
        provider is a typed error, never a silently dropped call."""
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
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        return parsed

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
