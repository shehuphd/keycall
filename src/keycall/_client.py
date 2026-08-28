"""KeyCall and AsyncKeyCall clients.

Provider, credential, protocol, and base URL are immutable client identity,
bound once at construction with no setters and no per-call override.
The raw credential is wrapped in the redacting Credential type here, at
its single entry boundary; only the transport layer ever reveals it again.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._realtime import AsyncRealtimeSession, RealtimeSession
    from ._transcription import AsyncTranscriptionSession, TranscriptionSession

import httpx

from . import _cache, _capabilities, _tracing
from ._cache import CachedModels
from ._credential import Credential
from ._enums import ModelCategory, ProviderProtocol
from ._errors import ErrorCode, KeyCallError, VideoJobTimeout
from ._registry import (
    ResolvedProvider,
    catalog_age_days,
    catalog_is_stale,
    catalog_version,
    resolve_provider,
)
from ._transport import AsyncTransport, Transport
from ._types import (
    EmbeddingRequest,
    ImageGenerationRequest,
    InvocationResult,
    Message,
    Model,
    ModelDiscovery,
    RealtimeConfig,
    SpeechGenerationRequest,
    StreamEvent,
    TextGenerationRequest,
    Tool,
    TranscriptionConfig,
    Usage,
    VideoGenerationRequest,
    VideoJob,
)
from .adapters import ProviderAdapter, adapter_for
from .adapters._base import InbandStreamError, StreamAssembler

__all__ = ["AsyncKeyCall", "AsyncTextStream", "KeyCall", "TextStream"]

_MAX_LIST_PAGES = 10
_DEFAULT_CATEGORIES = frozenset({ModelCategory.TEXT_GENERATION})


def _validate_categories(
    categories: set[ModelCategory] | frozenset[ModelCategory] | None,
) -> frozenset[ModelCategory]:
    if categories is None:
        return _DEFAULT_CATEGORIES
    validated = set()
    for category in categories:
        if not isinstance(category, ModelCategory):
            raise TypeError(
                "categories accepts ModelCategory members only; plain strings are "
                "validated at config/CLI boundaries, not here "
                f"(got {type(category).__name__})"
            )
        validated.add(category)
    if not validated:
        return _DEFAULT_CATEGORIES
    return frozenset(validated)


def _with_schema_warning(
    invocation: InvocationResult, request: TextGenerationRequest, provider: str
) -> InvocationResult:
    """Append a warning when response_schema was requested but this
    provider only guarantees valid JSON, not schema conformance: never
    claim enforcement that isn't delivered — the same posture as the
    unreported-usage and stale-catalog warnings elsewhere."""
    if request.response_schema is None or provider in _capabilities.SCHEMA_ENFORCING_PROVIDERS:
        return invocation
    warnings = list(invocation.warnings)
    warnings.append(
        f"provider {provider!r} does not enforce response_schema; the response "
        "is guaranteed valid JSON but not guaranteed to match the schema — "
        "validate client-side"
    )
    if not _capabilities.mentions_json(request.messages):
        warnings.append(
            "keycall added a 'respond only with JSON' system instruction "
            f"because {provider!r} requires the word 'json' to appear in the "
            "prompt for its JSON response mode"
        )
    return dataclasses.replace(invocation, warnings=tuple(warnings))


# Every provider says "I ran out of output budget" in its own words:
# OpenAI Responses reports incomplete_details.reason, which the adapter
# renders as "incomplete:max_output_tokens"; Anthropic says "max_tokens";
# Gemini "MAX_TOKENS"; the Chat Completions family "length". A caller
# should not have to learn four spellings to notice its answer was cut off.
_TRUNCATION_REASONS = frozenset(
    {"incomplete:max_output_tokens", "max_tokens", "length"}
)


def was_truncated(finish_reason: str | None) -> bool:
    """Whether the provider stopped because the output budget ran out,
    normalized across the four wire protocols. Case-insensitive because
    Gemini shouts its finish reasons and the rest do not."""
    if not finish_reason:
        return False
    return finish_reason.lower() in _TRUNCATION_REASONS


def _with_truncation_warning(invocation: InvocationResult) -> InvocationResult:
    """Say plainly that the answer is incomplete and what to change.

    The finish reason already carries this, but only to a reader who knows
    that provider's vocabulary, and it sits beside timing and token counts
    where it reads as one more statistic. Reasoning models make it easy to
    hit: their hidden reasoning is billed against the same output budget,
    so a small max_output_tokens can be spent before any answer is emitted.
    """
    if not was_truncated(invocation.finish_reason):
        return invocation
    return dataclasses.replace(
        invocation,
        warnings=(
            *invocation.warnings,
            (
                "the reply stopped because max_output_tokens ran out, so it is "
                "cut off mid-answer — raise max_output_tokens and send again. "
                "On a reasoning model the hidden reasoning is charged to the "
                "same budget, so it can be used up before any text appears"
            ),
        ),
    )


def _with_custom_tool_warning(
    invocation: InvocationResult,
    request: TextGenerationRequest,
    provider: str,
    *,
    is_custom: bool,
) -> InvocationResult:
    """Custom targets get tools passed through unverified, so say so on
    every result rather than implying the endpoint honored them."""
    if not request.tools or not is_custom:
        return invocation
    return dataclasses.replace(
        invocation,
        warnings=(
            *invocation.warnings,
            (
                f"tool calling on custom target {provider!r} is unverified; "
                "keycall passes the standard tools field through without "
                "evidence the endpoint honors it"
            ),
        ),
    )


def _filter_models(
    models: tuple[Model, ...], categories: frozenset[ModelCategory]
) -> tuple[Model, ...]:
    # Unknown models never enter a picker unless UNKNOWN was requested.
    return tuple(model for model in models if model.categories & categories)


def _build_discovery(
    *,
    provider: str,
    cached: CachedModels,
    categories: frozenset[ModelCategory],
    from_cache: bool,
) -> ModelDiscovery:
    warnings = list(cached.warnings)
    stale = catalog_is_stale()
    if stale:
        # The bundled catalog carries endpoints, auth schemes, and
        # capability evidence. When it is this old the evidence predates
        # provider changes KeyCall hasn't seen, so say so instead of
        # presenting it as current.
        warnings.append(
            f"keycall's bundled provider catalog was last verified "
            f"{catalog_age_days()} days ago (version {catalog_version()}); "
            "provider endpoints and capabilities may have changed since — "
            "upgrade keycall for the current catalog"
        )
    return ModelDiscovery(
        provider=provider,
        models=_filter_models(cached.models, categories),
        categories=categories,
        fetched_at=cached.fetched_at,
        from_cache=from_cache,
        catalog_version=catalog_version(),
        catalog_stale=stale,
        warnings=tuple(warnings),
    )


class _BaseClient:
    __slots__ = ("_adapter", "_credential", "_resolved", "_transport")

    _resolved: ResolvedProvider
    _credential: Credential | None
    _adapter: ProviderAdapter
    _transport: Any

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        # Wrap the credential first so no later failure path ever handles
        # the raw string.
        credential = Credential(api_key)
        resolved = resolve_provider(
            provider,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        object.__setattr__(self, "_resolved", resolved)
        object.__setattr__(self, "_credential", credential)
        object.__setattr__(self, "_adapter", adapter_for(resolved))
        object.__setattr__(self, "_transport", None)

    # Immutable identity: no attribute may be rebound after construction.
    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError(
            f"{type(self).__name__} identity is immutable; construct a new client instead"
        )

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(
            f"{type(self).__name__} identity is immutable; construct a new client instead"
        )

    @property
    def provider(self) -> str:
        return self._resolved.provider

    @property
    def protocol(self) -> ProviderProtocol:
        return self._resolved.protocol

    @property
    def base_url(self) -> str:
        return self._resolved.base_url

    # Deliberately no api_key property.

    @property
    def closed(self) -> bool:
        return self._credential is None

    def _require_open(self) -> Credential:
        credential = self._credential
        if credential is None:
            raise RuntimeError(f"{type(self).__name__} is closed; construct a new client")
        return credential

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"protocol={self.protocol.value!r}, {state})"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("clients hold credentials and cannot be pickled or copied")

    # --- pure logic shared by the sync and async clients; only the awaits
    # --- differ in the public methods below.

    def _cached_discovery(
        self, categories: frozenset[ModelCategory], fingerprint: str, trace: Any
    ) -> ModelDiscovery | None:
        cached = _cache.shared_cache.get(self.provider, self.base_url, fingerprint)
        if cached is None:
            return None
        trace.event("app", operation="cache_hit", status="ok")
        return _build_discovery(
            provider=self.provider, cached=cached, categories=categories, from_cache=True
        )

    def _parse_page(self, trace: Any, spec: Any, result: Any) -> tuple[list[Model], Any]:
        trace.event(
            "http",
            operation=f"{spec.method} {spec.path}",
            target=self.provider,
            status=str(result.status_code),
            duration_ms=result.duration_ms,
        )
        return self._adapter.parse_model_page(result.payload)

    def _store_discovery(
        self,
        models: list[Model],
        *,
        truncated: bool,
        categories: frozenset[ModelCategory],
        fingerprint: str,
        trace: Any,
    ) -> ModelDiscovery:
        warnings: tuple[str, ...] = ()
        if truncated:
            warnings = (
                (
                    f"provider reported more model pages after the "
                    f"{_MAX_LIST_PAGES}-page limit; this list is truncated"
                ),
            )
        cached = CachedModels(
            models=tuple(models), fetched_at=datetime.now(timezone.utc), warnings=warnings
        )
        _cache.shared_cache.put(self.provider, self.base_url, fingerprint, cached)
        discovery = _build_discovery(
            provider=self.provider, cached=cached, categories=categories, from_cache=False
        )
        trace.event(
            "model",
            operation="normalize",
            status="ok",
            result={"models": len(models), "filtered": len(discovery.models)},
        )
        return discovery

    def _image_spec(self, request: ImageGenerationRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_image_spec, whose
        # default covers every adapter without an implementation.
        return self._adapter.build_image_spec(request)

    def _parse_image(
        self, request: ImageGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_image_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        trace.event(
            "model",
            operation="image_generation",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"images": len(invocation.parts)},
        )
        return invocation

    def _speech_spec(self, request: SpeechGenerationRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_speech_spec, whose
        # default covers every adapter without an implementation.
        return self._adapter.build_speech_spec(request)

    def _parse_speech(
        self, request: SpeechGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_speech_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        trace.event(
            "model",
            operation="speech_generation",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"clips": len(invocation.parts)},
        )
        return invocation

    def _require_video_job(self, job: VideoJob) -> None:
        if not isinstance(job, VideoJob):
            raise TypeError(f"expected a VideoJob, got {type(job).__name__}")
        if job.provider != self.provider:
            raise KeyCallError(
                f"this job belongs to provider {job.provider!r}; this client is bound "
                f"to {self.provider!r} and its credential must not poll another "
                "provider's job",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.provider,
                operation="video_generation",
            )

    def _raise_video_failure(self, job: VideoJob) -> NoReturn:
        detail = job.error_message or "no detail from the provider"
        state = f" ({job.provider_status})" if job.provider_status else ""
        raise KeyCallError(
            f"video generation failed{state}: {detail}",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.provider,
            operation="video_generation",
        )

    def _video_result_from_download(self, job: VideoJob, result: Any) -> InvocationResult:
        import base64 as _b64

        if not isinstance(result.payload, bytes) or not result.payload:
            raise KeyCallError(
                "video download did not return the file's bytes",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.provider,
                operation="video_generation",
            )
        media_type = (
            str(result.headers.get("content-type", "video/mp4")).split(";")[0].strip()
        )
        return self._adapter.video_result(
            base64_data=_b64.b64encode(result.payload).decode("ascii"),
            media_type=media_type,
            url=job.video_url,
            model=job.model,
            round_trip_duration_ms=result.duration_ms,
        )

    def _embedding_spec(self, request: EmbeddingRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_embedding_spec, whose
        # default raises for every adapter that hasn't implemented one.
        # Gating here as well would duplicate the message in a second
        # place that could drift from it.
        return self._adapter.build_embedding_spec(request)

    def _parse_embedding(
        self, request: EmbeddingRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_embedding_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
            expected=len(request.inputs),
        )
        trace.event(
            "model",
            operation="embedding",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"inputs": len(request.inputs), "vectors": len(invocation.parts)},
        )
        return invocation

    def _generation_spec(self, request: TextGenerationRequest) -> Any:
        if not isinstance(request, TextGenerationRequest):
            raise KeyCallError(
                f"invoke() accepts typed request objects, got {type(request).__name__}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        return self._adapter.build_generation_spec(request)

    def _parse_invocation(
        self, request: TextGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_generation_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        invocation = _with_schema_warning(invocation, request, self.provider)
        invocation = _with_truncation_warning(invocation)
        invocation = _with_custom_tool_warning(
            invocation, request, self.provider, is_custom=self._resolved.is_custom
        )
        trace.event(
            "model",
            operation="text_generation",
            target=invocation.model,
            status=invocation.finish_reason or "ok",
            duration_ms=invocation.round_trip_duration_ms,
            result={
                "input_tokens": invocation.usage.input_tokens,
                "output_tokens": invocation.usage.output_tokens,
                "parts": len(invocation.parts),
            },
        )
        return invocation


# A provider that keeps demanding server-tool echo rounds past this many
# billable calls is broken; refusing beats an unbounded spend.
_SERVER_TOOL_ROUND_BUDGET = 5


def _added(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _merged_usage(carried: Usage, current: Usage) -> Usage:
    """Token spend across the rounds of one logical call (Moonshot's
    web-search echo loop). Sums where both rounds report, keeps the
    reporting side where only one does."""
    return Usage(
        input_tokens=_added(carried.input_tokens, current.input_tokens),
        output_tokens=_added(carried.output_tokens, current.output_tokens),
        cached_input_tokens=_added(carried.cached_input_tokens, current.cached_input_tokens),
        reasoning_tokens=_added(carried.reasoning_tokens, current.reasoning_tokens),
        total_tokens=_added(carried.total_tokens, current.total_tokens),
        provider_units=current.provider_units or carried.provider_units,
    )


def _hide_server_tool_event(event: StreamEvent, hidden_ids: set[str]) -> bool:
    """Whether a stream event belongs to a server-side builtin tool's echo
    handshake rather than to the answer. Those calls are KeyCall's to
    complete, and surfacing them would tell the caller to act on a call
    that is not theirs."""
    if event.kind == "tool_call_started" and event.name.startswith("$"):
        hidden_ids.add(event.id)
        return True
    if event.kind == "tool_call_arguments_delta" and event.id in hidden_ids:
        return True
    return bool(
        event.kind == "tool_call_complete" and event.tool_call.name.startswith("$")
    )


class _StreamCore:
    """State shared by the sync and async stream wrappers."""

    def __init__(self, client: _BaseClient, request: TextGenerationRequest) -> None:
        self._client = client
        self._request = request
        self._assembler: StreamAssembler = client._adapter.stream_assembler(request)
        self._spec = client._adapter.build_stream_spec(request)
        self._started_at: float | None = None
        self._result: InvocationResult | None = None
        self._failed = False
        # Server-tool echo rounds (Moonshot web search): usage carried
        # from finished rounds, and how many rounds have run.
        self._carry_usage: Usage | None = None
        self._rounds = 1

    def _continuation(self) -> TextGenerationRequest | None:
        """After a round's terminal event: the follow-up request when the
        provider still owes the answer (a server-side tool wants its echo),
        None when this round's result is the answer. Advancing rebuilds the
        assembler and spec for the next round and banks this round's
        usage."""
        interim = self._assembler.finalize(round_trip_duration_ms=0.0)
        follow_up: TextGenerationRequest | None = (
            self._client._adapter.server_tool_continuation(self._request, interim)
        )
        if follow_up is None:
            return None
        if self._rounds >= _SERVER_TOOL_ROUND_BUDGET:
            self._failed = True
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self._client.provider,
                operation="text_generation",
                retryable=True,
            )
        self._carry_usage = (
            interim.usage
            if self._carry_usage is None
            else _merged_usage(self._carry_usage, interim.usage)
        )
        self._rounds += 1
        self._request = follow_up
        self._assembler = self._client._adapter.stream_assembler(follow_up)
        self._assembler.response_headers = {}
        self._spec = self._client._adapter.build_stream_spec(follow_up)
        return follow_up

    def _feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        try:
            return self._assembler.feed(event_name, data)
        except InbandStreamError as exc:
            self._failed = True
            raise KeyCallError(
                self._client._transport._scrub(exc.raw_message),
                code=exc.code,
                provider=self._client.provider,
                operation="text_generation",
                retryable=exc.retryable,
            ) from None
        except KeyCallError:
            self._failed = True
            raise

    def _check_terminal(self) -> None:
        """The stream closed; without the provider's terminal signal that is
        a truncation, never a completion."""
        if not self._assembler.saw_terminal:
            self._failed = True
            raise KeyCallError(
                "the stream ended before the provider's terminal event; "
                "the response is incomplete",
                code=ErrorCode.NETWORK_ERROR,
                provider=self._client.provider,
                operation="text_generation",
                retryable=True,
            )

    def _build_result(self) -> InvocationResult:
        if self._failed or not self._assembler.saw_terminal:
            raise KeyCallError(
                "the stream did not complete; no result is available",
                code=ErrorCode.NETWORK_ERROR,
                provider=self._client.provider,
                operation="text_generation",
            )
        if self._result is None:
            duration = (
                (time.monotonic() - self._started_at) * 1000.0 if self._started_at else 0.0
            )
            invocation = self._assembler.finalize(round_trip_duration_ms=duration)
            if self._carry_usage is not None:
                # The clock spans every round already; only tokens carry.
                invocation = dataclasses.replace(
                    invocation, usage=_merged_usage(self._carry_usage, invocation.usage)
                )
            invocation = _with_schema_warning(invocation, self._request, self._client.provider)
            invocation = _with_truncation_warning(invocation)
            self._result = _with_custom_tool_warning(
                invocation,
                self._request,
                self._client.provider,
                is_custom=self._client._resolved.is_custom,
            )
        return self._result


class TextStream(_StreamCore):
    """Iterate typed stream events; call result() after exhaustion for the
    full InvocationResult. The context manager owns the connection: leaving
    the block closes it, even on early break or exception."""

    def __enter__(self) -> Self:
        # Before the request goes out: time to first byte is part of the
        # round trip, and on providers that buffer it is most of it.
        self._started_at = time.monotonic()
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = self._ctx.__enter__()
        self._assembler.response_headers = headers
        self._events = events
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._ctx.__exit__(exc_type, exc, tb)

    def __iter__(self) -> Any:
        while True:
            # The round's finish event is held back until it is known to be
            # the last round: an echo round's finish is plumbing, not the
            # end of the answer.
            held: list[StreamEvent] = []
            hidden_ids: set[str] = set()
            for event_name, data in self._events:
                for event in self._feed(event_name, data):
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
                if self._assembler.saw_terminal:
                    break
            if not self._assembler.saw_terminal:
                for event in self._assembler.on_close():
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
            self._check_terminal()
            if self._continuation() is None:
                yield from held
                return
            self._ctx.__exit__(None, None, None)
            self._ctx = self._client._transport.stream_request(
                self._spec,
                operation="text_generation",
                translate_error=self._client._adapter.translate_error,
            )
            headers, events = self._ctx.__enter__()
            self._assembler.response_headers = headers
            self._events = events

    def result(self) -> InvocationResult:
        return self._build_result()


class AsyncTextStream(_StreamCore):
    """Async twin of TextStream."""

    async def __aenter__(self) -> Self:
        # See TextStream.__enter__: the clock starts before the request.
        self._started_at = time.monotonic()
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = await self._ctx.__aenter__()
        self._assembler.response_headers = headers
        self._events = events
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._ctx.__aexit__(exc_type, exc, tb)

    async def __aiter__(self) -> Any:
        while True:
            # See TextStream.__iter__: echo rounds hold their finish back.
            held: list[StreamEvent] = []
            hidden_ids: set[str] = set()
            async for event_name, data in self._events:
                for event in self._feed(event_name, data):
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
                if self._assembler.saw_terminal:
                    break
            if not self._assembler.saw_terminal:
                for event in self._assembler.on_close():
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
            self._check_terminal()
            if self._continuation() is None:
                for event in held:
                    yield event
                return
            await self._ctx.__aexit__(None, None, None)
            self._ctx = self._client._transport.stream_request(
                self._spec,
                operation="text_generation",
                translate_error=self._client._adapter.translate_error,
            )
            headers, events = await self._ctx.__aenter__()
            self._assembler.response_headers = headers
            self._events = events

    def result(self) -> InvocationResult:
        return self._build_result()


class KeyCall(_BaseClient):
    """Synchronous client. See AsyncKeyCall for the awaitable equivalent."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        trust_env: bool = True,
        httpx_transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        object.__setattr__(
            self,
            "_transport",
            Transport(
                self._resolved,
                self._require_open(),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_response_bytes=max_response_bytes,
                trust_env=trust_env,
                allow_private_network=allow_private_network,
                httpx_transport=httpx_transport,
            ),
        )

    def close(self) -> None:
        """Release the credential reference and the HTTP client."""
        self._transport.close()
        object.__setattr__(self, "_credential", None)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def list_models(
        self,
        *,
        categories: set[ModelCategory] | frozenset[ModelCategory] | None = None,
        refresh: bool = False,
    ) -> ModelDiscovery:
        credential = self._require_open()
        requested = _validate_categories(categories)
        fingerprint = credential.fingerprint()

        with _tracing.span(
            "keycall.list_models", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            if not refresh:
                cached = self._cached_discovery(requested, fingerprint, trace)
                if cached is not None:
                    return cached

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            next_spec = spec
            for _ in range(_MAX_LIST_PAGES):
                result = self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page_models, next_spec = self._parse_page(trace, spec, result)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec
            return self._store_discovery(
                models,
                truncated=next_spec is not None,
                categories=requested,
                fingerprint=fingerprint,
                trace=trace,
            )

    def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            carried: Usage | None = None
            carried_ms = 0.0
            for _round in range(_SERVER_TOOL_ROUND_BUDGET):
                spec = self._generation_spec(request)
                result = self._transport.request(
                    spec,
                    operation="text_generation",
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                invocation = self._parse_invocation(request, result, trace)
                follow_up = self._adapter.server_tool_continuation(request, invocation)
                if follow_up is None:
                    if carried is not None:
                        invocation = dataclasses.replace(
                            invocation,
                            usage=_merged_usage(carried, invocation.usage),
                            round_trip_duration_ms=carried_ms
                            + invocation.round_trip_duration_ms,
                        )
                    return invocation
                # A server-side tool wants its echo before the answer
                # comes (Moonshot web search); the tokens and time spent
                # this round belong to the one logical call.
                carried = (
                    invocation.usage
                    if carried is None
                    else _merged_usage(carried, invocation.usage)
                )
                carried_ms += invocation.round_trip_duration_ms
                request = follow_up
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self.provider,
                operation="text_generation",
                retryable=True,
            )

    def generate_image(self, *, model: str, prompt: str) -> InvocationResult:
        """Generate a picture. The result's parts are ImageOutput values
        carrying base64 data and the media type the provider produced."""
        self._require_open()
        request = ImageGenerationRequest(model=model, prompt=prompt)
        spec = self._image_spec(request)
        with _tracing.span(
            "keycall.image_generation", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="image_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_image(request, result, trace)

    def embed(self, *, model: str, inputs: Sequence[str]) -> InvocationResult:
        """Embed one or more strings. The result's parts are EmbeddingOutput
        values in the order the inputs were given, so they zip together."""
        self._require_open()
        request = EmbeddingRequest(model=model, inputs=inputs)
        spec = self._embedding_spec(request)
        with _tracing.span("keycall.embedding", provider=self.provider, model=model) as trace:
            result = self._transport.request(
                spec,
                operation="embedding",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_embedding(request, result, trace)

    def generate_speech(
        self, *, model: str, text: str, voice: str | None = None
    ) -> InvocationResult:
        """Speak text aloud. The result's one part is an AudioOutput
        carrying base64 data and the media type the provider produced —
        not necessarily a playable container; Gemini answers with raw PCM
        and says so in the media type."""
        self._require_open()
        request = SpeechGenerationRequest(model=model, text=text, voice=voice)
        spec = self._speech_spec(request)
        with _tracing.span(
            "keycall.speech_generation", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="speech_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_speech(request, result, trace)

    def start_video(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> VideoJob:
        """Start a video render and return its job handle immediately.
        Rendering takes anywhere from seconds to many minutes depending on
        provider load; poll with check_video(), then fetch_video() once
        the job reports succeeded — or let generate_video() do all three
        against a waiting budget."""
        self._require_open()
        request = VideoGenerationRequest(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        spec = self._adapter.build_video_start_spec(request)
        with _tracing.span(
            "keycall.video_generation.start", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="video_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_video_start(result.payload, model=request.model)
            trace.event(
                "model",
                operation="video_generation",
                target=model,
                duration_ms=result.duration_ms,
                result={"job": "started"},
            )
            return job

    def check_video(self, job: VideoJob) -> VideoJob:
        """Ask the provider where a render stands. Returns a new VideoJob
        rather than mutating; a job that already finished is returned
        as-is without a network call."""
        self._require_open()
        self._require_video_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_video_status_spec(job)
        result = self._transport.request(
            spec,
            operation="video_generation",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_video_status(result.payload, job=job)

    def fetch_video(self, job: VideoJob) -> InvocationResult:
        """Download a finished render. The result's one part is a
        VideoOutput carrying base64 data, the media type the provider
        served, and the provider's own download URL for as long as the
        provider keeps the file alive."""
        self._require_open()
        self._require_video_job(job)
        if job.status == "failed":
            self._raise_video_failure(job)
        if job.status != "succeeded" or not job.video_url:
            raise ValueError(
                "this job has not succeeded yet; call check_video() until it does"
            )
        plan = self._adapter.video_download_plan(job)
        with _tracing.span(
            "keycall.video_generation.fetch", provider=self.provider, model=job.model
        ) as trace:
            result = self._transport.download(
                plan,
                operation="video_generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._video_result_from_download(job, result)
            trace.event(
                "model",
                operation="video_generation",
                target=job.model,
                duration_ms=result.duration_ms,
                result={"videos": len(invocation.parts)},
            )
            return invocation

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
        poll_interval: float = 10.0,
    ) -> InvocationResult:
        """Start, poll, and download in one call. ``timeout`` is the
        caller's waiting budget in seconds and has no default: render
        times observed live range from 10 seconds to over 11 minutes, so
        only the caller can say how long is too long. When the budget
        runs out the raised VideoJobTimeout carries the still-valid job —
        the render keeps going provider-side and check_video() resumes
        where the wait left off."""
        job = self.start_video(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        deadline = time.monotonic() + timeout
        while True:
            job = self.check_video(job)
            if job.status == "succeeded":
                return self.fetch_video(job)
            if job.status == "failed":
                self._raise_video_failure(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoJobTimeout(
                    f"video render still going after {timeout:g}s; the job remains "
                    "valid — poll it with check_video(error.job)",
                    provider=self.provider,
                    job=job,
                )
            time.sleep(min(poll_interval, remaining))

    def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> InvocationResult:
        return self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    def stream_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> TextStream:
        """Stream a text generation. Use as a context manager; iterate the
        typed events, then call result() for the full InvocationResult."""
        self._require_open()
        return TextStream(
            self,
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def realtime(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        provider_config: Mapping[str, Any] | None = None,
    ) -> RealtimeSession:
        """A live WebSocket session with a realtime model. Use as a
        context manager; push turns with send_text/send_audio and read
        normalized events from events()."""
        self._require_open()
        config = RealtimeConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            provider_config=provider_config,
        )
        path, translator = self._adapter.realtime_plan(config)
        from ._realtime import RealtimeSession

        return RealtimeSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def transcribe_stream(
        self,
        *,
        model: str | None = None,
        sample_rate: int = 16000,
    ) -> TranscriptionSession:
        """A live speech-to-text session (AssemblyAI, Deepgram). Use as a
        context manager; push raw 16-bit mono PCM with send_audio, call
        finish() when the audio ends, and read normalized events from
        events(). model None takes the provider's default streaming
        model."""
        self._require_open()
        config = TranscriptionConfig(model=model, sample_rate=sample_rate)
        path, translator = self._adapter.transcription_plan(config)
        from ._transcription import TranscriptionSession

        return TranscriptionSession(self._transport, path=path, translator=translator)


class AsyncKeyCall(_BaseClient):
    """Asynchronous client. Same identity rules and methods as KeyCall."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        trust_env: bool = True,
        httpx_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        object.__setattr__(
            self,
            "_transport",
            AsyncTransport(
                self._resolved,
                self._require_open(),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_response_bytes=max_response_bytes,
                trust_env=trust_env,
                allow_private_network=allow_private_network,
                httpx_transport=httpx_transport,
            ),
        )

    async def close(self) -> None:
        await self._transport.close()
        object.__setattr__(self, "_credential", None)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def list_models(
        self,
        *,
        categories: set[ModelCategory] | frozenset[ModelCategory] | None = None,
        refresh: bool = False,
    ) -> ModelDiscovery:
        credential = self._require_open()
        requested = _validate_categories(categories)
        fingerprint = credential.fingerprint()

        with _tracing.span(
            "keycall.list_models", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            if not refresh:
                cached = self._cached_discovery(requested, fingerprint, trace)
                if cached is not None:
                    return cached

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            next_spec = spec
            for _ in range(_MAX_LIST_PAGES):
                result = await self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page_models, next_spec = self._parse_page(trace, spec, result)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec
            return self._store_discovery(
                models,
                truncated=next_spec is not None,
                categories=requested,
                fingerprint=fingerprint,
                trace=trace,
            )

    async def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            carried: Usage | None = None
            carried_ms = 0.0
            for _round in range(_SERVER_TOOL_ROUND_BUDGET):
                spec = self._generation_spec(request)
                result = await self._transport.request(
                    spec,
                    operation="text_generation",
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                invocation = self._parse_invocation(request, result, trace)
                follow_up = self._adapter.server_tool_continuation(request, invocation)
                if follow_up is None:
                    if carried is not None:
                        invocation = dataclasses.replace(
                            invocation,
                            usage=_merged_usage(carried, invocation.usage),
                            round_trip_duration_ms=carried_ms
                            + invocation.round_trip_duration_ms,
                        )
                    return invocation
                # See KeyCall.invoke: the echo rounds are one logical call.
                carried = (
                    invocation.usage
                    if carried is None
                    else _merged_usage(carried, invocation.usage)
                )
                carried_ms += invocation.round_trip_duration_ms
                request = follow_up
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self.provider,
                operation="text_generation",
                retryable=True,
            )

    async def generate_image(self, *, model: str, prompt: str) -> InvocationResult:
        """Async twin of KeyCall.generate_image()."""
        self._require_open()
        request = ImageGenerationRequest(model=model, prompt=prompt)
        spec = self._image_spec(request)
        with _tracing.span(
            "keycall.image_generation", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="image_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_image(request, result, trace)

    async def embed(self, *, model: str, inputs: Sequence[str]) -> InvocationResult:
        """Async twin of KeyCall.embed()."""
        self._require_open()
        request = EmbeddingRequest(model=model, inputs=inputs)
        spec = self._embedding_spec(request)
        with _tracing.span("keycall.embedding", provider=self.provider, model=model) as trace:
            result = await self._transport.request(
                spec,
                operation="embedding",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_embedding(request, result, trace)

    async def generate_speech(
        self, *, model: str, text: str, voice: str | None = None
    ) -> InvocationResult:
        """Async twin of KeyCall.generate_speech()."""
        self._require_open()
        request = SpeechGenerationRequest(model=model, text=text, voice=voice)
        spec = self._speech_spec(request)
        with _tracing.span(
            "keycall.speech_generation", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="speech_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_speech(request, result, trace)

    async def start_video(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> VideoJob:
        """Async twin of KeyCall.start_video()."""
        self._require_open()
        request = VideoGenerationRequest(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        spec = self._adapter.build_video_start_spec(request)
        with _tracing.span(
            "keycall.video_generation.start", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="video_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_video_start(result.payload, model=request.model)
            trace.event(
                "model",
                operation="video_generation",
                target=model,
                duration_ms=result.duration_ms,
                result={"job": "started"},
            )
            return job

    async def check_video(self, job: VideoJob) -> VideoJob:
        """Async twin of KeyCall.check_video()."""
        self._require_open()
        self._require_video_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_video_status_spec(job)
        result = await self._transport.request(
            spec,
            operation="video_generation",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_video_status(result.payload, job=job)

    async def fetch_video(self, job: VideoJob) -> InvocationResult:
        """Async twin of KeyCall.fetch_video()."""
        self._require_open()
        self._require_video_job(job)
        if job.status == "failed":
            self._raise_video_failure(job)
        if job.status != "succeeded" or not job.video_url:
            raise ValueError(
                "this job has not succeeded yet; call check_video() until it does"
            )
        plan = self._adapter.video_download_plan(job)
        with _tracing.span(
            "keycall.video_generation.fetch", provider=self.provider, model=job.model
        ) as trace:
            result = await self._transport.download(
                plan,
                operation="video_generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._video_result_from_download(job, result)
            trace.event(
                "model",
                operation="video_generation",
                target=job.model,
                duration_ms=result.duration_ms,
                result={"videos": len(invocation.parts)},
            )
            return invocation

    async def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
        poll_interval: float = 10.0,
    ) -> InvocationResult:
        """Async twin of KeyCall.generate_video()."""
        import anyio

        job = await self.start_video(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        deadline = time.monotonic() + timeout
        while True:
            job = await self.check_video(job)
            if job.status == "succeeded":
                return await self.fetch_video(job)
            if job.status == "failed":
                self._raise_video_failure(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoJobTimeout(
                    f"video render still going after {timeout:g}s; the job remains "
                    "valid — poll it with check_video(error.job)",
                    provider=self.provider,
                    job=job,
                )
            await anyio.sleep(min(poll_interval, remaining))

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> InvocationResult:
        return await self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    def stream_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> AsyncTextStream:
        """Stream a text generation. Use as an async context manager;
        iterate with `async for`, then call result()."""
        self._require_open()
        return AsyncTextStream(
            self,
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def realtime(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        provider_config: Mapping[str, Any] | None = None,
    ) -> AsyncRealtimeSession:
        """A live WebSocket session with a realtime model. Use as an
        async context manager; push turns with send_text/send_audio and
        read normalized events with `async for`."""
        self._require_open()
        config = RealtimeConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            provider_config=provider_config,
        )
        path, translator = self._adapter.realtime_plan(config)
        from ._realtime import AsyncRealtimeSession

        return AsyncRealtimeSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def transcribe_stream(
        self,
        *,
        model: str | None = None,
        sample_rate: int = 16000,
    ) -> AsyncTranscriptionSession:
        """Async twin of KeyCall.transcribe_stream."""
        self._require_open()
        config = TranscriptionConfig(model=model, sample_rate=sample_rate)
        path, translator = self._adapter.transcription_plan(config)
        from ._transcription import AsyncTranscriptionSession

        return AsyncTranscriptionSession(
            self._transport, path=path, translator=translator
        )
