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

import httpx

from . import _cache, _capabilities, _tracing
from ._cache import CachedModels
from ._credential import Credential
from ._enums import ModelCategory, ProviderProtocol
from ._errors import ErrorCode, KeyCallError
from ._registry import ResolvedProvider, catalog_version, resolve_provider
from ._transport import AsyncTransport, Transport
from ._types import (
    InvocationResult,
    Message,
    Model,
    ModelDiscovery,
    StreamEvent,
    TextGenerationRequest,
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
    return ModelDiscovery(
        provider=provider,
        models=_filter_models(cached.models, categories),
        categories=categories,
        fetched_at=cached.fetched_at,
        from_cache=from_cache,
        catalog_version=catalog_version(),
        warnings=cached.warnings,
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
            self._result = _with_schema_warning(invocation, self._request, self._client.provider)
        return self._result


class TextStream(_StreamCore):
    """Iterate typed stream events; call result() after exhaustion for the
    full InvocationResult. The context manager owns the connection: leaving
    the block closes it, even on early break or exception."""

    def __enter__(self) -> Self:
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = self._ctx.__enter__()
        self._assembler.response_headers = headers
        self._events = events
        self._started_at = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._ctx.__exit__(exc_type, exc, tb)

    def __iter__(self) -> Any:
        for event_name, data in self._events:
            yield from self._feed(event_name, data)
            if self._assembler.saw_terminal:
                break
        if not self._assembler.saw_terminal:
            yield from self._assembler.on_close()
        self._check_terminal()

    def result(self) -> InvocationResult:
        return self._build_result()


class AsyncTextStream(_StreamCore):
    """Async twin of TextStream."""

    async def __aenter__(self) -> Self:
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = await self._ctx.__aenter__()
        self._assembler.response_headers = headers
        self._events = events
        self._started_at = time.monotonic()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._ctx.__aexit__(exc_type, exc, tb)

    async def __aiter__(self) -> Any:
        async for event_name, data in self._events:
            for event in self._feed(event_name, data):
                yield event
            if self._assembler.saw_terminal:
                break
        if not self._assembler.saw_terminal:
            for event in self._assembler.on_close():
                yield event
        self._check_terminal()

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
        spec = self._generation_spec(request)
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="text_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_invocation(request, result, trace)

    def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        response_schema: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        return self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                response_schema=response_schema,
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
        response_schema: Mapping[str, Any] | None = None,
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
                response_schema=response_schema,
            ),
        )


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
        spec = self._generation_spec(request)
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="text_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_invocation(request, result, trace)

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
        response_schema: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        return await self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
                response_schema=response_schema,
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
        response_schema: Mapping[str, Any] | None = None,
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
                response_schema=response_schema,
            ),
        )
