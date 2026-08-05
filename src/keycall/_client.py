"""KeyCall and AsyncKeyCall clients.

Provider, credential, protocol, and base URL are immutable client identity,
bound once at construction with no setters and no per-call override
(naming-final.md section 2). The raw credential is wrapped in the redacting
Credential type here, at its single entry boundary; only the transport
layer ever reveals it again.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from typing_extensions import Self

import httpx

from . import _cache, _tracing
from ._cache import CachedModels
from ._credential import Credential
from ._enums import ModelCategory, ProviderProtocol
from ._errors import ErrorCode, KeyCallError
from ._registry import ResolvedProvider, catalog_version, resolve_provider
from ._transport import AsyncTransport, Transport
from ._types import InvocationResult, Message, Model, ModelDiscovery, TextGenerationRequest
from .adapters import ProviderAdapter, adapter_for

__all__ = ["AsyncKeyCall", "KeyCall"]

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
        # the raw string (PRD 10.1).
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
                cached = _cache.shared_cache.get(self.provider, self.base_url, fingerprint)
                if cached is not None:
                    trace.event("app", operation="cache_hit", status="ok")
                    return _build_discovery(
                        provider=self.provider, cached=cached, categories=requested, from_cache=True
                    )

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            for _ in range(_MAX_LIST_PAGES):
                result = self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                trace.event(
                    "http",
                    operation=f"{spec.method} {spec.path}",
                    target=self.provider,
                    status=str(result.status_code),
                    duration_ms=result.duration_ms,
                )
                page_models, next_spec = self._adapter.parse_model_page(result.payload)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec

            cached = CachedModels(models=tuple(models), fetched_at=datetime.now(timezone.utc))
            _cache.shared_cache.put(self.provider, self.base_url, fingerprint, cached)
            discovery = _build_discovery(
                provider=self.provider, cached=cached, categories=requested, from_cache=False
            )
            trace.event(
                "model",
                operation="normalize",
                status="ok",
                result={"models": len(models), "filtered": len(discovery.models)},
            )
            return discovery

    def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        if not isinstance(request, TextGenerationRequest):
            raise KeyCallError(
                f"invoke() accepts typed request objects, got {type(request).__name__}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            spec = self._adapter.build_generation_spec(request)
            result = self._transport.request(
                spec,
                operation="text_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._adapter.parse_generation_response(
                result.payload,
                headers=result.headers,
                round_trip_duration_ms=result.duration_ms,
                model=request.model,
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

    def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
    ) -> InvocationResult:
        return self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
            )
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
                cached = _cache.shared_cache.get(self.provider, self.base_url, fingerprint)
                if cached is not None:
                    trace.event("app", operation="cache_hit", status="ok")
                    return _build_discovery(
                        provider=self.provider, cached=cached, categories=requested, from_cache=True
                    )

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            for _ in range(_MAX_LIST_PAGES):
                result = await self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                trace.event(
                    "http",
                    operation=f"{spec.method} {spec.path}",
                    target=self.provider,
                    status=str(result.status_code),
                    duration_ms=result.duration_ms,
                )
                page_models, next_spec = self._adapter.parse_model_page(result.payload)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec

            cached = CachedModels(models=tuple(models), fetched_at=datetime.now(timezone.utc))
            _cache.shared_cache.put(self.provider, self.base_url, fingerprint, cached)
            discovery = _build_discovery(
                provider=self.provider, cached=cached, categories=requested, from_cache=False
            )
            trace.event(
                "model",
                operation="normalize",
                status="ok",
                result={"models": len(models), "filtered": len(discovery.models)},
            )
            return discovery

    async def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        if not isinstance(request, TextGenerationRequest):
            raise KeyCallError(
                f"invoke() accepts typed request objects, got {type(request).__name__}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            spec = self._adapter.build_generation_spec(request)
            result = await self._transport.request(
                spec,
                operation="text_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._adapter.parse_generation_response(
                result.payload,
                headers=result.headers,
                round_trip_duration_ms=result.duration_ms,
                model=request.model,
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

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        web_search: bool = False,
    ) -> InvocationResult:
        return await self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                web_search=web_search,
            )
        )
