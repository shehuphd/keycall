"""HTTP transport: the single audited place credentials are revealed.

``_build_headers`` is the only call site of ``Credential.reveal()`` in the
package. Adapters produce pure request specs and parse pure payloads; they
never see the credential. All provider error text is scrubbed here before
it can reach a KeyCallError (PRD sections 10.1, 10.2).

Retry policy is operation-aware (PRD section 12): model listing gets a
small bounded retry budget for transient failures; generation is never
retried after possible transmission because no supported provider documents
generation idempotency.

Response bodies are read incrementally against a size cap so a hostile or
broken endpoint cannot buffer unbounded data into memory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from . import _dnsguard
from ._credential import Credential
from ._errors import ErrorCode, KeyCallError
from ._registry import ResolvedProvider
from ._sanitize import safe_request_id, scrub

__all__ = ["AsyncTransport", "RequestSpec", "Transport", "TransportResult"]

RetryPolicy = Literal["list", "generation"]
ErrorTranslator = Callable[[int, Any], tuple[ErrorCode, bool, str]]

_LIST_RETRY_BUDGET = 2
_RETRY_BACKOFF_SECONDS = (0.5, 1.5)
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestSpec:
    method: str
    path: str
    params: Mapping[str, str] = field(default_factory=dict)
    json_body: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportResult:
    payload: Any
    headers: Mapping[str, str]
    status_code: int
    duration_ms: float


def _warn_if_proxy_bypasses_guard(provider: str) -> None:
    """httpx routes proxied requests through its own proxy transports, not
    the guarded default transport, so the DNS-rebinding guard cannot see
    them. With a proxy the proxy resolves DNS anyway, but the private-address
    check is also skipped — surface that instead of staying silent."""
    import os

    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    if any(os.environ.get(name) for name in names):
        import warnings

        warnings.warn(
            f"keycall: a proxy environment variable is set; requests to custom "
            f"target {provider!r} will route through the proxy and bypass the "
            "DNS-rebinding/private-address guard. Unset the proxy or pass "
            "trust_env=False if this is not intended",
            RuntimeWarning,
            stacklevel=4,
        )


def _build_headers(resolved: ResolvedProvider, credential: Credential) -> dict[str, str]:
    # The one reveal() site in the package. Keep it that way.
    revealed = credential.reveal()
    if resolved.auth_scheme == "bearer":
        headers = {resolved.auth_header: f"Bearer {revealed}"}
    elif resolved.auth_scheme == "api_key":
        headers = {resolved.auth_header: revealed}
    else:
        raise KeyCallError(
            f"unknown auth scheme {resolved.auth_scheme!r} in provider profile",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
            provider=resolved.provider,
        )
    if resolved.api_version_header is not None:
        name, value = resolved.api_version_header
        headers[name] = value
    headers["Content-Type"] = "application/json"
    return headers


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    # RFC 9110 also permits an HTTP-date.
    from email.utils import parsedate_to_datetime

    try:
        target = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    from datetime import datetime, timezone

    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


class _TransportCore:
    """Shared pure logic; subclasses provide the sync/async I/O."""

    def __init__(
        self,
        resolved: ResolvedProvider,
        credential: Credential,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._resolved = resolved
        self._credential = credential
        self._max_response_bytes = max_response_bytes
        self._timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout
        )

    def _url(self, path: str) -> str:
        return self._resolved.base_url.rstrip("/") + path

    def _scrub(self, text: str) -> str:
        return scrub(text, credential_value=self._credential.reveal())

    def _request_id(self, headers: Mapping[str, str]) -> str | None:
        header = self._resolved.provider_request_id_header
        return safe_request_id(headers.get(header)) if header else None

    def _size_error(self, operation: str) -> KeyCallError:
        return KeyCallError(
            f"provider response exceeded the {self._max_response_bytes} byte limit",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=self._resolved.provider,
            operation=operation,
        )

    def _network_error(self, exc: Exception, operation: str) -> KeyCallError:
        if isinstance(exc, httpx.TimeoutException):
            code, message = ErrorCode.TIMEOUT, "the provider did not respond within the timeout"
        else:
            code, message = ErrorCode.NETWORK_ERROR, "could not reach the provider"
        return KeyCallError(
            f"{message}: {self._scrub(str(exc))}",
            code=code,
            provider=self._resolved.provider,
            operation=operation,
            retryable=True,
        )

    def _classify_response(
        self,
        *,
        status_code: int,
        headers: Mapping[str, str],
        body: bytes,
        translate: ErrorTranslator | None,
        operation: str,
        duration_ms: float,
    ) -> TransportResult | KeyCallError:
        """Turn a fully-read response into a result or a typed error."""
        try:
            payload = json.loads(body) if body else None
        except ValueError:
            payload = None

        if status_code < 300:
            if payload is None:
                return KeyCallError(
                    "provider returned a non-JSON response body",
                    code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                    provider=self._resolved.provider,
                    operation=operation,
                    status_code=status_code,
                )
            return TransportResult(
                payload=payload,
                headers=dict(headers),
                status_code=status_code,
                duration_ms=duration_ms,
            )

        if status_code < 400:
            # Redirects are never followed: a cross-origin redirect must not
            # carry the credential (registry research section 9).
            return KeyCallError(
                "provider attempted a redirect; KeyCall refuses to follow redirects "
                "while carrying a credential",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._resolved.provider,
                operation=operation,
                status_code=status_code,
            )

        if translate is not None:
            code, retryable, raw_message = translate(status_code, payload)
        else:
            code, retryable, raw_message = (
                ErrorCode.INVALID_PROVIDER_RESPONSE,
                False,
                f"provider returned unexpected status {status_code}",
            )
        return KeyCallError(
            self._scrub(raw_message),
            code=code,
            provider=self._resolved.provider,
            operation=operation,
            retryable=retryable,
            status_code=status_code,
            provider_request_id=self._request_id(headers),
            retry_after=_parse_retry_after(headers),
        )

    def _should_retry(
        self, policy: RetryPolicy, attempt: int, error: KeyCallError
    ) -> float | None:
        """Return a backoff delay to retry, or None to raise."""
        if policy != "list" or attempt >= _LIST_RETRY_BUDGET:
            return None
        if not error.retryable:
            return None
        delay = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
        if error.retry_after is not None:
            delay = max(delay, min(error.retry_after, 30.0))
        return delay


class Transport(_TransportCore):
    def __init__(
        self,
        resolved: ResolvedProvider,
        credential: Credential,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        trust_env: bool = True,
        allow_private_network: bool = False,
        httpx_transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            resolved,
            credential,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_response_bytes=max_response_bytes,
        )
        if httpx_transport is None and resolved.is_custom and not allow_private_network:
            # Custom targets are user-supplied: pin DNS to defeat rebinding.
            if trust_env:
                _warn_if_proxy_bypasses_guard(resolved.provider)
            httpx_transport = _dnsguard.GuardedTransport(
                httpx.HTTPTransport(trust_env=trust_env), provider=resolved.provider
            )
        self._client = httpx.Client(
            timeout=self._timeout,
            transport=httpx_transport,
            follow_redirects=False,
            trust_env=trust_env,
        )

    def close(self) -> None:
        self._client.close()

    def _read_capped(self, response: httpx.Response, operation: str) -> bytes:
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > self._max_response_bytes:
                raise self._size_error(operation)
        return bytes(body)

    def request(
        self,
        spec: RequestSpec,
        *,
        operation: str,
        retry_policy: RetryPolicy,
        translate_error: ErrorTranslator | None = None,
    ) -> TransportResult:
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                http_request = self._client.build_request(
                    spec.method,
                    self._url(spec.path),
                    params=dict(spec.params) or None,
                    json=spec.json_body,
                    headers=_build_headers(self._resolved, self._credential),
                )
                response = self._client.send(http_request, stream=True)
                try:
                    body = self._read_capped(response, operation)
                finally:
                    response.close()
            except KeyCallError as exc:
                # The DNS guard raises typed errors from inside send();
                # routing them through the outcome path keeps a transient
                # resolution failure eligible for the list retry budget.
                outcome: TransportResult | KeyCallError = exc
            except httpx.HTTPError as exc:
                outcome = self._network_error(exc, operation)
            else:
                outcome = self._classify_response(
                    status_code=response.status_code,
                    headers=response.headers,
                    body=body,
                    translate=translate_error,
                    operation=operation,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )

            if isinstance(outcome, TransportResult):
                return outcome
            delay = self._should_retry(retry_policy, attempt, outcome)
            if delay is None:
                raise outcome
            attempt += 1
            time.sleep(delay)


class AsyncTransport(_TransportCore):
    def __init__(
        self,
        resolved: ResolvedProvider,
        credential: Credential,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        trust_env: bool = True,
        allow_private_network: bool = False,
        httpx_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            resolved,
            credential,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_response_bytes=max_response_bytes,
        )
        if httpx_transport is None and resolved.is_custom and not allow_private_network:
            if trust_env:
                _warn_if_proxy_bypasses_guard(resolved.provider)
            httpx_transport = _dnsguard.AsyncGuardedTransport(
                httpx.AsyncHTTPTransport(trust_env=trust_env), provider=resolved.provider
            )
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            transport=httpx_transport,
            follow_redirects=False,
            trust_env=trust_env,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _read_capped(self, response: httpx.Response, operation: str) -> bytes:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self._max_response_bytes:
                raise self._size_error(operation)
        return bytes(body)

    async def request(
        self,
        spec: RequestSpec,
        *,
        operation: str,
        retry_policy: RetryPolicy,
        translate_error: ErrorTranslator | None = None,
    ) -> TransportResult:
        import anyio

        attempt = 0
        while True:
            started = time.monotonic()
            try:
                http_request = self._client.build_request(
                    spec.method,
                    self._url(spec.path),
                    params=dict(spec.params) or None,
                    json=spec.json_body,
                    headers=_build_headers(self._resolved, self._credential),
                )
                response = await self._client.send(http_request, stream=True)
                try:
                    body = await self._read_capped(response, operation)
                finally:
                    await response.aclose()
            except KeyCallError as exc:
                outcome: TransportResult | KeyCallError = exc
            except httpx.HTTPError as exc:
                outcome = self._network_error(exc, operation)
            else:
                outcome = self._classify_response(
                    status_code=response.status_code,
                    headers=response.headers,
                    body=body,
                    translate=translate_error,
                    operation=operation,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )

            if isinstance(outcome, TransportResult):
                return outcome
            delay = self._should_retry(retry_policy, attempt, outcome)
            if delay is None:
                raise outcome
            attempt += 1
            await anyio.sleep(delay)
