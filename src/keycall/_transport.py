"""HTTP transport: the single audited place credentials are revealed.

``_build_headers`` is the only call site of ``Credential.reveal()`` in the
package. Adapters produce pure request specs and parse pure payloads; they
never see the credential. All provider error text is scrubbed here before
it can reach a KeyCallError.

Retry policy is operation-aware: model listing gets a small bounded
retry budget for transient failures; generation is never
retried after possible transmission because no supported provider documents
generation idempotency.

Response bodies are read incrementally against a size cap so a hostile or
broken endpoint can't buffer unbounded data into memory.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

import httpx

from . import _dnsguard
from ._credential import Credential
from ._errors import ErrorCode, KeyCallError
from ._registry import ResolvedProvider
from ._sanitize import safe_request_id, scrub

__all__ = ["AsyncTransport", "DownloadPlan", "RequestSpec", "Transport", "TransportResult"]

RetryPolicy = Literal["list", "generation"]
ErrorTranslator = Callable[[int, Any], tuple[ErrorCode, bool, str]]

_LIST_RETRY_BUDGET = 2
_RETRY_BACKOFF_SECONDS = (0.5, 1.5)
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
# Finished-video downloads carry whole media files, so the ordinary
# response cap would refuse legitimate results; a distinct bound keeps
# downloads finite without loosening every other operation.
DOWNLOAD_MAX_RESPONSE_BYTES = 256 * 1024 * 1024
# One SSE event may not exceed this even when the cumulative cap has room:
# a single unbounded data: line must not buffer arbitrarily.
_SSE_MAX_EVENT_BYTES = 1024 * 1024


class _SSEDecoder:
    """Incremental server-sent-events decoder: feed lines, collect
    (event_name, data) pairs at blank-line boundaries. Comment lines and
    unknown fields are dropped, never interpreted."""

    __slots__ = ("_data", "_data_len", "_event")

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []
        self._data_len = 0

    def feed(self, line: str) -> tuple[str | None, str] | None:
        """Returns a completed (event_name, data) pair, None otherwise.
        Raises ValueError when a single event exceeds the per-event cap."""
        if line == "":
            if not self._data:
                self._event = None
                return None
            pair = (self._event, "\n".join(self._data))
            self._event, self._data, self._data_len = None, [], 0
            return pair
        if line.startswith("event:"):
            self._event = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            value = value.removeprefix(" ")
            self._data_len += len(value)
            if self._data_len > _SSE_MAX_EVENT_BYTES:
                raise ValueError("SSE event exceeded the per-event size limit")
            self._data.append(value)
        return None

    def flush(self) -> tuple[str | None, str] | None:
        """A final event unterminated by a blank line (some providers close
        the connection right after the last data line)."""
        if not self._data:
            return None
        pair = (self._event, "\n".join(self._data))
        self._event, self._data, self._data_len = None, [], 0
        return pair


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestSpec:
    method: str
    path: str
    params: Mapping[str, str] = field(default_factory=dict)
    json_body: Mapping[str, Any] | None = None
    # Per-request headers layered on top of the provider's standard set
    # (auth, api_version_header, Content-Type) — for a header a request
    # needs only conditionally, such as Anthropic's beta feature flags,
    # which must not be sent on every request the way api_version_header is.
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class DownloadPlan:
    """How to fetch a provider-hosted asset from the URL a finished job
    reported. The adapter that parsed the job declares the mechanics it
    live-verified for its provider; the transport enforces them:

    - ``send_credential`` attaches the provider auth header — permitted
      only when the URL's host is the provider's own catalog host, so the
      credential can never be sent to a host outside the provider profile,
      no matter what URL a response names.
    - ``allow_same_origin_redirect`` follows at most one redirect, and
      only to the same scheme and host. Gemini's file download answers a
      302 to another path on its own host and needs the auth header on the
      second hop (verified live 2026-08-13); everything else keeps the
      package-wide refusal.
    - ``allowed_hosts`` pins where the URL may point at all. xAI serves
      finished videos from vidgen.x.ai as unsigned public URLs (verified
      live 2026-08-13), so its adapter pins that host with no credential;
      a response naming any other host is refused before a request."""

    url: str
    allowed_hosts: tuple[str, ...]
    send_credential: bool = False
    allow_same_origin_redirect: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class TransportResult:
    # A dict for every JSON-answering operation, which is most of them.
    # `bytes` for the few whose successful response is not JSON
    # (OpenAI's speech endpoint returns a raw audio file, not a JSON
    # envelope, at 200 — verified live 2026-08-12). Which kind a given
    # result carries is decided per response, from its own Content-Type
    # header, in _classify_response below; an adapter that calls such a
    # route already knows which shape to expect back.
    payload: Any
    headers: Mapping[str, str]
    status_code: int
    duration_ms: float


def _refuse_if_proxy_bypasses_guard(provider: str) -> None:
    """httpx routes proxied requests through its own proxy transports, not
    the guarded default transport, so the DNS-rebinding guard can't see
    them. With a proxy the proxy resolves DNS anyway, but the private-address
    check is also skipped. Every other guard in KeyCall fails closed, so
    this one refuses at construction too (it used to warn and proceed):
    a custom target whose guard the environment would silently disable is
    a configuration error until the caller says which way to resolve it."""
    import os

    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    if any(os.environ.get(name) for name in names):
        raise KeyCallError(
            f"a proxy environment variable is set, which would route requests "
            f"to custom target {provider!r} through the proxy and bypass the "
            "DNS-rebinding/private-address guard. Unset the proxy variable, "
            "pass trust_env=False to ignore it for this client, or pass "
            "allow_private_network=True if routing this target through the "
            "proxy is deliberate",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=provider,
        )


_EMPTY_HEADERS: Mapping[str, str] = {}


def _build_headers(
    resolved: ResolvedProvider,
    credential: Credential,
    *,
    extra: Mapping[str, str] = _EMPTY_HEADERS,
) -> dict[str, str]:
    # The one reveal() site in the package. Keep it that way.
    revealed = credential.reveal()
    if resolved.auth_scheme == "bearer":
        headers = {resolved.auth_header: f"Bearer {revealed}"}
    elif resolved.auth_scheme == "api_key":
        headers = {resolved.auth_header: revealed}
    elif resolved.auth_scheme == "token":
        # Deepgram's convention: "Token <key>", same header, different word.
        headers = {resolved.auth_header: f"Token {revealed}"}
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
    headers.update(extra)
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

    def _realtime_url(self, path: str) -> str:
        # Realtime paths are host-rooted: the base URL's own path prefix
        # (/v1, /v1beta) does not apply to the WebSocket endpoints. A full
        # wss:// path passes through as-is, for the provider whose
        # WebSocket host differs from its REST host (AssemblyAI streams
        # from streaming.assemblyai.com while api.assemblyai.com answers
        # the credential check).
        if path.startswith("wss://"):
            return path
        host = urlsplit(self._resolved.base_url).netloc
        return f"wss://{host}{path}"

    def _realtime_headers(self) -> dict[str, str]:
        headers = _build_headers(self._resolved, self._credential)
        # A WebSocket handshake carries no body.
        headers.pop("Content-Type", None)
        return headers

    def _upgrade_error(self, exc: Exception) -> KeyCallError:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        code = {
            401: ErrorCode.INVALID_API_KEY,
            403: ErrorCode.PERMISSION_DENIED,
            429: ErrorCode.RATE_LIMITED,
        }.get(status or 0, ErrorCode.PROVIDER_UNAVAILABLE)
        return KeyCallError(
            f"the provider refused the realtime connection (HTTP {status})",
            code=code,
            provider=self._resolved.provider,
            operation="realtime",
            retryable=code in (ErrorCode.RATE_LIMITED, ErrorCode.PROVIDER_UNAVAILABLE),
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
        """Turn a fully-read response into a result or a typed error.

        Whether the body is JSON is read from the response's own
        Content-Type, per response — never declared up front by the
        caller. A per-request "this route returns binary" flag would be
        one more fact to keep in sync with what a provider does;
        the header already says it, on every single response, including
        error ones. OpenAI's speech endpoint proves why the distinction
        is needed: success is `audio/mpeg`, but a 404 from the very same
        route is `application/json` (both verified live 2026-08-12) — a
        flag fixed per route would have gotten the error case wrong.

        Conservative on purpose: only a Content-Type that unambiguously
        names a binary kind (audio/image/video/octet-stream/pdf) skips
        JSON parsing. Anything else — a JSON type, a text type, or the
        header missing entirely, which happens on some custom
        OpenAI-compatible targets — keeps today's behavior. A new
        binary-returning route only has to exist; it never has to be
        registered here.
        """
        content_type = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
        is_declared_binary = any(
            content_type.startswith(prefix)
            for prefix in ("audio/", "image/", "video/", "application/octet-stream", "application/pdf")
        )

        if is_declared_binary and status_code < 300:
            if not body:
                return KeyCallError(
                    "provider returned an empty response body",
                    code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                    provider=self._resolved.provider,
                    operation=operation,
                    status_code=status_code,
                )
            return TransportResult(
                payload=bytes(body),
                headers=dict(headers),
                status_code=status_code,
                duration_ms=duration_ms,
            )

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

    def _validate_download(self, plan: DownloadPlan, operation: str) -> None:
        """Refuse a download before any request leaves. The URL comes out
        of a provider response, so it is data, not something to trust: it
        must be https, on a host the adapter pinned from live evidence,
        and the credential may only ever be sent to the provider's own
        catalog host."""
        parts = urlsplit(plan.url)
        host = parts.hostname or ""
        if parts.scheme != "https" or not host:
            raise KeyCallError(
                "provider reported a download URL that is not https",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._resolved.provider,
                operation=operation,
            )
        if host not in plan.allowed_hosts:
            raise KeyCallError(
                f"provider reported a download URL on unexpected host {host!r}; "
                "downloads are pinned to hosts verified for this provider",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._resolved.provider,
                operation=operation,
            )
        if plan.send_credential and host != (urlsplit(self._resolved.base_url).hostname or ""):
            raise KeyCallError(
                "refusing to send the credential to a host other than the provider's own",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._resolved.provider,
                operation=operation,
            )

    def _redirect_target(
        self, *, current_url: str, location: str, operation: str
    ) -> str:
        """Resolve a Location header under the same-origin rule: one hop,
        same scheme and host, or a typed refusal."""
        target = urljoin(current_url, location)
        t, o = urlsplit(target), urlsplit(current_url)
        if t.scheme != "https" or t.hostname != o.hostname:
            raise KeyCallError(
                "provider redirected the download off its own host; KeyCall refuses "
                "to follow cross-origin redirects",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self._resolved.provider,
                operation=operation,
            )
        return target

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
                _refuse_if_proxy_bypasses_guard(resolved.provider)
            httpx_transport = _dnsguard.GuardedTransport(
                httpx.HTTPTransport(trust_env=trust_env), provider=resolved.provider
            )
        self._client = httpx.Client(
            timeout=self._timeout,
            transport=httpx_transport,
            follow_redirects=False,
            trust_env=trust_env,
        )
        self._trust_env = trust_env

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def realtime_connect(self, path: str) -> Iterator[RealtimeWire]:
        from httpx_ws import WebSocketUpgradeError, connect_ws

        headers = self._realtime_headers()
        client = httpx.Client(
            timeout=self._timeout, trust_env=self._trust_env, headers=headers
        )
        ws: Any
        try:
            with connect_ws(self._realtime_url(path), client) as ws:
                yield RealtimeWire(ws, scrub=self._scrub)
        except WebSocketUpgradeError as exc:
            raise self._upgrade_error(exc) from None
        except httpx.HTTPError as exc:
            raise self._network_error(exc, "realtime") from None
        finally:
            client.close()

    def _read_capped(
        self, response: httpx.Response, operation: str, *, cap: int | None = None
    ) -> bytes:
        limit = cap if cap is not None else self._max_response_bytes
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > limit:
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
                    headers=_build_headers(self._resolved, self._credential, extra=spec.headers),
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

    def download(
        self,
        plan: DownloadPlan,
        *,
        operation: str,
        translate_error: ErrorTranslator | None = None,
    ) -> TransportResult:
        """Fetch a provider-hosted asset under the plan's rules. A plain
        GET with the list retry budget — a download is an idempotent read,
        unlike generation. The body cap is the download bound, not the
        ordinary response cap, because the payload is a whole media file."""
        self._validate_download(plan, operation)
        url = plan.url
        hops = 0
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                http_request = self._client.build_request(
                    "GET",
                    url,
                    headers=_build_headers(self._resolved, self._credential)
                    if plan.send_credential
                    else None,
                )
                response = self._client.send(http_request, stream=True)
                try:
                    body = self._read_capped(
                        response, operation, cap=DOWNLOAD_MAX_RESPONSE_BYTES
                    )
                finally:
                    response.close()
            except KeyCallError as exc:
                outcome: TransportResult | KeyCallError = exc
            except httpx.HTTPError as exc:
                outcome = self._network_error(exc, operation)
            else:
                if (
                    300 <= response.status_code < 400
                    and plan.allow_same_origin_redirect
                    and hops == 0
                ):
                    url = self._redirect_target(
                        current_url=url,
                        location=response.headers.get("location") or "",
                        operation=operation,
                    )
                    hops = 1
                    continue
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
            delay = self._should_retry("list", attempt, outcome)
            if delay is None:
                raise outcome
            attempt += 1
            time.sleep(delay)

    @contextmanager
    def stream_request(
        self,
        spec: RequestSpec,
        *,
        operation: str,
        translate_error: ErrorTranslator | None = None,
    ) -> Iterator[tuple[httpx.Headers, Iterator[tuple[str | None, str]]]]:
        """Open a streaming request. Yields (headers, event_iterator) where
        the iterator produces (event_name, data) SSE pairs. Never retries:
        streaming is generation, and generation is never retried. Pre-stream
        HTTP errors classify exactly as non-streaming ones."""
        try:
            http_request = self._client.build_request(
                spec.method,
                self._url(spec.path),
                params=dict(spec.params) or None,
                json=spec.json_body,
                headers=_build_headers(self._resolved, self._credential, extra=spec.headers),
            )
            response = self._client.send(http_request, stream=True)
        except httpx.HTTPError as exc:
            raise self._network_error(exc, operation) from None
        try:
            if response.status_code >= 300:
                body = self._read_capped(response, operation)
                outcome = self._classify_response(
                    status_code=response.status_code,
                    headers=response.headers,
                    body=body,
                    translate=translate_error,
                    operation=operation,
                    duration_ms=0.0,
                )
                assert isinstance(outcome, KeyCallError)
                raise outcome
            yield response.headers, self._iter_sse(response, operation)
        finally:
            response.close()

    def _iter_sse(self, response: httpx.Response, operation: str) -> Iterator[tuple[str | None, str]]:
        decoder = _SSEDecoder()
        total = 0
        try:
            for line in response.iter_lines():
                total += len(line) + 1
                if total > self._max_response_bytes:
                    raise self._size_error(operation)
                try:
                    pair = decoder.feed(line)
                except ValueError:
                    raise self._size_error(operation) from None
                if pair is not None:
                    yield pair
            final = decoder.flush()
            if final is not None:
                yield final
        except httpx.HTTPError as exc:
            raise self._network_error(exc, operation) from None


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
                _refuse_if_proxy_bypasses_guard(resolved.provider)
            httpx_transport = _dnsguard.AsyncGuardedTransport(
                httpx.AsyncHTTPTransport(trust_env=trust_env), provider=resolved.provider
            )
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            transport=httpx_transport,
            follow_redirects=False,
            trust_env=trust_env,
        )
        self._trust_env = trust_env

    async def close(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def realtime_connect(self, path: str) -> AsyncIterator[AsyncRealtimeWire]:
        from httpx_ws import WebSocketUpgradeError, aconnect_ws

        headers = self._realtime_headers()
        client = httpx.AsyncClient(
            timeout=self._timeout, trust_env=self._trust_env, headers=headers
        )
        ws: Any
        try:
            async with aconnect_ws(self._realtime_url(path), client) as ws:
                yield AsyncRealtimeWire(ws, scrub=self._scrub)
        except WebSocketUpgradeError as exc:
            raise self._upgrade_error(exc) from None
        except httpx.HTTPError as exc:
            raise self._network_error(exc, "realtime") from None
        finally:
            await client.aclose()

    async def _read_capped(
        self, response: httpx.Response, operation: str, *, cap: int | None = None
    ) -> bytes:
        limit = cap if cap is not None else self._max_response_bytes
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > limit:
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
                    headers=_build_headers(self._resolved, self._credential, extra=spec.headers),
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

    async def download(
        self,
        plan: DownloadPlan,
        *,
        operation: str,
        translate_error: ErrorTranslator | None = None,
    ) -> TransportResult:
        """Async twin of Transport.download; same rules, same caps."""
        import anyio

        self._validate_download(plan, operation)
        url = plan.url
        hops = 0
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                http_request = self._client.build_request(
                    "GET",
                    url,
                    headers=_build_headers(self._resolved, self._credential)
                    if plan.send_credential
                    else None,
                )
                response = await self._client.send(http_request, stream=True)
                try:
                    body = await self._read_capped(
                        response, operation, cap=DOWNLOAD_MAX_RESPONSE_BYTES
                    )
                finally:
                    await response.aclose()
            except KeyCallError as exc:
                outcome: TransportResult | KeyCallError = exc
            except httpx.HTTPError as exc:
                outcome = self._network_error(exc, operation)
            else:
                if (
                    300 <= response.status_code < 400
                    and plan.allow_same_origin_redirect
                    and hops == 0
                ):
                    url = self._redirect_target(
                        current_url=url,
                        location=response.headers.get("location") or "",
                        operation=operation,
                    )
                    hops = 1
                    continue
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
            delay = self._should_retry("list", attempt, outcome)
            if delay is None:
                raise outcome
            attempt += 1
            await anyio.sleep(delay)

    @asynccontextmanager
    async def stream_request(
        self,
        spec: RequestSpec,
        *,
        operation: str,
        translate_error: ErrorTranslator | None = None,
    ) -> AsyncIterator[tuple[httpx.Headers, AsyncIterator[tuple[str | None, str]]]]:
        try:
            http_request = self._client.build_request(
                spec.method,
                self._url(spec.path),
                params=dict(spec.params) or None,
                json=spec.json_body,
                headers=_build_headers(self._resolved, self._credential, extra=spec.headers),
            )
            response = await self._client.send(http_request, stream=True)
        except httpx.HTTPError as exc:
            raise self._network_error(exc, operation) from None
        try:
            if response.status_code >= 300:
                body = await self._read_capped(response, operation)
                outcome = self._classify_response(
                    status_code=response.status_code,
                    headers=response.headers,
                    body=body,
                    translate=translate_error,
                    operation=operation,
                    duration_ms=0.0,
                )
                assert isinstance(outcome, KeyCallError)
                raise outcome
            yield response.headers, self._aiter_sse(response, operation)
        finally:
            await response.aclose()

    async def _aiter_sse(
        self, response: httpx.Response, operation: str
    ) -> AsyncIterator[tuple[str | None, str]]:
        decoder = _SSEDecoder()
        total = 0
        try:
            async for line in response.aiter_lines():
                total += len(line) + 1
                if total > self._max_response_bytes:
                    raise self._size_error(operation)
                try:
                    pair = decoder.feed(line)
                except ValueError:
                    raise self._size_error(operation) from None
                if pair is not None:
                    yield pair
            final = decoder.flush()
            if final is not None:
                yield final
        except httpx.HTTPError as exc:
            raise self._network_error(exc, operation) from None


class RealtimeWire:
    """One live realtime WebSocket, wrapped so nothing above the
    transport touches httpx-ws types or the raw close reason. ``receive``
    answers a frame payload, or None once the peer closed (with the
    scrubbed close reason kept on ``close_reason``)."""

    def __init__(self, ws: Any, *, scrub: Callable[[str], str]) -> None:
        self._ws = ws
        self._scrub = scrub
        self.close_reason: str | None = None

    def send(self, message: str) -> None:
        self._ws.send_text(message)

    def send_bytes(self, data: bytes) -> None:
        """A binary frame — how STT providers take raw audio."""
        self._ws.send_bytes(data)

    def receive(self, timeout: float | None = None) -> str | bytes | None:
        from httpx_ws import WebSocketDisconnect

        try:
            event = self._ws.receive(timeout=timeout)
        except WebSocketDisconnect as exc:
            reason = self._scrub(str(exc.reason or ""))[:300]
            self.close_reason = f"{exc.code}: {reason}" if reason else str(exc.code)
            return None
        except TimeoutError:
            raise KeyCallError(
                "no realtime frame arrived within the timeout",
                code=ErrorCode.TIMEOUT,
                operation="realtime",
                retryable=True,
            ) from None
        data = getattr(event, "data", None)
        if not isinstance(data, (str, bytes)):
            raise KeyCallError(
                "realtime frame carried no payload",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                operation="realtime",
            )
        return data


class AsyncRealtimeWire:
    """Async twin of RealtimeWire."""

    def __init__(self, ws: Any, *, scrub: Callable[[str], str]) -> None:
        self._ws = ws
        self._scrub = scrub
        self.close_reason: str | None = None

    async def send(self, message: str) -> None:
        await self._ws.send_text(message)

    async def send_bytes(self, data: bytes) -> None:
        """A binary frame — how STT providers take raw audio."""
        await self._ws.send_bytes(data)

    async def receive(self, timeout: float | None = None) -> str | bytes | None:
        from httpx_ws import WebSocketDisconnect

        try:
            event = await self._ws.receive(timeout=timeout)
        except WebSocketDisconnect as exc:
            reason = self._scrub(str(exc.reason or ""))[:300]
            self.close_reason = f"{exc.code}: {reason}" if reason else str(exc.code)
            return None
        except TimeoutError:
            raise KeyCallError(
                "no realtime frame arrived within the timeout",
                code=ErrorCode.TIMEOUT,
                operation="realtime",
                retryable=True,
            ) from None
        data = getattr(event, "data", None)
        if not isinstance(data, (str, bytes)):
            raise KeyCallError(
                "realtime frame carried no payload",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                operation="realtime",
            )
        return data
