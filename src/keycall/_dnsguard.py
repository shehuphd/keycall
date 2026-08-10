"""DNS-rebinding defense for custom endpoints.

Validating a hostname's address at config time is not enough: the attacker
answers the first lookup with a public address and the second (the one httpx
actually connects with) with an internal one. Closing that race means
connecting to the *same* address that was validated.

This transport wrapper resolves the hostname once per request, rejects the
request if any resolved address is private/loopback/link-local/reserved,
then rewrites the URL to the validated address while preserving the original
hostname for TLS SNI and the Host header — so certificate verification still
runs against the original hostname, and the connection cannot be re-pointed
between check and connect.

Only wraps custom targets. Named providers route to hostnames from the
KeyCall-maintained catalog and are not user-supplied.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import httpx

from ._errors import ErrorCode, KeyCallError

__all__ = ["AsyncGuardedTransport", "GuardedTransport", "validate_and_pin"]


def _blocked(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_and_pin(url: httpx.URL, *, provider: str) -> tuple[httpx.URL, str] | None:
    """Resolve and validate a URL's host.

    Returns (pinned_url, original_host) when a rewrite is needed, or None
    when the host is already a literal IP (the registry validated it at
    construction, and there is no resolution step to race).
    """
    host = url.host
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return None

    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise KeyCallError(
            f"could not resolve {host}",
            code=ErrorCode.NETWORK_ERROR,
            provider=provider,
            retryable=True,
        ) from exc

    addresses = [info[4][0] for info in infos]
    if not addresses:
        raise KeyCallError(
            f"{host} resolved to no addresses",
            code=ErrorCode.NETWORK_ERROR,
            provider=provider,
            retryable=True,
        )
    # Every resolved address must be public: one internal answer in the set
    # is enough for a rebinding attack to land.
    for address in addresses:
        if _blocked(address):
            raise KeyCallError(
                f"{host} resolves to a private/internal address; refusing to send "
                "a credential there (pass allow_private_network=True if deliberate)",
                code=ErrorCode.UNSUPPORTED_PROVIDER,
                provider=provider,
            )
    return url.copy_with(host=addresses[0]), host


def _pin_request(request: httpx.Request, provider: str) -> httpx.Request:
    pinned = validate_and_pin(request.url, provider=provider)
    if pinned is None:
        return request
    url, original_host = pinned
    request.url = url
    # Preserve the original hostname for TLS verification and routing.
    request.headers["Host"] = original_host
    request.extensions = dict(request.extensions)
    request.extensions["sni_hostname"] = original_host
    return request


class GuardedTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport, *, provider: str) -> None:
        self._inner = inner
        self._provider = provider

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._inner.handle_request(_pin_request(request, self._provider))

    def close(self) -> None:
        self._inner.close()


class AsyncGuardedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, *, provider: str) -> None:
        self._inner = inner
        self._provider = provider

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(_pin_request(request, self._provider))

    async def aclose(self) -> None:
        await self._inner.aclose()


def wrap(
    inner: Any, *, provider: str, is_async: bool
) -> Any:
    return (
        AsyncGuardedTransport(inner, provider=provider)
        if is_async
        else GuardedTransport(inner, provider=provider)
    )
