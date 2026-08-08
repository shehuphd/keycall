"""Provider registry: resolves a provider name (or explicit custom target)
to a trusted endpoint profile.

Credential-routing data comes only from the bundled, KeyCall-maintained
catalog: third-party data must never choose where a credential is sent.
Resolution is a registry lookup followed by an explicit custom-adapter
path — never a chain of if/else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from ._enums import ProviderProtocol
from ._errors import ErrorCode, KeyCallError

__all__ = ["ResolvedProvider", "resolve_provider"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedProvider:
    provider: str
    protocol: ProviderProtocol
    base_url: str
    auth_scheme: str
    auth_header: str
    operations: dict[str, dict[str, str]]
    api_version_header: tuple[str, str] | None = None
    provider_request_id_header: str | None = None
    is_custom: bool = False
    # Providers whose model list is not API-discoverable supply it here.
    catalog_models: tuple[dict[str, Any], ...] = ()
    min_max_output_tokens: int | None = None


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Any]:
    raw = resources.files("keycall._catalog").joinpath("catalog.json").read_text("utf-8")
    catalog: dict[str, Any] = json.loads(raw)
    return catalog


def catalog_version() -> str:
    return str(_load_catalog()["catalog_version"])


def _canonical_name(name: str) -> str | None:
    providers = _load_catalog()["providers"]
    lowered = name.strip().lower()
    if lowered in providers:
        return lowered
    for canonical, profile in providers.items():
        if lowered in profile.get("aliases", []):
            return str(canonical)
    return None


def _validate_custom_base_url(
    base_url: str, *, allow_insecure_localhost: bool, allow_private_network: bool
) -> str:
    parts = urlsplit(base_url)
    if parts.query or parts.fragment:
        raise KeyCallError(
            "base_url must not contain a query string or fragment",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
        )
    if parts.username or parts.password:
        raise KeyCallError(
            "base_url must not contain userinfo",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
        )
    if not parts.hostname:
        raise KeyCallError(
            "base_url must be an absolute URL with a hostname",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
        )

    # SSRF guard: a literal private/loopback/link-local/reserved IP address
    # needs an explicit opt-in. This catches direct-IP targeting; a public
    # hostname that *resolves* to a private address (DNS rebinding) is not
    # detectable here and remains consuming-application policy.
    import ipaddress

    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
    ):
        # allow_insecure_localhost already covers deliberate loopback targets.
        loopback_ok = address.is_loopback and allow_insecure_localhost
        if not allow_private_network and not loopback_ok:
            raise KeyCallError(
                f"base_url targets a private/internal address ({parts.hostname}); "
                "pass allow_private_network=True if this is deliberate",
                code=ErrorCode.UNSUPPORTED_PROVIDER,
            )

    if parts.scheme == "https":
        return base_url.rstrip("/")
    if parts.scheme == "http":
        is_local = parts.hostname in ("localhost", "127.0.0.1", "::1")
        if is_local and allow_insecure_localhost:
            return base_url.rstrip("/")
        raise KeyCallError(
            "base_url must use HTTPS (plain HTTP is allowed only for localhost "
            "with allow_insecure_localhost=True)",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
        )
    raise KeyCallError(
        f"base_url has unsupported scheme {parts.scheme!r}; HTTPS is required",
        code=ErrorCode.UNSUPPORTED_PROVIDER,
    )


def resolve_provider(
    provider: str,
    *,
    protocol: ProviderProtocol | str | None = None,
    base_url: str | None = None,
    allow_insecure_localhost: bool = False,
    allow_private_network: bool = False,
) -> ResolvedProvider:
    """Resolve a known provider name, or an explicit custom OpenAI-compatible
    target carrying its own base_url. Unknown name without both explicit
    protocol and base_url raises ``unsupported_provider``.
    """
    if not provider or not provider.strip():
        raise KeyCallError("provider must be a non-empty name", code=ErrorCode.UNSUPPORTED_PROVIDER)

    requested_protocol: ProviderProtocol | None = None
    if protocol is not None:
        try:
            requested_protocol = ProviderProtocol(protocol)
        except ValueError:
            supported = ", ".join(p.value for p in ProviderProtocol)
            raise KeyCallError(
                f"unknown protocol {protocol!r}; supported: {supported}",
                code=ErrorCode.UNSUPPORTED_PROVIDER,
            ) from None

    canonical = _canonical_name(provider)

    if canonical is not None:
        if base_url is not None:
            raise KeyCallError(
                f"provider {canonical!r} uses its maintained endpoint; "
                "base_url is only for custom openai-compatible targets",
                code=ErrorCode.UNSUPPORTED_PROVIDER,
                provider=canonical,
            )
        profile = _load_catalog()["providers"][canonical]
        catalog_protocol = ProviderProtocol(profile["protocol"])
        if requested_protocol is not None and requested_protocol is not catalog_protocol:
            raise KeyCallError(
                f"provider {canonical!r} speaks {catalog_protocol.value!r}, "
                f"not {requested_protocol.value!r}",
                code=ErrorCode.UNSUPPORTED_PROVIDER,
                provider=canonical,
            )
        version_header = profile.get("api_version_header")
        return ResolvedProvider(
            provider=canonical,
            protocol=catalog_protocol,
            base_url=profile["base_url"],
            auth_scheme=profile["auth"]["scheme"],
            auth_header=profile["auth"]["header"],
            operations=profile["operations"],
            api_version_header=(
                (version_header["name"], version_header["value"]) if version_header else None
            ),
            provider_request_id_header=profile.get("provider_request_id_header"),
            catalog_models=tuple(profile.get("models", ())),
            min_max_output_tokens=profile.get("min_max_output_tokens"),
        )

    # Unknown name: only valid as an explicit custom openai-compatible target.
    if requested_protocol is not ProviderProtocol.OPENAI_COMPATIBLE or base_url is None:
        raise KeyCallError(
            f"unknown provider {provider!r}. For a custom endpoint, pass "
            "protocol='openai-compatible' and an explicit base_url",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
            provider=provider,
        )

    validated = _validate_custom_base_url(
        base_url,
        allow_insecure_localhost=allow_insecure_localhost,
        allow_private_network=allow_private_network,
    )
    return ResolvedProvider(
        provider=provider.strip(),
        protocol=ProviderProtocol.OPENAI_COMPATIBLE,
        base_url=validated,
        auth_scheme="bearer",
        auth_header="Authorization",
        operations={
            "list_models": {"method": "GET", "path": "/models"},
            "text_generation": {"method": "POST", "path": "/chat/completions"},
        },
        is_custom=True,
    )
