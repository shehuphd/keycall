"""In-memory registry mapping loaded targets to live KeyCall clients.

Targets are addressed by a stable integer id assigned at load time, never
by name (names aren't guaranteed unique) and never by key. This is the
boundary that keeps the raw credential server-side: everything the browser
sends or receives references a target by id.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .._cache import DEFAULT_TTL_SECONDS
from .._client import KeyCall
from .._sanitize import safe_display_name
from .._sources import Target
from .._types import ModelDiscovery

__all__ = ["Registry", "TargetView"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetView:
    """What the browser is allowed to know about a target. No key, ever."""

    id: int
    name: str
    provider: str
    protocol: str | None
    base_url: str | None
    # Which attachment kinds this provider takes, as {"image": {"bytes":
    # true, "url": true}, ...}. The Playground uses it to disable a control
    # the selected key can never satisfy, rather than letting the user pick
    # a file and discover the refusal after a round trip.
    accepts: dict[str, dict[str, bool]]


def _accepts(capabilities: Any) -> dict[str, dict[str, bool]]:
    return {
        kind: {
            "bytes": getattr(capabilities, f"{kind}_input_bytes"),
            "url": getattr(capabilities, f"{kind}_input_url"),
        }
        for kind in ("image", "audio", "file")
    }


@dataclass(slots=True, kw_only=True)
class _Entry:
    target: Target
    client: KeyCall
    discovery: ModelDiscovery | None = None
    # Monotonic time the cached discovery was stored; expires after the same
    # TTL the library's availability cache uses, so the viewer never shows
    # older entitlement data than the library itself would.
    discovery_at: float = 0.0
    fetch_lock: threading.Lock = field(default_factory=threading.Lock)


class Registry:
    """Owns every loaded credential for the life of the viewer process."""

    def __init__(self, targets: list[Target], *, httpx_transport=None) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, _Entry] = {}
        self._next_id = 0
        self._httpx_transport = httpx_transport
        self.add_targets(targets)

    def add_targets(self, targets: list[Target]) -> None:
        """Open a client per target and register it. Raises KeyCallError if
        a target fails resolution (unknown provider etc.); nothing is
        partially added in that case because all clients are constructed
        before any registration, and every already-opened client is closed
        on a later target's failure."""
        clients: list[tuple[Target, KeyCall]] = []
        try:
            for target in targets:
                clients.append(
                    (
                        target,
                        KeyCall(
                            provider=target.provider,
                            api_key=target.key,
                            protocol=target.protocol,
                            base_url=target.base_url,
                            httpx_transport=self._httpx_transport,
                        ),
                    )
                )
        except BaseException:
            for _, client in clients:
                client.close()
            raise
        with self._lock:
            for target, client in clients:
                self._entries[self._next_id] = _Entry(target=target, client=client)
                self._next_id += 1

    def views(self) -> list[TargetView]:
        with self._lock:
            entries = list(self._entries.items())
        return [
            TargetView(
                id=entry_id,
                name=safe_display_name(entry.target.display_name),
                provider=entry.client.provider,
                protocol=entry.client.protocol.value,
                base_url=entry.target.base_url,
                # Reaching past the client's public surface on purpose: the
                # viewer ships with the library and these are the same
                # frozen catalog facts the adapters gate on, so reading
                # them here can't drift from what a call would enforce.
                accepts=_accepts(entry.client._resolved.capabilities),
            )
            for entry_id, entry in entries
        ]

    def client(self, target_id: int) -> KeyCall:
        with self._lock:
            entry = self._entries.get(target_id)
        if entry is None:
            raise KeyError(target_id)
        return entry.client

    def target(self, target_id: int) -> Target:
        """Server-side only: the original Target (carries the key). Never
        serialize this — the browser gets TargetView, nothing else."""
        with self._lock:
            entry = self._entries.get(target_id)
        if entry is None:
            raise KeyError(target_id)
        return entry.target

    def fetch_lock(self, target_id: int) -> threading.Lock:
        with self._lock:
            entry = self._entries.get(target_id)
        if entry is None:
            raise KeyError(target_id)
        return entry.fetch_lock

    def cached_discovery(self, target_id: int) -> ModelDiscovery | None:
        with self._lock:
            entry = self._entries.get(target_id)
            if entry is None or entry.discovery is None:
                return None
            if time.monotonic() - entry.discovery_at >= DEFAULT_TTL_SECONDS:
                entry.discovery = None
                return None
            return entry.discovery

    def set_cached_discovery(self, target_id: int, discovery: ModelDiscovery) -> None:
        with self._lock:
            entry = self._entries.get(target_id)
            if entry is not None:
                entry.discovery = discovery
                entry.discovery_at = time.monotonic()

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            entry.client.close()
