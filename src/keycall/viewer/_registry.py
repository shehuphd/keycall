"""In-memory registry mapping loaded targets to live KeyCall clients.

Targets are addressed by a stable integer id assigned at load time, never
by name (names aren't guaranteed unique) and never by key. This is the
boundary that keeps the raw credential server-side: everything the browser
sends or receives references a target by id.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

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


@dataclass(slots=True, kw_only=True)
class _Entry:
    target: Target
    client: KeyCall
    discovery: ModelDiscovery | None = None
    fetch_lock: threading.Lock = field(default_factory=threading.Lock)


class Registry:
    """Owns every loaded credential for the life of the viewer process."""

    def __init__(self, targets: list[Target], *, httpx_transport=None) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, _Entry] = {}
        for index, target in enumerate(targets):
            client = KeyCall(
                provider=target.provider,
                api_key=target.key,
                protocol=target.protocol,
                base_url=target.base_url,
                httpx_transport=httpx_transport,
            )
            self._entries[index] = _Entry(target=target, client=client)

    def views(self) -> list[TargetView]:
        return [
            TargetView(
                id=entry_id,
                name=safe_display_name(entry.target.display_name),
                provider=entry.client.provider,
                protocol=entry.client.protocol.value,
                base_url=entry.target.base_url,
            )
            for entry_id, entry in self._entries.items()
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
        return entry.discovery if entry else None

    def set_cached_discovery(self, target_id: int, discovery: ModelDiscovery) -> None:
        with self._lock:
            entry = self._entries.get(target_id)
            if entry is not None:
                entry.discovery = discovery

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            entry.client.close()
