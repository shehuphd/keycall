"""Credential-dependent availability cache.

Process-local, bounded, in-memory, cleared on restart. Keyed by provider +
base URL + HMAC credential fingerprint — never the raw key or an unkeyed
digest. Stores the full pre-filter model tuple; category filtering happens
locally so switching filters never re-contacts the provider.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from ._types import Model

__all__ = ["AvailabilityCache", "CachedModels"]

_MAX_ENTRIES = 64
DEFAULT_TTL_SECONDS = 300.0


@dataclass(frozen=True, slots=True, kw_only=True)
class CachedModels:
    models: tuple[Model, ...]
    fetched_at: datetime
    # Carried into every ModelDiscovery built from this entry, so a
    # truncated fetch stays visible on cache hits too.
    warnings: tuple[str, ...] = ()


class AvailabilityCache:
    def __init__(self, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str, str], tuple[float, CachedModels]] = OrderedDict()

    def get(self, provider: str, base_url: str, fingerprint: str) -> CachedModels | None:
        key = (provider, base_url, fingerprint)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, cached = entry
            if time.monotonic() >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return cached

    def put(self, provider: str, base_url: str, fingerprint: str, cached: CachedModels) -> None:
        key = (provider, base_url, fingerprint)
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl, cached)
            self._entries.move_to_end(key)
            while len(self._entries) > _MAX_ENTRIES:
                self._entries.popitem(last=False)

    def invalidate(self, provider: str, base_url: str, fingerprint: str) -> None:
        with self._lock:
            self._entries.pop((provider, base_url, fingerprint), None)


# One process-local cache shared by all clients, matching the process-local
# HMAC fingerprint secret.
shared_cache = AvailabilityCache()
