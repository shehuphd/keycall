"""In-process request log backing the viewer's Traces tab.

One bounded ring of the viewer's own API activity, so "why did that
button hang" is answerable from the page itself instead of a terminal.
Entries carry timing and outcome only — never a prompt, a reply, or a
credential. The prompt text a user typed is theirs; the trace exists to
show where time went and what failed, and the model, provider, duration,
and error text answer that without quoting anyone."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

_MAX_ENTRIES = 200


class TraceLog:
    def __init__(self) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=_MAX_ENTRIES)
        self._lock = threading.Lock()
        self._next_id = 1

    def record(
        self,
        *,
        route: str,
        method: str,
        duration_ms: float,
        status: str,
        target: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        detail: str | None = None,
        events: int | None = None,
    ) -> None:
        entry = {
            "id": 0,
            "at": time.strftime("%H:%M:%S", time.localtime()),
            "route": route,
            "method": method,
            "duration_ms": round(duration_ms, 1),
            "status": status,
            "target": target,
            "provider": provider,
            "model": model,
            # Bounded: error text is already sanitized upstream, but a
            # trace row is a summary, not a transcript.
            "detail": (detail or "")[:300] or None,
            "events": events,
        }
        with self._lock:
            entry["id"] = self._next_id
            self._next_id += 1
            self._entries.append(entry)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._entries))

    def clear(self) -> None:
        """Wipe the log. The id counter keeps counting, so rows recorded
        after a clear never reuse an id a client already saw."""
        with self._lock:
            self._entries.clear()
