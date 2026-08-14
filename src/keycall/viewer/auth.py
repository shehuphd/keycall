"""Viewer auth: a token is always required, generated fresh per run and
never persisted. Persisting it (as TraceAct's opt-in token does, in
~/.traceact/viewer.json) would make sense for a read-only trace viewer a
user reopens often; it's the wrong tradeoff here, since this server holds
live credentials and a stale leaked token would grant standing access to
whatever source is loaded on some future run.
"""

from __future__ import annotations

import hmac
import secrets

__all__ = ["Token"]


class Token:
    __slots__ = ("_value",)

    def __init__(self, value: str | None = None) -> None:
        # A value is only ever passed by the dev-reload restart, which must
        # keep the running browser tab's token working across the restart.
        # Every other construction generates fresh.
        self._value = value or secrets.token_urlsafe(32)

    def matches(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(self._value, candidate)

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Token(<redacted>)"
