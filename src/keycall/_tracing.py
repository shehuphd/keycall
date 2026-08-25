"""Optional TraceAct integration.

KeyCall never calls traceact.configure() — that's the host application's
decision. When TraceAct is absent, spans are free no-ops. When it's present
but unconfigured, ActionTrace.start returns TraceAct's own no-op trace.
When an incompatible version is installed, KeyCall disables the integration
with one warning rather than emitting through an API it can't trust.

Safety posture: every span carries a per-call TraceConfig override that
disables function-input and event-input capture, and pins the api_keys and
ai_prompts redaction presets. KeyCall additionally only ever hands TraceAct
explicitly chosen safe fields (provider, model IDs, counts, status,
durations) — the override is defense in depth; KeyCall's own boundary
sanitization is the primary protection. Prompts, responses, credentials,
and auth headers are never passed in.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["span"]

_COMPATIBLE_MIN = (0, 13)
_COMPATIBLE_MAX_EXCLUSIVE = (2, 0)

# Module state: None = not yet checked; False = unavailable; module = usable.
_traceact_module: Any = None
_checked = False


class _NoOpSpan:
    def event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def step(self, *args: Any, **kwargs: Any) -> None:
        pass


_NOOP = _NoOpSpan()


def _version_tuple(raw: str) -> tuple[int, int]:
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (-1, -1)


def _load() -> Any:
    """Import and version-check traceact once. Returns module or False."""
    global _traceact_module, _checked
    if _checked:
        return _traceact_module
    _checked = True
    try:
        import traceact
    except ImportError:
        _traceact_module = False
        return _traceact_module
    version = _version_tuple(getattr(traceact, "__version__", "0.0"))
    if not (_COMPATIBLE_MIN <= version < _COMPATIBLE_MAX_EXCLUSIVE):
        warnings.warn(
            f"keycall: installed traceact {getattr(traceact, '__version__', '?')} is outside "
            f"the supported range (>= {'.'.join(map(str, _COMPATIBLE_MIN))}, "
            f"< {'.'.join(map(str, _COMPATIBLE_MAX_EXCLUSIVE))}); "
            "KeyCall tracing is disabled",
            RuntimeWarning,
            stacklevel=3,
        )
        _traceact_module = False
        return _traceact_module
    _traceact_module = traceact
    return _traceact_module


def _safe_config(traceact: Any) -> Any:
    # Never capture function/event inputs on KeyCall spans regardless of the
    # host's global settings; pin the credential/prompt redaction presets.
    return traceact.TraceConfig(
        capture_inputs=False,
        capture_event_inputs=False,
        redaction_presets=["api_keys", "ai_prompts"],
    )


def _host_configured(traceact: Any) -> bool:
    """True only when the host application has called configure().

    A library must not cause trace output for an app that never opted in:
    emitting through an unconfigured TraceAct writes to its default sink and
    warns about a missing project on every call.
    """
    cfg = traceact.config
    return bool(
        cfg.get_package_sinks() or cfg.get_package_project() or cfg.get_package_config()
    )


@contextmanager
def span(action: str, **meta: Any) -> Iterator[Any]:
    """Trace one KeyCall operation. meta must already be safe fields only."""
    traceact = _load()
    if traceact is False or not _host_configured(traceact):
        yield _NOOP
        return
    with traceact.ActionTrace.start(
        action=action,
        kind="app",
        actor="keycall",
        meta=dict(meta),
        config=_safe_config(traceact),
    ) as trace:
        yield trace


def _reset_for_tests() -> None:
    global _traceact_module, _checked
    _traceact_module = None
    _checked = False
