"""Boundary sanitization for provider-originated text (PRD section 10.2).

Every provider error message passes through scrub() before entering a
public result, exception, log, or trace. The transport layer is the single
call site, so nothing constructs a KeyCallError from raw provider text.
"""

from __future__ import annotations

import re

_MAX_MESSAGE_LENGTH = 400

# Distinctive credential shapes that may appear inside provider-echoed text
# even when the exact current credential is already replaced.
_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9._\-]{10,}"),  # JWTs
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def scrub(text: str, *, credential_value: str | None = None) -> str:
    """Redact credentials and credential-shaped values, strip control
    characters, and bound the length."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS.sub("", text)
    if credential_value:
        cleaned = cleaned.replace(credential_value, "<redacted>")
    for pattern in _VALUE_PATTERNS:
        cleaned = pattern.sub("<redacted>", cleaned)
    if len(cleaned) > _MAX_MESSAGE_LENGTH:
        cleaned = cleaned[:_MAX_MESSAGE_LENGTH] + "…"
    return cleaned


def safe_display_name(name: str, *, max_length: int = 80) -> str:
    """Bound and clean a user-controlled label for terminal/log output."""
    cleaned = _CONTROL_CHARS.sub("", name).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "…"
    return cleaned
