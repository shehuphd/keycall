"""Boundary sanitization for provider-originated text.

Every provider error message passes through scrub() before entering a
public result, exception, log, or trace. The transport layer is the single
call site, so nothing constructs a KeyCallError from raw provider text.
"""

from __future__ import annotations

import base64
import re
from urllib.parse import quote

_MAX_MESSAGE_LENGTH = 400
_MAX_REQUEST_ID_LENGTH = 128

# Distinctive credential shapes that may appear inside provider-echoed text
# even when the exact current credential is already replaced.
_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"pplx-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9._\-]{10,}"),  # JWTs
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Request identifiers additionally drop tab/newline/carriage-return: a
# header value is a single token, and an embedded newline forges log lines.
_ALL_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _credential_variants(value: str) -> tuple[str, ...]:
    """The literal credential plus its common encoded forms: a provider
    echoing the key back URL- or base64-encoded must still redact."""
    raw = value.encode("utf-8")
    variants = {
        value,
        quote(value, safe=""),
        base64.b64encode(raw).decode("ascii"),
        base64.urlsafe_b64encode(raw).decode("ascii"),
    }
    # Longest first so an encoded form containing the literal (or vice versa)
    # is replaced whole rather than leaving a recognizable fragment.
    return tuple(sorted(variants, key=len, reverse=True))


def scrub(text: str, *, credential_value: str | None = None) -> str:
    """Redact credentials and credential-shaped values, strip control
    characters, and bound the length."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS.sub("", text)
    if credential_value:
        for variant in _credential_variants(credential_value):
            cleaned = cleaned.replace(variant, "<redacted>")
    for pattern in _VALUE_PATTERNS:
        cleaned = pattern.sub("<redacted>", cleaned)
    if len(cleaned) > _MAX_MESSAGE_LENGTH:
        cleaned = cleaned[:_MAX_MESSAGE_LENGTH] + "…"
    return cleaned


def safe_request_id(raw: object) -> str | None:
    """Bound and clean a provider-supplied request identifier before it
    enters a result or error object. Providers are untrusted input: a
    hostile endpoint could put control characters or log-forging text in a
    response header."""
    if not isinstance(raw, str) or not raw:
        return None
    cleaned = _ALL_CONTROL_CHARS.sub("", raw).strip()
    if not cleaned:
        return None
    if len(cleaned) > _MAX_REQUEST_ID_LENGTH:
        cleaned = cleaned[:_MAX_REQUEST_ID_LENGTH] + "…"
    return cleaned


def safe_display_name(name: str, *, max_length: int = 80) -> str:
    """Bound and clean a user-controlled label for terminal/log output."""
    cleaned = _CONTROL_CHARS.sub("", name).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "…"
    return cleaned
