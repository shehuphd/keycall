"""Internal redacting credential wrapper.

Not exported. The raw key enters KeyCall at exactly one boundary (client
construction) and is wrapped here immediately, before any traced internal
workflow begins (PRD section 10.1).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any, NoReturn

_REDACTED = "<keycall:redacted-credential>"

# Process-local HMAC secret for cache fingerprints. Generated once from OS
# randomness, held only in memory, never logged or persisted (PRD 10.3).
_FINGERPRINT_SECRET = os.urandom(32)


class Credential:
    """Holds a provider API key. Redacted everywhere except `reveal()`.

    `reveal()` exists for the transport layer building an authentication
    header. Nothing else may call it.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("api_key must be a non-empty string")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def fingerprint(self) -> str:
        """Keyed HMAC fingerprint for cache identity. Changes each process."""
        return hmac.new(_FINGERPRINT_SECRET, self._value.encode(), hashlib.sha256).hexdigest()

    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, spec: str) -> str:
        return _REDACTED

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("credentials cannot be pickled or copied")

    def __deepcopy__(self, memo: Any) -> NoReturn:
        raise TypeError("credentials cannot be pickled or copied")

    def __copy__(self) -> NoReturn:
        raise TypeError("credentials cannot be pickled or copied")
