"""Internal redacting credential wrapper.

Not exported. The raw key enters KeyCall at a single boundary (client
construction) and is wrapped here immediately, before any traced internal
workflow begins.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from typing import Any, NoReturn

_REDACTED = "<keycall:redacted-credential>"

# The primary field. A single-key provider carries only this one; the
# fingerprint and the no-argument reveal() both read it, so a model
# provider's behavior is unchanged by the container generalization.
_PRIMARY_FIELD = "api_key"

# Process-local HMAC secret for cache fingerprints. Generated once from OS
# randomness, held only in memory, never logged or persisted.
_FINGERPRINT_SECRET = os.urandom(32)


class Credential:
    """Holds a provider's secret fields, one for a model provider (`api_key`)
    or several for a service provider that authenticates with a pair (LiveKit's
    `api_key` + `api_secret`). Redacted everywhere except `reveal()`.

    `reveal(field)` exists for the transport layer building an authentication
    header, or an adapter minting a token from a key/secret pair. Nothing else
    may call it. Every field value is a secret: `secret_values()` feeds the
    scrubber so none of them can surface in an error or a trace.
    """

    __slots__ = ("_fields",)

    def __init__(self, value: str | Mapping[str, str]) -> None:
        if isinstance(value, str):
            fields = {_PRIMARY_FIELD: value}
        elif isinstance(value, Mapping):
            fields = {str(name): field for name, field in value.items()}
        else:
            raise TypeError("credential must be a string or a mapping of named secret fields")
        if _PRIMARY_FIELD not in fields:
            raise ValueError(f"credential must carry an {_PRIMARY_FIELD!r} field")
        for name, field in fields.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError(f"credential field {name!r} must be a non-empty string")
        self._fields = fields

    def reveal(self, field: str = _PRIMARY_FIELD) -> str:
        try:
            return self._fields[field]
        except KeyError:
            raise ValueError(f"credential has no field {field!r}") from None

    def has_field(self, field: str) -> bool:
        return field in self._fields

    def field_names(self) -> tuple[str, ...]:
        """The field names only — names are catalog vocabulary, never
        secret, so a refusal can name what is missing without revealing
        anything."""
        return tuple(self._fields)

    def secret_values(self) -> tuple[str, ...]:
        """Every field's value, for the scrubber. A pair provider's secret is
        as sensitive as its key, so both must be redactable."""
        return tuple(self._fields.values())

    def fingerprint(self) -> str:
        """Keyed HMAC fingerprint for cache identity. Changes each process.
        Keyed on the primary field only: a service provider does not use the
        model-list cache, so the secret never needs to enter the identity."""
        return hmac.new(
            _FINGERPRINT_SECRET, self._fields[_PRIMARY_FIELD].encode(), hashlib.sha256
        ).hexdigest()

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
