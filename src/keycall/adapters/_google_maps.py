"""Google Maps Platform service adapter: no models, three probe categories.

Each probe is the cheapest request its category answers (live-probed
2026-09-10): geocoding rides the v4beta surface because it takes the
X-Goog-Api-Key header where the legacy endpoint refuses it, places asks
for ids only (Text Search's cheapest field mask), directions asks for a
route's duration alone. Hosts and paths come from the catalog entry, so
the credential can only reach a catalog-named host.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import ServiceStatus
from ._base import ServiceProviderAdapter


def _geocoding_spec(category: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        method=str(category["method"]),
        path=str(category["path"]),
        host=str(category["host"]),
        params={"addressQuery": "1600 Amphitheatre Parkway, Mountain View, CA"},
    )


def _places_spec(category: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        method=str(category["method"]),
        path=str(category["path"]),
        host=str(category["host"]),
        json_body={"textQuery": "coffee", "maxResultCount": 1},
        headers={"X-Goog-FieldMask": "places.id"},
    )


def _directions_spec(category: Mapping[str, Any]) -> RequestSpec:
    return RequestSpec(
        method=str(category["method"]),
        path=str(category["path"]),
        host=str(category["host"]),
        json_body={
            "origin": {"address": "Victoria Station, London"},
            "destination": {"address": "London Bridge Station, London"},
            "travelMode": "TRANSIT",
        },
        headers={"X-Goog-FieldMask": "routes.duration"},
    )


_PROBE_BUILDERS = {
    "geocoding": _geocoding_spec,
    "places": _places_spec,
    "directions": _directions_spec,
}


class GoogleMapsAdapter(ServiceProviderAdapter):
    def service_probe_specs(self) -> tuple[tuple[str, RequestSpec], ...]:
        specs: list[tuple[str, RequestSpec]] = []
        for category in self.resolved.service_categories:
            name = str(category["name"])
            build = _PROBE_BUILDERS.get(name)
            if build is None:
                # A catalog category this adapter has no probe recipe for
                # is a drift defect between the two, caught loudly rather
                # than silently reporting fewer categories than declared.
                raise KeyCallError(
                    f"the catalog names a google_maps category {name!r} "
                    "this adapter has no probe for; the catalog and the "
                    "adapter must move together",
                    code=ErrorCode.CATALOG_UPDATE_REQUIRED,
                    provider=self.resolved.provider,
                )
            specs.append((name, build(category)))
        return tuple(specs)

    def service_status_from_payload(self, category: str, payload: Any) -> ServiceStatus:
        return ServiceStatus(name=category, status="enabled")

    def service_status_from_error(self, category: str, error: KeyCallError) -> ServiceStatus:
        if error.code is ErrorCode.PERMISSION_DENIED:
            # The 403 message names the API and carries the console enable
            # URL for it (live-probed 2026-09-10), which is the fix.
            return ServiceStatus(name=category, status="denied", detail=error.message)
        return ServiceStatus(name=category, status="unknown", detail=error.message)

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        code, retryable, message = super().translate_error(status_code, payload)
        if status_code == 400 and "API key not valid" in message:
            # google.rpc spells a bad key as 400 INVALID_ARGUMENT, not 401
            # (live-probed 2026-09-10); without this it would read as a
            # malformed request rather than a bad credential.
            return ErrorCode.INVALID_API_KEY, False, message
        return code, retryable, message
