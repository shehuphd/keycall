"""LiveKit service adapter: no models, one probe category.

The probe is RoomService ListRooms over Twirp on the caller's per-project
host, authenticated by a per-request HS256 token the transport mints from
the api_key/api_secret pair with the video.roomList grant. An empty
project answers 200 with an empty rooms array, which validates the pair
(live-probed 2026-09-10). The two 401 bodies are distinguishable: a bad
signature answers plain text, a valid signature without the grant answers
Twirp JSON — so a wrong secret and a key without admin permission each
get their own refusal.
"""

from __future__ import annotations

from typing import Any

from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import ServiceStatus
from ._base import ServiceProviderAdapter


class LiveKitAdapter(ServiceProviderAdapter):
    def service_probe_specs(self) -> tuple[tuple[str, RequestSpec], ...]:
        specs: list[tuple[str, RequestSpec]] = []
        for category in self.resolved.service_categories:
            name = str(category["name"])
            if name != "realtime":
                raise KeyCallError(
                    f"the catalog names a livekit category {name!r} this "
                    "adapter has no probe for; the catalog and the adapter "
                    "must move together",
                    code=ErrorCode.CATALOG_UPDATE_REQUIRED,
                    provider=self.resolved.provider,
                )
            specs.append(
                (
                    name,
                    RequestSpec(
                        method=str(category["method"]),
                        path=str(category["path"]),
                        json_body={},
                        jwt_grants={"roomList": True},
                    ),
                )
            )
        return tuple(specs)

    def service_status_from_payload(self, category: str, payload: Any) -> ServiceStatus:
        return ServiceStatus(name=category, status="enabled")

    def service_status_from_error(self, category: str, error: KeyCallError) -> ServiceStatus:
        if error.code is ErrorCode.PERMISSION_DENIED:
            return ServiceStatus(name=category, status="denied", detail=error.message)
        return ServiceStatus(name=category, status="unknown", detail=error.message)

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        if status_code == 401:
            if isinstance(payload, dict) and "permissions" in str(payload.get("msg", "")):
                # Twirp JSON with a valid signature: the pair authenticates
                # but this key lacks the admin grant the probe needs.
                return (
                    ErrorCode.PERMISSION_DENIED,
                    False,
                    (
                        "the key pair authenticates, but this key lacks the "
                        "roomList admin permission; grant it in the LiveKit "
                        "project's key settings"
                    ),
                )
            # Plain-text "invalid token": the signature failed, so the
            # api_secret does not match this api_key.
            return (
                ErrorCode.INVALID_API_KEY,
                False,
                (
                    "livekit rejected the token signature: the api_secret "
                    "does not match this api_key"
                ),
            )
        return super().translate_error(status_code, payload)
