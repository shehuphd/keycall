"""Service providers (kind "service"): probe_services() and the two
adapters, wire-mocked. Live behavior is pinned by the drift probes in
test_live.py; these cover the translation and refusal logic."""


import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, ServiceReport, TextInput

MAPS_KEY = "maps-canary-key-000"
LK_PAIR = {"api_key": "lk-canary-key", "api_secret": "lk-canary-secret"}
LK_URL = "https://demo-abc.livekit.cloud"


def maps_client(handler):
    return KeyCall(
        provider="google_maps", api_key=MAPS_KEY, httpx_transport=httpx.MockTransport(handler)
    )


def livekit_client(handler, base_url=LK_URL):
    return KeyCall(
        provider="livekit",
        credential=dict(LK_PAIR),
        base_url=base_url,
        httpx_transport=httpx.MockTransport(handler),
    )


# --- google maps -------------------------------------------------------------


def test_maps_probe_reports_categories_in_catalog_order_across_hosts():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path))
        return httpx.Response(200, json={"results": []})

    report = maps_client(handler).probe_services()
    assert isinstance(report, ServiceReport)
    assert report.provider == "google_maps"
    assert [(s.name, s.status) for s in report.services] == [
        ("geocoding", "enabled"),
        ("places", "enabled"),
        ("directions", "enabled"),
    ]
    assert [host for host, _ in seen] == [
        "geocode.googleapis.com",
        "places.googleapis.com",
        "routes.googleapis.com",
    ], "each category rides its own catalog host"


def test_maps_probe_sends_the_key_in_the_goog_header_never_the_url():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == MAPS_KEY
        assert MAPS_KEY not in str(request.url)
        return httpx.Response(200, json={})

    maps_client(handler).probe_services()


def test_maps_not_enabled_category_is_denied_with_the_enable_url():
    """One disabled API must not sink the others: per-category status,
    detail carrying Google's own sentence with the console enable URL."""
    enable_url = "https://console.developers.google.com/apis/api/routes.googleapis.com/overview"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "routes.googleapis.com":
            return httpx.Response(
                403,
                json={
                    "error": {
                        "code": 403,
                        "status": "PERMISSION_DENIED",
                        "message": (
                            "Routes API has not been used in project 000 before or "
                            f"it is disabled. Enable it by visiting {enable_url}"
                        ),
                    }
                },
            )
        return httpx.Response(200, json={})

    report = maps_client(handler).probe_services()
    by_name = {s.name: s for s in report.services}
    assert by_name["geocoding"].status == "enabled"
    assert by_name["places"].status == "enabled"
    assert by_name["directions"].status == "denied"
    assert enable_url in by_name["directions"].detail


def test_maps_bad_key_raises_instead_of_reporting():
    """google.rpc spells a bad key as HTTP 400 INVALID_ARGUMENT; that is a
    credential fact, not a category fact, so the report never forms."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "API key not valid. Please pass a valid API key.",
                }
            },
        )

    with pytest.raises(KeyCallError) as excinfo:
        maps_client(handler).probe_services()
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY


def test_maps_unrecognized_failure_is_unknown_not_a_guess():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "places.googleapis.com":
            calls["n"] += 1
            return httpx.Response(500, json={"error": {"message": "backend exploded"}})
        return httpx.Response(200, json={})

    report = maps_client(handler).probe_services()
    by_name = {s.name: s.status for s in report.services}
    assert by_name == {"geocoding": "enabled", "places": "unknown", "directions": "enabled"}


# --- livekit -----------------------------------------------------------------


def test_livekit_probe_enabled_on_an_empty_project():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/twirp/livekit.RoomService/ListRooms"
        assert request.headers.get("authorization", "").startswith("Bearer ")
        return httpx.Response(200, json={"rooms": []})

    report = livekit_client(handler).probe_services()
    assert [(s.name, s.status) for s in report.services] == [("realtime", "enabled")]


def test_livekit_bad_signature_raises_invalid_key():
    """The plain-text 401 body is the signature failure: the api_secret
    does not match, which is credential-level and raises."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid token")

    with pytest.raises(KeyCallError) as excinfo:
        livekit_client(handler).probe_services()
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY
    assert "api_secret" in str(excinfo.value)


def test_livekit_missing_grant_is_a_denied_category():
    """The Twirp-JSON 401 means the pair authenticates but lacks the
    roomList admin permission: a category denial naming the grant."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": "unauthenticated", "msg": "permissions denied"})

    report = livekit_client(handler).probe_services()
    assert report.services[0].status == "denied"
    assert "roomList" in report.services[0].detail


def test_livekit_accepts_the_dashboards_wss_spelling():
    """LiveKit's dashboard hands the project URL out as wss://; the REST
    API is https on the same origin, so the pasted form works as-is."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"rooms": []})

    client = livekit_client(handler, base_url="wss://demo-abc.livekit.cloud")
    assert client.base_url == "https://demo-abc.livekit.cloud"
    client.probe_services()
    assert seen["url"].startswith("https://demo-abc.livekit.cloud/")


# --- kind gates --------------------------------------------------------------


def test_model_provider_probe_services_refuses_naming_the_alternative():
    client = KeyCall(
        provider="anthropic",
        api_key="sk-canary-000",
        httpx_transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.probe_services()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "list_models" in str(excinfo.value)


def test_service_provider_refuses_model_operations_naming_probe_services():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must refuse before any network call")

    client = maps_client(handler)
    with pytest.raises(KeyCallError) as list_err:
        client.list_models()
    with pytest.raises(KeyCallError) as gen_err:
        client.generate_text(
            model="gemini-flash-latest",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    for excinfo in (list_err, gen_err):
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        assert "probe_services" in str(excinfo.value)


def test_service_probe_error_details_scrub_the_secrets():
    """A hostile endpoint echoing the credential into an error body must
    not surface it through a ServiceStatus detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"message": f"denied for {MAPS_KEY} today"}},
        )

    report = maps_client(handler).probe_services()
    for status in report.services:
        assert MAPS_KEY not in (status.detail or "")


@pytest.mark.anyio
async def test_async_probe_services_parity():
    from keycall import AsyncKeyCall

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"rooms": []})

    client = AsyncKeyCall(
        provider="livekit",
        credential=dict(LK_PAIR),
        base_url=LK_URL,
        httpx_transport=httpx.MockTransport(handler),
    )
    report = await client.probe_services()
    assert [(s.name, s.status) for s in report.services] == [("realtime", "enabled")]
    await client.close()
