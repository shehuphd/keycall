import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-transport-key"


def counting_handler(responses):
    """Yield the given responses in order; count calls."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["count"], len(responses) - 1)
        calls["count"] += 1
        return responses[index]

    return handler, calls


def make_client(handler):
    return KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def test_auth_header_built_per_scheme():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    make_client(handler).list_models(refresh=True)
    assert captured["auth"] == f"Bearer {CANARY}"

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        captured["x-api-key"] = request.headers.get("x-api-key")
        captured["version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={"data": [], "has_more": False})

    KeyCall(
        provider="anthropic",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(anthropic_handler),
    ).list_models(refresh=True)
    assert captured["x-api-key"] == CANARY
    assert captured["version"] == "2023-06-01"


def test_list_retries_transient_500_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    handler, calls = counting_handler(
        [
            httpx.Response(500, json={"error": {"message": "boom"}}),
            httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}),
        ]
    )
    discovery = make_client(handler).list_models(refresh=True)
    assert calls["count"] == 2
    assert [m.id for m in discovery.models] == ["gpt-4o"]


def test_list_does_not_retry_auth_failure():
    handler, calls = counting_handler(
        [httpx.Response(401, json={"error": {"message": "bad key"}})]
    )
    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).list_models(refresh=True)
    assert calls["count"] == 1
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY


def test_generation_never_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    handler, calls = counting_handler(
        [httpx.Response(500, json={"error": {"message": "boom"}})]
    )
    client = make_client(handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="gpt-4o", messages=[Message(role="user", content=[TextInput(text="hi")])]
        )
    assert calls["count"] == 1
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert excinfo.value.retryable  # retryable by the caller — not by KeyCall


def test_rate_limit_carries_retry_after():
    handler, _ = counting_handler(
        [
            httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"retry-after": "7"},
            )
        ]
    )
    client = make_client(handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="gpt-4o", messages=[Message(role="user", content=[TextInput(text="hi")])]
        )
    error = excinfo.value
    assert error.code is ErrorCode.RATE_LIMITED
    assert error.retry_after == 7.0


def test_redirect_refused_with_credential():
    handler, calls = counting_handler(
        [httpx.Response(302, headers={"location": "https://elsewhere.example.com/v1/models"})]
    )
    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).list_models(refresh=True)
    assert calls["count"] == 1
    assert "redirect" in excinfo.value.message


def test_network_error_maps_and_never_leaks_key(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused for {CANARY}")

    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).list_models(refresh=True)
    error = excinfo.value
    assert error.code is ErrorCode.NETWORK_ERROR
    assert CANARY not in str(error)


def test_non_json_success_body_is_typed_error():
    handler, _ = counting_handler([httpx.Response(200, text="<html>gateway</html>")])
    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).list_models(refresh=True)
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_retry_after_http_date_form_parsed():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    handler, _ = counting_handler(
        [httpx.Response(429, json={"error": {"message": "later"}}, headers={"retry-after": when})]
    )
    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).generate_text(
            model="gpt-4o", messages=[Message(role="user", content=[TextInput(text="hi")])]
        )
    assert excinfo.value.retry_after is not None
    assert 20.0 < excinfo.value.retry_after <= 30.0


def test_transport_keycall_error_uses_list_retry_budget(monkeypatch):
    """A retryable typed error raised inside send() (the DNS guard's path)
    consumes the list retry budget instead of propagating on first failure."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyCallError(
                "could not resolve host",
                code=ErrorCode.NETWORK_ERROR,
                retryable=True,
            )
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    discovery = make_client(handler).list_models(refresh=True)
    assert calls["count"] == 2
    assert discovery.models[0].id == "gpt-4o-mini"


def test_transport_nonretryable_keycall_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        raise KeyCallError(
            "private address refused",
            code=ErrorCode.UNSUPPORTED_PROVIDER,
        )

    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler).list_models(refresh=True)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_PROVIDER


def test_pagination_truncation_adds_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        # Always reports another page: the 10-page limit must trip.
        return httpx.Response(
            200,
            json={"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "x"},
        )

    client = KeyCall(
        provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    discovery = client.list_models(refresh=True)
    assert any("truncated" in warning for warning in discovery.warnings)
    # The warning survives a cache hit.
    cached = client.list_models()
    assert cached.from_cache
    assert any("truncated" in warning for warning in cached.warnings)


def test_untruncated_list_has_no_truncation_warning():
    handler, _ = counting_handler([httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})])
    discovery = make_client(handler).list_models(refresh=True)
    assert not any("truncated" in warning for warning in discovery.warnings)


def test_proxy_env_with_guarded_custom_target_refuses(monkeypatch):
    """A proxy env var routes custom-target requests around the
    DNS-rebinding/private-address guard, so construction fails closed
    (it used to warn and proceed with the guard silently disabled)."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    with pytest.raises(KeyCallError) as excinfo:
        KeyCall(
            provider="my-lab",
            api_key=CANARY,
            protocol="openai-compatible",
            base_url="https://llm.example.edu/v1",
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "bypass the DNS-rebinding" in excinfo.value.message
    # The message names every way out.
    assert "trust_env=False" in excinfo.value.message
    assert "allow_private_network=True" in excinfo.value.message
    assert CANARY not in excinfo.value.message


def test_proxy_env_with_trust_env_false_constructs(monkeypatch):
    """trust_env=False ignores the proxy variables, so the guard applies
    and construction goes through."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    client = KeyCall(
        provider="my-lab",
        api_key=CANARY,
        protocol="openai-compatible",
        base_url="https://llm.example.edu/v1",
        trust_env=False,
    )
    assert client.provider == "my-lab"
    client.close()


def test_proxy_env_with_allow_private_network_constructs(monkeypatch):
    """allow_private_network=True waives the guard explicitly, so a
    deliberate proxy route is accepted rather than refused."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    client = KeyCall(
        provider="my-lab",
        api_key=CANARY,
        protocol="openai-compatible",
        base_url="https://llm.example.edu/v1",
        allow_private_network=True,
    )
    assert client.provider == "my-lab"
    client.close()


def test_no_proxy_env_constructs(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    client = KeyCall(
        provider="my-lab",
        api_key=CANARY,
        protocol="openai-compatible",
        base_url="https://llm.example.edu/v1",
    )
    assert client.provider == "my-lab"
    client.close()


@pytest.mark.parametrize(
    "name",
    ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"],
)
def test_every_proxy_variable_spelling_refuses(monkeypatch, name):
    """The check must cover every spelling httpx honours; losing one from
    the tuple would silently fail open for that spelling alone."""
    for other in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(other, raising=False)
        monkeypatch.delenv(other.lower(), raising=False)
    monkeypatch.setenv(name, "http://proxy.example.com:8080")
    with pytest.raises(KeyCallError):
        KeyCall(
            provider="my-lab",
            api_key=CANARY,
            protocol="openai-compatible",
            base_url="https://llm.example.edu/v1",
        )


def test_proxy_env_does_not_refuse_named_providers(monkeypatch):
    """The refusal is scoped to guarded custom targets. Named providers
    route to catalog hostnames and never had the guard, so a corporate
    proxy must not break them — widening the check would."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    client = KeyCall(provider="openai", api_key=CANARY)
    assert client.provider == "openai"
    client.close()


def test_async_client_refuses_proxy_env_for_guarded_custom_target(monkeypatch):
    """AsyncTransport duplicates the guard wiring, so it needs its own
    refusal test: a mutation removing only the async-side check survived
    the sync-only test (found by mutation testing 2026-08-29)."""
    from keycall import AsyncKeyCall

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    with pytest.raises(KeyCallError) as excinfo:
        AsyncKeyCall(
            provider="my-lab",
            api_key=CANARY,
            protocol="openai-compatible",
            base_url="https://llm.example.edu/v1",
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert CANARY not in excinfo.value.message


def test_pair_credential_secret_scrubbed_from_error_bodies():
    """A hostile or echoing provider error that repeats the api_secret of a
    key/secret pair must come back redacted, through the transport's own
    error path, the same way the api_key already does. Sends the secret in
    the raw response body, the shape production traffic takes."""
    from keycall._credential import Credential
    from keycall._registry import resolve_provider
    from keycall._transport import RequestSpec, Transport

    secret = "livekit-pair-secret-f00ba4"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"bad signature for {CANARY} with {secret}"}}
        )

    def translate(status_code, payload):
        # Surface the provider body the way a production adapter's translate_error
        # does, so the scrub runs against text that carries the secret.
        message = payload.get("error", {}).get("message", "")
        return ErrorCode.INVALID_API_KEY, False, f"provider said: {message}"

    resolved = resolve_provider("openai")
    transport = Transport(
        resolved,
        Credential({"api_key": CANARY, "api_secret": secret}),
        httpx_transport=httpx.MockTransport(handler),
    )
    spec = RequestSpec(method="GET", path="/v1/models")
    with pytest.raises(KeyCallError) as excinfo:
        transport.request(
            spec, operation="list_models", retry_policy="generation", translate_error=translate
        )
    rendered = str(excinfo.value)
    assert "provider said" in rendered, "the body must reach the message"
    assert CANARY not in rendered
    assert secret not in rendered


def test_jwt_hs256_scheme_mints_a_verifiable_token_per_request():
    """A livekit-resolved transport sends Bearer <JWT>: HS256-signed with
    the api_secret, iss = api_key, the spec's grants under video, and a
    bounded expiry. Verified by re-signing in the test, not by decoding
    claims alone."""
    import base64 as b64
    import hashlib
    import hmac as hmac_mod
    import json as json_mod
    import time as time_mod

    from keycall._credential import Credential
    from keycall._registry import resolve_provider
    from keycall._transport import RequestSpec, Transport

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"rooms": []})

    resolved = resolve_provider("livekit", base_url="https://demo-abc.livekit.cloud")
    transport = Transport(
        resolved,
        Credential({"api_key": "lk-key-canary", "api_secret": "lk-secret-canary"}),
        httpx_transport=httpx.MockTransport(handler),
    )
    spec = RequestSpec(
        method="POST",
        path="/twirp/livekit.RoomService/ListRooms",
        json_body={},
        jwt_grants={"roomList": True},
    )
    transport.request(spec, operation="list_models", retry_policy="generation")

    assert captured["url"] == "https://demo-abc.livekit.cloud/twirp/livekit.RoomService/ListRooms"
    assert captured["auth"].startswith("Bearer ")
    token = captured["auth"].removeprefix("Bearer ")
    header_part, payload_part, signature_part = token.split(".")

    def unpad(part: str) -> bytes:
        return b64.urlsafe_b64decode(part + "=" * (-len(part) % 4))

    expected = b64.urlsafe_b64encode(
        hmac_mod.new(
            b"lk-secret-canary", f"{header_part}.{payload_part}".encode(), hashlib.sha256
        ).digest()
    ).rstrip(b"=").decode()
    assert signature_part == expected, "signature must verify against the api_secret"
    claims = json_mod.loads(unpad(payload_part))
    assert claims["iss"] == "lk-key-canary"
    assert claims["video"] == {"roomList": True}
    assert claims["exp"] - time_mod.time() < 700, "short-lived, never a standing token"
    assert json_mod.loads(unpad(header_part)) == {"alg": "HS256", "typ": "JWT"}


def test_spec_host_overrides_the_origin_for_one_request():
    """A catalog category host (Maps splits categories across hosts) routes
    that one request; the resolved base_url still serves specs without one."""
    from keycall._credential import Credential
    from keycall._registry import resolve_provider
    from keycall._transport import RequestSpec, Transport

    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={})

    transport = Transport(
        resolve_provider("google_maps"),
        Credential("maps-key-canary"),
        httpx_transport=httpx.MockTransport(handler),
    )
    transport.request(
        RequestSpec(method="GET", path="/v4beta/geocode/address", host="geocode.googleapis.com"),
        operation="list_models",
        retry_policy="generation",
    )
    transport.request(
        RequestSpec(method="POST", path="/v1/places:searchText", json_body={}),
        operation="list_models",
        retry_policy="generation",
    )
    assert urls[0].startswith("https://geocode.googleapis.com/v4beta/")
    assert urls[1].startswith("https://places.googleapis.com/v1/")


def test_google_maps_auth_rides_the_goog_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["goog"] = request.headers.get("x-goog-api-key")
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    from keycall._credential import Credential
    from keycall._registry import resolve_provider
    from keycall._transport import RequestSpec, Transport

    transport = Transport(
        resolve_provider("google_maps"),
        Credential("maps-key-canary"),
        httpx_transport=httpx.MockTransport(handler),
    )
    transport.request(
        RequestSpec(method="GET", path="/x"), operation="list_models", retry_policy="generation"
    )
    assert captured["goog"] == "maps-key-canary"
    assert captured["authorization"] is None, "the key never rides a bearer header or a URL"
