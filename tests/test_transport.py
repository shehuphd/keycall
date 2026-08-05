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
