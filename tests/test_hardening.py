"""Sampling-param gating, response-size cap, SSRF guard, Sonar route."""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-hardening-key"


def make_client(provider="openai", handler=None, **kwargs):
    transport = httpx.MockTransport(handler) if handler else None
    return KeyCall(provider=provider, api_key=CANARY, httpx_transport=transport, **kwargs)


def simple_messages():
    return [Message(role="user", content=[TextInput(text="hi")])]


# --- sampling params -------------------------------------------------------


def test_temperature_passed_through_for_supporting_models():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"status": "completed", "output": [], "usage": {}},
        )

    client = make_client(handler=handler)
    client.generate_text(
        model="gpt-4o-mini", messages=simple_messages(), temperature=0.2, top_p=0.9
    )
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["top_p"] == 0.9


def test_temperature_omitted_when_unset():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    make_client(handler=handler).generate_text(model="gpt-4o-mini", messages=simple_messages())
    assert "temperature" not in captured["body"]
    assert "top_p" not in captured["body"]


@pytest.mark.parametrize("model", ["o3-mini", "o1", "gpt-5", "gpt-5-mini"])
def test_sampling_rejected_before_network_for_reasoning_models(model):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client(handler=handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model=model, messages=simple_messages(), temperature=0.5)
    assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE
    assert model in excinfo.value.message


def test_reasoning_model_without_sampling_params_passes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    result = make_client(handler=handler).generate_text(
        model="o3-mini", messages=simple_messages()
    )
    assert result.model == "o3-mini"


def test_gemini_sampling_lands_in_generation_config():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [], "usageMetadata": {}})

    make_client(provider="gemini", handler=handler).generate_text(
        model="gemini-2.5-flash",
        messages=simple_messages(),
        temperature=0.3,
        top_p=0.8,
        max_output_tokens=64,
    )
    config = captured["body"]["generationConfig"]
    assert config == {"maxOutputTokens": 64, "temperature": 0.3, "topP": 0.8}


def test_invalid_sampling_values_rejected_at_request_construction():
    from keycall import TextGenerationRequest

    with pytest.raises(ValueError):
        TextGenerationRequest(model="m", messages=simple_messages(), temperature=3.0)
    with pytest.raises(ValueError):
        TextGenerationRequest(model="m", messages=simple_messages(), top_p=0.0)


# --- response-size cap -----------------------------------------------------


def test_oversized_response_rejected():
    big = b'{"data": ["' + b"x" * 2048 + b'"]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "application/json"})

    client = make_client(handler=handler, max_response_bytes=1024)
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "byte limit" in excinfo.value.message


def test_normal_response_passes_under_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})

    discovery = make_client(handler=handler, max_response_bytes=1024).list_models(refresh=True)
    assert [m.id for m in discovery.models] == ["gpt-4o"]


# --- SSRF guard ------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "https://10.0.0.5/v1",
        "https://192.168.1.10/v1",
        "https://172.16.0.1/v1",
        "https://169.254.169.254/v1",  # cloud metadata endpoint
        "https://127.0.0.1/v1",
    ],
)
def test_private_ip_base_url_rejected_by_default(base_url):
    with pytest.raises(KeyCallError) as excinfo:
        KeyCall(
            provider="internal-target",
            protocol="openai-compatible",
            api_key=CANARY,
            base_url=base_url,
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_PROVIDER
    assert "private" in excinfo.value.message


def test_private_ip_allowed_with_explicit_opt_in():
    client = KeyCall(
        provider="internal-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://10.0.0.5/v1",
        allow_private_network=True,
        httpx_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": []})
        ),
    )
    assert client.base_url == "https://10.0.0.5/v1"


def test_localhost_http_still_needs_only_the_localhost_flag():
    client = KeyCall(
        provider="local-dev",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="http://127.0.0.1:8000",
        allow_insecure_localhost=True,
        httpx_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": []})
        ),
    )
    assert client.base_url == "http://127.0.0.1:8000"


def test_public_hostname_still_fine():
    client = KeyCall(
        provider="lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": []})
        ),
    )
    assert client.base_url == "https://llm.example.edu/v1"


# --- Perplexity Sonar canonical route --------------------------------------


def test_perplexity_generation_uses_sonar_route():
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "sonar-pro"}]})
        return httpx.Response(
            200,
            json={
                "model": "sonar-pro",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    client = make_client(provider="perplexity", handler=handler)
    client.list_models(refresh=True)
    result = client.generate_text(model="sonar-pro", messages=simple_messages())
    assert "/v1/sonar" in seen_paths
    assert "/chat/completions" not in seen_paths
    assert result.text == "answer"


# --- DNS rebinding guard ---------------------------------------------------


def test_dns_guard_rejects_hostname_resolving_to_private_ip(monkeypatch):
    import socket as socket_module

    from keycall import _dnsguard

    def fake_getaddrinfo(host, port, *args, **kwargs):
        # Attacker's second answer: an internal address.
        return [(socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("10.0.0.7", port))]

    monkeypatch.setattr(_dnsguard.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(KeyCallError) as excinfo:
        _dnsguard.validate_and_pin(
            httpx.URL("https://rebind.example.com/v1"), provider="lab"
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_PROVIDER
    assert "private/internal" in excinfo.value.message


def test_dns_guard_pins_public_address_and_keeps_hostname(monkeypatch):
    import socket as socket_module

    from keycall import _dnsguard

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(_dnsguard.socket, "getaddrinfo", fake_getaddrinfo)
    pinned = _dnsguard.validate_and_pin(
        httpx.URL("https://llm.example.edu/v1/models"), provider="lab"
    )
    assert pinned is not None
    url, original_host = pinned
    assert url.host == "93.184.216.34"          # connects to the checked IP
    assert original_host == "llm.example.edu"   # TLS/Host keep the real name
    assert url.path == "/v1/models"


def test_dns_guard_rejects_when_any_resolved_address_is_private(monkeypatch):
    import socket as socket_module

    from keycall import _dnsguard

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("192.168.0.5", port)),
        ]

    monkeypatch.setattr(_dnsguard.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(KeyCallError):
        _dnsguard.validate_and_pin(httpx.URL("https://mixed.example.com/v1"), provider="lab")


def test_dns_guard_skips_literal_ip_targets():
    from keycall import _dnsguard

    # Registry already validated literal IPs; nothing to race.
    assert _dnsguard.validate_and_pin(httpx.URL("https://93.184.216.34/v1"), provider="lab") is None


def test_guarded_transport_applied_to_custom_targets_only():
    from keycall import _dnsguard

    custom = KeyCall(
        provider="lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
    )
    assert isinstance(custom._transport._client._transport, _dnsguard.GuardedTransport)

    named = KeyCall(provider="openai", api_key=CANARY)
    assert not isinstance(named._transport._client._transport, _dnsguard.GuardedTransport)


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-7", "claude-opus-4-8-20260101", "claude-opus-5", "claude-sonnet-5"],
)
def test_anthropic_deprecated_sampling_models_gated(model):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client(provider="anthropic", handler=handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model=model, messages=simple_messages(), temperature=0.5)
    assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE


@pytest.mark.parametrize("model", ["claude-opus-4-1", "claude-3-5-sonnet-20241022"])
def test_older_anthropic_models_still_accept_sampling(model):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}], "usage": {}})

    make_client(provider="anthropic", handler=handler).generate_text(
        model=model, messages=simple_messages(), temperature=0.5
    )
    assert captured["body"]["temperature"] == 0.5


# --- Perplexity: catalog-supplied models ------------------------------------


def test_perplexity_returns_catalog_models_not_agent_router_list():
    """GET /v1/models is Agent-API-scoped; its entries reject on /v1/sonar."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/claude-fable-5"},
                    {"id": "perplexity/sonar"},
                ]
            },
        )

    discovery = make_client(provider="perplexity", handler=handler).list_models(refresh=True)
    ids = [m.id for m in discovery.models]
    assert ids == ["sonar", "sonar-pro", "sonar-reasoning-pro"]
    assert "anthropic/claude-fable-5" not in ids
    assert "perplexity/sonar" not in ids
    assert discovery.models[0].classification_source == "keycall_catalog"
    assert discovery.models[0].warnings


def test_perplexity_invalid_model_400_maps_to_model_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid model 'anthropic/claude-fable-5'."}},
        )

    client = make_client(provider="perplexity", handler=handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="anthropic/claude-fable-5", messages=simple_messages(), max_output_tokens=16
        )
    assert excinfo.value.code is ErrorCode.MODEL_NOT_AVAILABLE


def test_perplexity_rejects_max_tokens_below_provider_minimum():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client(provider="perplexity", handler=handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model="sonar", messages=simple_messages(), max_output_tokens=8)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert ">= 16" in excinfo.value.message


# --- Gemini: generateContent is transport, not modality ---------------------


def test_gemini_tts_model_kept_out_of_text_picker():
    """A TTS variant advertises generateContent but rejects a TEXT response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash-preview-tts",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/gemini-flash-latest",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemma-4-31b-it",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            },
        )

    client = make_client(provider="gemini", handler=handler)
    text_ids = [m.id for m in client.list_models(refresh=True).models]
    assert text_ids == ["gemini-flash-latest", "gemma-4-31b-it"]

    from keycall import ModelCategory

    speech = client.list_models(categories={ModelCategory.SPEECH_GENERATION})
    assert [m.id for m in speech.models] == ["gemini-2.5-flash-preview-tts"]
    assert speech.models[0].classification_source == "provider_metadata+keycall_rule"


# --- boundary sanitization (scrub, request ids) -----------------------------


def test_scrub_redacts_encoded_credential_forms():
    import base64
    from urllib.parse import quote

    from keycall._sanitize import scrub

    key = "sk-canary+key/with=chars"
    forms = (
        key,
        quote(key, safe=""),
        base64.b64encode(key.encode()).decode(),
        base64.urlsafe_b64encode(key.encode()).decode(),
    )
    for form in forms:
        cleaned = scrub(f"provider rejected {form} outright", credential_value=key)
        assert form not in cleaned
        assert "<redacted>" in cleaned


def test_scrub_redacts_perplexity_key_pattern():
    from keycall._sanitize import scrub

    cleaned = scrub("invalid key pplx-abcdefgh12345678 supplied")
    assert "pplx-abcdefgh12345678" not in cleaned
    assert "<redacted>" in cleaned


def test_safe_request_id_strips_controls_and_bounds():
    from keycall._sanitize import safe_request_id

    assert safe_request_id("req\x1b[31m-1\r\nfake: line") == "req[31m-1fake: line"
    assert safe_request_id("x" * 300) == "x" * 128 + "…"
    assert safe_request_id(None) is None
    assert safe_request_id("") is None
    assert safe_request_id("\x00\x01") is None
    assert safe_request_id(42) is None


def test_hostile_request_id_header_sanitized_in_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            headers={"x-request-id": "req\x1b[2Jwipe"},
        )

    with pytest.raises(KeyCallError) as excinfo:
        make_client(handler=handler).list_models(refresh=True)
    assert excinfo.value.provider_request_id == "req[2Jwipe"


def test_hostile_request_id_header_sanitized_in_result():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
            headers={"x-request-id": "ok\x1b[31mid"},
        )

    result = make_client(handler=handler).generate_text(
        model="gpt-4o-mini", messages=simple_messages()
    )
    assert result.provider_request_id == "ok[31mid"


def test_payment_required_is_not_reported_as_a_malformed_response():
    """A 402 means the key is fine and the account is not funded. Falling
    through to invalid_provider_response sends the caller hunting for a bug
    in their request. Seen live from Tinker: "Access ... is blocked due to
    billing status. Please add payment at ...\"."""
    detail = "Access is blocked due to billing status. Please add payment at https://example/billing"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": detail}})

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.list_models(refresh=True)
    client.close()
    assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
    # The provider's message carries the only actionable part: where to pay.
    assert "billing" in str(excinfo.value)
