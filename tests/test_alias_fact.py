"""alias_fact(): rolling-alias classification from recorded catalog
conventions. Adversarial cases first: the function must never guess — a
provider with no recorded convention returns None for every id shape, and
an unknown provider is an error, not a silent None."""

import json

import httpx
import pytest

from keycall import AliasFact, ErrorCode, KeyCall, KeyCallError, alias_fact

CANARY = "sk-canary-alias-key"


# --- refusals and absences before the happy path ------------------------------


def test_unknown_provider_raises_not_none():
    with pytest.raises(KeyCallError) as excinfo:
        alias_fact(provider="not-a-provider", model_id="anything-latest")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_PROVIDER


def test_provider_without_a_recorded_convention_returns_none_even_for_latest_shapes():
    """Absent, never a guess: an alias-looking id under a provider with no
    recorded convention gets no fact, because there's no evidence to back
    one."""
    for provider in ("anthropic", "deepseek", "moonshot", "perplexity", "xai"):
        assert alias_fact(provider=provider, model_id="some-model-latest") is None


def test_dated_and_pinned_ids_return_none_under_a_convention_provider():
    assert alias_fact(provider="openai", model_id="gpt-5.2-2025-12-11") is None
    assert alias_fact(provider="openai", model_id="gpt-5.6") is None
    assert alias_fact(provider="gemini", model_id="gemini-2.5-pro-002") is None


def test_mid_string_latest_does_not_match_a_suffix_convention():
    assert alias_fact(provider="gemini", model_id="gemini-latest-preview-002") is None
    assert alias_fact(provider="openai", model_id="chat-latest-gpt") is None


def test_openai_bare_latest_suffix_is_not_the_chat_latest_convention():
    """OpenAI's recorded convention is the -chat-latest family only; a
    hypothetical bare -latest id has no recorded evidence there."""
    assert alias_fact(provider="openai", model_id="gpt-6-latest") is None


# --- convention-correct facts, per provider ----------------------------------


def test_openai_chat_latest_family_reports_not_maintained():
    fact = alias_fact(provider="openai", model_id="gpt-5.3-chat-latest")
    assert isinstance(fact, AliasFact)
    assert fact.maintained is False
    assert fact.provider == "openai"
    assert fact.convention == "-chat-latest suffix"
    assert fact.verified == "2026-08-10"
    assert fact.note


def test_gemini_latest_reports_maintained():
    fact = alias_fact(provider="gemini", model_id="gemini-flash-latest")
    assert isinstance(fact, AliasFact)
    assert fact.maintained is True
    assert fact.convention == "-latest suffix"
    assert fact.verified == "2026-08-09"


def test_provider_aliases_resolve_to_the_same_conventions():
    via_alias = alias_fact(provider="google", model_id="gemini-pro-latest")
    direct = alias_fact(provider="gemini", model_id="gemini-pro-latest")
    assert via_alias == direct
    assert via_alias is not None and via_alias.provider == "gemini"


def test_matching_is_case_insensitive_and_preserves_the_original_id():
    fact = alias_fact(provider="openai", model_id="GPT-5.3-CHAT-LATEST")
    assert fact is not None
    assert fact.model_id == "GPT-5.3-CHAT-LATEST"


# --- Model.alias at discovery -------------------------------------------------


def _gemini_client(model_ids):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/models" in request.url.path
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": f"models/{mid}",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                    for mid in model_ids
                ]
            },
        )

    return KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def test_discovery_attaches_the_fact_only_to_convention_matching_ids():
    client = _gemini_client(["gemini-flash-latest", "gemini-2.5-pro-002"])
    try:
        discovery = client.list_models(refresh=True)
        by_id = {m.id: m for m in discovery.models}
        assert by_id["gemini-flash-latest"].alias is not None
        assert by_id["gemini-flash-latest"].alias.maintained is True
        assert by_id["gemini-2.5-pro-002"].alias is None
    finally:
        client.close()


def test_discovery_fact_survives_the_cache():
    client = _gemini_client(["gemini-flash-latest"])
    try:
        client.list_models(refresh=True)
        cached = client.list_models()
        assert cached.from_cache
        assert cached.models[0].alias is not None
    finally:
        client.close()


def test_alias_fact_never_leaks_into_serialized_model_payloads():
    """The viewer serializer mirrors the fact; a canary key must never ride
    along anywhere in that structure."""
    from keycall.viewer._api import _model_dict

    client = _gemini_client(["gemini-flash-latest"])
    try:
        discovery = client.list_models(refresh=True)
        payload = _model_dict(discovery.models[0])
        assert payload["alias"]["maintained"] is True
        assert CANARY not in json.dumps(payload)
    finally:
        client.close()


def test_viewer_serializer_sends_null_without_a_fact():
    from keycall.viewer._api import _model_dict

    client = _gemini_client(["gemini-2.5-pro-002"])
    try:
        model = client.list_models(refresh=True).models[0]
    finally:
        client.close()
    assert _model_dict(model)["alias"] is None
