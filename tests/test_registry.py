import pytest

from keycall import ErrorCode, KeyCallError, ProviderProtocol
from keycall._registry import resolve_provider


def test_known_provider_resolves_with_maintained_endpoint():
    resolved = resolve_provider("openai")
    assert resolved.protocol is ProviderProtocol.OPENAI
    assert resolved.base_url == "https://api.openai.com/v1"
    assert not resolved.is_custom


def test_alias_resolves_to_canonical_name():
    assert resolve_provider("claude").provider == "anthropic"
    assert resolve_provider("kimi").provider == "moonshot"
    assert resolve_provider("pplx").provider == "perplexity"


def test_all_v1_named_providers_resolve():
    for name in ("openai", "anthropic", "gemini", "deepseek", "perplexity", "moonshot"):
        resolved = resolve_provider(name)
        assert resolved.base_url.startswith("https://")
        assert "list_models" in resolved.operations
        assert "text_generation" in resolved.operations


def test_unknown_provider_without_custom_path_raises():
    with pytest.raises(KeyCallError) as excinfo:
        resolve_provider("mystery-vendor")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_PROVIDER


def test_base_url_rejected_for_known_provider():
    with pytest.raises(KeyCallError):
        resolve_provider("openai", base_url="https://evil.example.com/v1")


def test_protocol_mismatch_rejected_for_known_provider():
    with pytest.raises(KeyCallError):
        resolve_provider("anthropic", protocol="openai")


def test_custom_target_requires_https():
    with pytest.raises(KeyCallError):
        resolve_provider(
            "lab", protocol="openai-compatible", base_url="http://llm.example.edu/v1"
        )


def test_custom_target_rejects_query_fragment_userinfo():
    for bad in (
        "https://llm.example.edu/v1?debug=1",
        "https://llm.example.edu/v1#frag",
        "https://user:pass@llm.example.edu/v1",
    ):
        with pytest.raises(KeyCallError):
            resolve_provider("lab", protocol="openai-compatible", base_url=bad)


def test_custom_https_target_resolves_and_preserves_base_path():
    resolved = resolve_provider(
        "university-lab",
        protocol="openai-compatible",
        base_url="https://example.edu/gateway/openai/v1",
    )
    assert resolved.is_custom
    assert resolved.base_url == "https://example.edu/gateway/openai/v1"


def test_localhost_http_requires_explicit_opt_in():
    with pytest.raises(KeyCallError):
        resolve_provider("local", protocol="openai-compatible", base_url="http://localhost:8000")
    resolved = resolve_provider(
        "local",
        protocol="openai-compatible",
        base_url="http://localhost:8000",
        allow_insecure_localhost=True,
    )
    assert resolved.base_url == "http://localhost:8000"


def test_capability_facts_come_from_the_catalog():
    """The gates and the error messages that list alternatives read the
    same registry data, so they can't drift apart."""
    from keycall._capabilities import (
        SCHEMA_ENFORCING_PROVIDERS,
        TOOL_CALLING_PROVIDERS,
        WEB_SEARCH_PROVIDERS,
    )
    from keycall._registry import resolve_provider

    assert sorted(TOOL_CALLING_PROVIDERS) == [
        "anthropic", "deepseek", "gemini", "moonshot", "openai", "xai"
    ]
    assert sorted(WEB_SEARCH_PROVIDERS) == [
        "anthropic", "gemini", "moonshot", "openai", "perplexity", "xai"
    ]
    assert "deepseek" not in SCHEMA_ENFORCING_PROVIDERS

    for name in TOOL_CALLING_PROVIDERS:
        assert resolve_provider(name).capabilities.tool_calling
    assert not resolve_provider("perplexity").capabilities.tool_calling
    # Every claim carries the date it was last checked against the live API.
    assert resolve_provider("moonshot").capabilities.verified


def test_custom_targets_get_the_permissive_but_warned_posture():
    from keycall._registry import resolve_provider

    resolved = resolve_provider(
        "mine", protocol="openai-compatible", base_url="https://example.com/v1"
    )
    assert resolved.capabilities.tool_calling, "tools pass through, with a result warning"
    assert not resolved.capabilities.web_search, "never assumed from a protocol label"
    assert resolved.capabilities.schema_enforcement is None


def test_catalog_staleness_is_computed_from_the_verified_stamp():
    from datetime import date

    from keycall._registry import (
        CATALOG_STALE_AFTER_DAYS,
        catalog_age_days,
        catalog_is_stale,
    )

    assert catalog_age_days(now=date(2026, 8, 9)) is not None
    fresh = date(2026, 8, 9)
    assert not catalog_is_stale(now=fresh)
    stale_day = date.fromordinal(fresh.toordinal() + CATALOG_STALE_AFTER_DAYS + 400)
    assert catalog_is_stale(now=stale_day)
    assert catalog_age_days(now=stale_day) > CATALOG_STALE_AFTER_DAYS
