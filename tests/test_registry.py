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
