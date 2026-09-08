"""The retired-model registry: pre-flight refusals on every operation that
names a model, listing withholding with visible warnings, and the catalog
data's own invariants. Wire facts behind the entries are re-verified live
by the release suite's drift probes in test_live.py."""

from __future__ import annotations

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError
from keycall._client import AsyncKeyCall
from keycall._registry import resolve_provider, retired_model_fact, supported_providers
from keycall._types import BatchRequest, Message, TextInput

MESSAGE = [Message(role="user", content=[TextInput(text="hi")])]


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a retired model reached the network: {request.url}")


def make_client(provider: str, key: str = "sk-test") -> KeyCall:
    return KeyCall(
        provider=provider, api_key=key,
        httpx_transport=httpx.MockTransport(no_network),
    )


# --- the lookup -------------------------------------------------------------


def test_retired_model_fact_matches_id_and_alias():
    resolved = resolve_provider("anthropic")
    by_id = retired_model_fact(resolved.retired_models, "claude-3-5-haiku-20241022")
    by_alias = retired_model_fact(resolved.retired_models, "claude-3-5-haiku-latest")
    assert by_id is not None and by_alias is not None
    assert by_id is by_alias
    assert by_id["retired"] == "2026-02-19"
    assert by_id["replacement"] == "claude-haiku-4-5-20251001"
    assert retired_model_fact(resolved.retired_models, "claude-sonnet-4-6") is None


# --- the refusal, per operation ---------------------------------------------


def _expect_retired(callable_, *, fragment: str):
    with pytest.raises(KeyCallError) as caught:
        callable_()
    error = caught.value
    assert error.code is ErrorCode.MODEL_RETIRED
    assert error.retryable is False
    assert fragment in error.message
    return error


def test_generate_text_refuses_a_retired_model_before_the_network():
    client = make_client("anthropic")
    error = _expect_retired(
        lambda: client.generate_text(model="claude-3-5-haiku-20241022", messages=MESSAGE),
        fragment="retired by anthropic on 2026-02-19",
    )
    assert "claude-haiku-4-5-20251001" in error.message
    client.close()


def test_generate_text_refuses_the_alias_spelling_too():
    client = make_client("anthropic")
    _expect_retired(
        lambda: client.generate_text(model="claude-3-5-haiku-latest", messages=MESSAGE),
        fragment="claude-3-5-haiku-latest was retired",
    )
    client.close()


def test_streaming_shares_the_gate():
    client = make_client("anthropic")
    def run():
        with client.stream_text(model="claude-3-opus-latest", messages=MESSAGE) as stream:
            for _ in stream:
                pass
    _expect_retired(run, fragment="claude-3-opus-latest was retired")
    client.close()


def test_image_generation_refuses():
    client = make_client("openai")
    error = _expect_retired(
        lambda: client.generate_image(model="dall-e-3", prompt="a fox"),
        fragment="dall-e-3 was retired by openai on 2026-05-12",
    )
    assert "gpt-image-2" in error.message
    client.close()


def test_embedding_refuses():
    client = make_client("openai")
    _expect_retired(
        lambda: client.embed(model="ada", inputs=["x"]),
        fragment="ada was retired",
    )
    client.close()


def test_video_generation_refuses():
    client = make_client("gemini", key="AIza-test")
    _expect_retired(
        lambda: client.start_video(model="veo-2.0-generate-001", prompt="a fox"),
        fragment="veo-2.0-generate-001 was retired by gemini on 2026-06-30",
    )
    client.close()


def test_batch_refuses_a_retired_model_anywhere_in_the_submission():
    client = make_client("openai")
    requests = [
        BatchRequest(model="gpt-4o-mini", messages=MESSAGE),
        BatchRequest(model="gpt-5-chat-latest", messages=MESSAGE),
    ]
    _expect_retired(
        lambda: client.start_batch(requests),
        fragment="gpt-5-chat-latest was retired",
    )
    client.close()


def test_transcription_refuses():
    client = make_client("elevenlabs")
    error = _expect_retired(
        lambda: client.generate_speech(model="eleven_monolingual_v1", text="hi", voice="v"),
        fragment="eleven_monolingual_v1 was retired by elevenlabs on 2026-07-09",
    )
    assert "eleven_multilingual_v2" in error.message
    client.close()


def test_an_undated_entry_reads_without_inventing_a_date():
    client = make_client("xai", key="xai-test")
    error = _expect_retired(
        lambda: client.generate_text(model="grok-2-1212", messages=MESSAGE),
        fragment="grok-2-1212 was retired by xai",
    )
    assert " on " not in error.message
    assert "named no replacement" in error.message
    client.close()


@pytest.mark.anyio
async def test_async_client_shares_the_gate():
    client = AsyncKeyCall(
        provider="moonshot", api_key="sk-test",
        httpx_transport=httpx.MockTransport(no_network),
    )
    with pytest.raises(KeyCallError) as caught:
        await client.generate_text(model="moonshot-v1-8k", messages=MESSAGE)
    assert caught.value.code is ErrorCode.MODEL_RETIRED
    assert "kimi-k3" in caught.value.message
    await client.close()


# --- the listing filter -----------------------------------------------------


def test_listing_withholds_retired_models_with_a_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"id": "gpt-4o-mini"}, {"id": "dall-e-3"}, {"id": "gpt-5-chat-latest"},
            {"id": "computer-use-preview-2025-03-11"},
        ]})

    client = KeyCall(
        provider="openai", api_key="sk-test",
        httpx_transport=httpx.MockTransport(handler),
    )
    discovery = client.list_models(refresh=True)
    client.close()
    ids = {m.id for m in discovery.models}
    assert ids == {"gpt-4o-mini"}
    assert any(
        "dall-e-3 was retired by openai on 2026-05-12" in w and "gpt-image-2" in w
        and "withheld" in w
        for w in discovery.warnings
    )
    assert any("gpt-5-chat-latest" in w for w in discovery.warnings)
    assert any("computer-use-preview-2025-03-11" in w for w in discovery.warnings)


def test_listing_without_retired_models_carries_no_new_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})

    client = KeyCall(
        provider="openai", api_key="sk-test",
        httpx_transport=httpx.MockTransport(handler),
    )
    discovery = client.list_models(refresh=True)
    client.close()
    assert discovery.warnings == ()


# --- catalog data invariants -------------------------------------------------


def _all_entries():
    for provider in supported_providers():
        resolved = resolve_provider(provider)
        for entry in resolved.retired_models:
            yield provider, resolved, entry


def test_every_entry_is_well_formed():
    seen_any = False
    for provider, _resolved, entry in _all_entries():
        seen_any = True
        assert entry.get("id"), f"{provider}: entry without an id"
        assert entry.get("note"), f"{provider}/{entry['id']}: entry without evidence"
        when = entry.get("retired")
        if when is not None:
            assert len(when) == 10 and when[4] == "-" and when[7] == "-", (
                f"{provider}/{entry['id']}: retired date {when!r} is not YYYY-MM-DD"
            )
    assert seen_any


def test_no_replacement_is_itself_retired():
    """A refusal pointing at another dead model would send the caller on a
    second failing round trip; chains must be resolved in the data."""
    for provider, resolved, entry in _all_entries():
        replacement = entry.get("replacement")
        if replacement:
            assert retired_model_fact(resolved.retired_models, replacement) is None, (
                f"{provider}/{entry['id']}: replacement {replacement} is itself retired"
            )


def test_no_retired_id_doubles_as_a_catalog_model():
    """A maintained catalog model list must never carry an id the same
    catalog says is shut down."""
    for provider, resolved, entry in _all_entries():
        catalog_ids = {m["id"] for m in resolved.catalog_models}
        assert entry["id"] not in catalog_ids, (
            f"{provider}: {entry['id']} is both a catalog model and retired"
        )
