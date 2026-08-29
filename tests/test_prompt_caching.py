"""Prompt caching: TextInput(cacheable=True) on Anthropic and OpenAI.

Anthropic requires this marker for any caching to happen at all; OpenAI
already caches automatically and the marker only asks for its optional
explicit-breakpoint mode. Every other provider ignores the field
completely, the same as it already does today with no marker at all.
"""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-caching-key"


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider,
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _anthropic_response(text="ok", cache_creation=0, cache_read=0):
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    )


def _openai_response(text="ok", cached_tokens=0):
    return httpx.Response(
        200,
        json={
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "total_tokens": 13,
            },
        },
    )


# --- TextInput's own validation ---------------------------------------------


def test_cache_ttl_seconds_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        TextInput(text="x", cacheable=True, cache_ttl_seconds=0)


def test_default_text_input_is_not_cacheable():
    part = TextInput(text="x")
    assert part.cacheable is False
    assert part.cache_ttl_seconds == 300


# --- Anthropic: the provider that needs the marker ---------------------------


def test_anthropic_sets_cache_control_on_a_marked_message_block():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _anthropic_response()

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-5",
        messages=[
            Message(role="user", content=[TextInput(text="hello", cacheable=True)]),
        ],
    )
    block = captured["body"]["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_anthropic_uncached_block_carries_no_cache_control():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _anthropic_response()

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-5",
        messages=[Message(role="user", content=[TextInput(text="hello")])],
    )
    assert "cache_control" not in captured["body"]["messages"][0]["content"][0]


def test_anthropic_1h_ttl_maps_to_the_1h_string():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _anthropic_response()

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-5",
        messages=[
            Message(
                role="user",
                content=[TextInput(text="hello", cacheable=True, cache_ttl_seconds=3600)],
            ),
        ],
    )
    block = captured["body"]["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_refuses_an_unsupported_ttl_before_any_network_call():
    def handler(request):
        raise AssertionError("must not reach the network")

    client = make_client("anthropic", handler)
    with pytest.raises(KeyCallError) as exc:
        client.generate_text(
            model="claude-opus-5",
            messages=[
                Message(
                    role="user",
                    content=[TextInput(text="hello", cacheable=True, cache_ttl_seconds=120)],
                ),
            ],
        )
    assert exc.value.code == ErrorCode.UNSUPPORTED_OPERATION
    assert "300" in exc.value.message and "3600" in exc.value.message


def test_anthropic_system_prompt_stays_a_plain_string_without_a_marker():
    """The plain-string shape is preserved when nothing is cache-marked, so
    an existing caller's request body is byte-for-byte unchanged."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _anthropic_response()

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-5",
        messages=[
            Message(role="system", content=[TextInput(text="be concise")]),
            Message(role="user", content=[TextInput(text="hi")]),
        ],
    )
    assert captured["body"]["system"] == "be concise"


def test_anthropic_system_prompt_becomes_a_block_array_when_cache_marked():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _anthropic_response()

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-5",
        messages=[
            Message(
                role="system",
                content=[
                    TextInput(text="a big stable prefix", cacheable=True),
                    TextInput(text="a second system block"),
                ],
            ),
            Message(role="user", content=[TextInput(text="hi")]),
        ],
    )
    system = captured["body"]["system"]
    assert system == [
        {
            "type": "text",
            "text": "a big stable prefix",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        },
        {"type": "text", "text": "a second system block"},
    ]


def test_anthropic_reports_cache_creation_and_read_via_usage():
    def handler(request):
        return _anthropic_response(cache_creation=500, cache_read=0)

    client = make_client("anthropic", handler)
    result = client.generate_text(
        model="claude-opus-5",
        messages=[Message(role="user", content=[TextInput(text="hi", cacheable=True)])],
    )
    # cache_creation_input_tokens has no dedicated Usage field (it isn't a
    # discount, it's the write cost); cache_read_input_tokens is the field
    # that answers "did caching pay off", and that's what's normalized.
    assert result.usage.cached_input_tokens == 0

    def handler_hit(request):
        return _anthropic_response(cache_creation=0, cache_read=500)

    result_hit = make_client("anthropic", handler_hit).generate_text(
        model="claude-opus-5",
        messages=[Message(role="user", content=[TextInput(text="hi", cacheable=True)])],
    )
    assert result_hit.usage.cached_input_tokens == 500


# --- OpenAI: caching already automatic; the marker is an optional aid -------


def test_openai_sets_the_explicit_breakpoint_on_a_marked_block():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _openai_response()

    client = make_client("openai", handler)
    client.generate_text(
        model="gpt-5.6",
        messages=[Message(role="user", content=[TextInput(text="hello", cacheable=True)])],
    )
    body = captured["body"]
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_openai_sends_no_cache_options_at_all_without_a_marker():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _openai_response()

    client = make_client("openai", handler)
    client.generate_text(
        model="gpt-5.6",
        messages=[Message(role="user", content=[TextInput(text="hello")])],
    )
    body = captured["body"]
    assert "prompt_cache_options" not in body
    assert "prompt_cache_breakpoint" not in body["input"][0]["content"][0]


def test_openai_cache_ttl_seconds_has_no_effect_and_is_never_sent():
    """OpenAI has no TTL concept for this; the field is a harmless no-op
    there rather than a refusal, matching the partial-support pattern."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _openai_response()

    client = make_client("openai", handler)
    client.generate_text(
        model="gpt-5.6",
        messages=[
            Message(
                role="user",
                content=[TextInput(text="hello", cacheable=True, cache_ttl_seconds=3600)],
            ),
        ],
    )
    assert "ttl" not in json.dumps(captured["body"]).lower().replace("throttle", "")


def test_openai_reports_cached_input_tokens():
    client = make_client("openai", lambda request: _openai_response(cached_tokens=8))
    result = client.generate_text(
        model="gpt-5.6",
        messages=[Message(role="user", content=[TextInput(text="hi", cacheable=True)])],
    )
    assert result.usage.cached_input_tokens == 8


# --- Every other provider: silent no-op, unchanged wire shape ---------------


@pytest.mark.parametrize("provider", ["deepseek", "moonshot", "perplexity", "xai"])
def test_the_marker_is_a_silent_no_op_on_every_openai_compatible_provider(provider):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    client = make_client(provider, handler)
    client.generate_text(
        model="some-model",
        messages=[
            Message(role="user", content=[TextInput(text="hello", cacheable=True)]),
        ],
    )
    body = captured["body"]
    assert "cache_control" not in json.dumps(body)
    assert "prompt_cache" not in json.dumps(body)


def test_the_marker_is_a_silent_no_op_on_gemini():
    """Gemini's own adapter (not the OpenAI-compatible family) reads only
    part.text, so this gets its own explicit check."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            },
        )

    client = make_client("gemini", handler)
    client.generate_text(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content=[TextInput(text="hello", cacheable=True)])],
    )
    assert "cachedContent" not in json.dumps(captured["body"])
