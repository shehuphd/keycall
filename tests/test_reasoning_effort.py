"""reasoning_effort mapping and gating, per provider.

The field is only offered where a native control was live-verified to
bind (2026-08-14): reasoning-token counts follow the requested level.
Providers that accept the parameter without honoring it (DeepSeek) are
refused rather than silently ignored.
"""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-effort-key"


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


def simple_messages():
    return [Message(role="user", content=[TextInput(text="Why is the sky blue?")])]


# --- request-side gating ----------------------------------------------------


@pytest.mark.parametrize("provider", ["deepseek", "moonshot"])
def test_reasoning_effort_refused_where_no_binding_control_exists(provider):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client(provider, handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="some-model", messages=simple_messages(), reasoning_effort="low"
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "reasoning" in excinfo.value.message
    # The error names every provider where the knob does work.
    for supported in ("openai", "anthropic", "gemini", "perplexity", "xai"):
        assert supported in excinfo.value.message


def test_reasoning_effort_refused_for_custom_targets():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = KeyCall(
        provider="my-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model="m", messages=simple_messages(), reasoning_effort="low")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_absent_effort_adds_no_reasoning_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    client = make_client("openai", handler)
    client.generate_text(model="gpt-5-nano", messages=simple_messages())
    client.close()
    assert "reasoning" not in captured["body"]
    assert "reasoning_effort" not in captured["body"]


# --- per-provider wire shapes -----------------------------------------------


def test_openai_effort_rides_the_responses_reasoning_field():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-nano",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Rayleigh scattering."}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_text(
        model="gpt-5-nano", messages=simple_messages(), reasoning_effort="minimal"
    )
    client.close()
    assert captured["body"]["reasoning"] == {"effort": "minimal"}
    assert result.text == "Rayleigh scattering."


def test_anthropic_effort_rides_output_config():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-4-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Rayleigh scattering."}],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )

    client = make_client("anthropic", handler)
    client.generate_text(
        model="claude-opus-4-5", messages=simple_messages(), reasoning_effort="low"
    )
    client.close()
    assert captured["body"]["output_config"] == {"effort": "low"}
    # A top-level effort field is refused by the API; it must not appear.
    assert "effort" not in captured["body"]
    assert "reasoning_effort" not in captured["body"]


def test_gemini_effort_becomes_an_uppercase_thinking_level():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-flash-latest",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Rayleigh scattering."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4},
            },
        )

    client = make_client("gemini", handler)
    client.generate_text(
        model="gemini-flash-latest", messages=simple_messages(), reasoning_effort="low"
    )
    client.close()
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_perplexity_effort_is_the_flat_field():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Rayleigh scattering."}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    client = make_client("perplexity", handler)
    client.generate_text(
        model="sonar-reasoning-pro",
        messages=simple_messages(),
        reasoning_effort="low",
        max_output_tokens=100,
    )
    client.close()
    assert captured["body"]["reasoning_effort"] == "low"


def test_xai_effort_reroutes_to_the_responses_surface():
    """grok's chat completions answers 200 to reasoning_effort without
    honoring it, while the responses route binds (both measured live
    2026-08-14) — so naming an effort must switch route and wire shape,
    the same detour web_search takes."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "grok-4.6",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Rayleigh scattering."}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
        )

    client = make_client("xai", handler)
    result = client.generate_text(
        model="grok-4.6", messages=simple_messages(), reasoning_effort="low"
    )
    client.close()
    assert captured["path"] == "/v1/responses"
    assert captured["body"]["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in captured["body"]
    assert result.text == "Rayleigh scattering."


def test_xai_effort_streams_through_the_responses_assembler():
    captured = {}
    events = [
        {"type": "response.output_text.delta", "delta": "Rayleigh"},
        {
            "type": "response.completed",
            "response": {
                "model": "grok-4.6",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Rayleigh"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
        },
    ]
    body = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    client = make_client("xai", handler)
    with client.stream_text(
        model="grok-4.6", messages=simple_messages(), reasoning_effort="low"
    ) as stream:
        seen = list(stream)
        result = stream.result()
    client.close()
    assert captured["path"] == "/v1/responses"
    assert captured["body"]["reasoning"] == {"effort": "low"}
    assert any(e.kind == "text_delta" for e in seen)
    assert result.text == "Rayleigh"
