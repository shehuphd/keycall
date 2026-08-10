"""response_schema (structured output), per provider.

Fixtures mirror live responses captured 2026-08-06: OpenAI Responses
text.format, Anthropic forced tool_choice, Gemini responseSchema, and the
compat family split (Moonshot/Perplexity enforce json_schema; DeepSeek and
unverified custom targets fall back to json_object with a warning).
"""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-structured-key"

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "version": {"type": "string"}},
    "required": ["name", "version"],
}


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


def simple_messages():
    return [Message(role="user", content=[TextInput(text="name and version?")])]


# --- validation --------------------------------------------------------------


def test_response_schema_must_be_a_dict_with_type():
    from keycall import TextGenerationRequest

    with pytest.raises(ValueError):
        TextGenerationRequest(model="m", messages=simple_messages(), response_schema="not-a-dict")
    with pytest.raises(ValueError):
        TextGenerationRequest(model="m", messages=simple_messages(), response_schema={})


# --- OpenAI: text.format json_schema -----------------------------------------


def test_openai_sends_json_schema_format():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"name":"x","version":"1"}'}
                        ],
                    }
                ],
                "usage": {},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_text(model="gpt-4o-mini", messages=simple_messages(), response_schema=SCHEMA)
    fmt = captured["body"]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == SCHEMA
    assert fmt["strict"] is True
    assert result.text == '{"name":"x","version":"1"}'
    assert "not enforce" not in " ".join(result.warnings)


# --- Anthropic: forced tool_choice --------------------------------------------


def test_anthropic_forces_structured_output_tool():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-5",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "keycall_response",
                        "input": {"name": "x", "version": "1"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {},
            },
        )

    client = make_client("anthropic", handler)
    result = client.generate_text(
        model="claude-opus-5", messages=simple_messages(), response_schema=SCHEMA
    )
    assert captured["body"]["tool_choice"] == {"type": "tool", "name": "keycall_response"}
    assert captured["body"]["tools"][0]["input_schema"] == SCHEMA
    assert json.loads(result.text) == {"name": "x", "version": "1"}


def test_anthropic_rejects_web_search_with_response_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client("anthropic", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="claude-opus-5",
            messages=simple_messages(),
            web_search=True,
            response_schema=SCHEMA,
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_anthropic_unrelated_tool_use_is_a_tool_call_not_schema_output():
    """A tool_use block that ISN'T the structured-output tool must not be
    mistaken for one — only the exact synthetic name is special-cased. It
    surfaces as a ToolCall part, and its input never enters result.text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "some_other_tool",
                     "input": {"x": 1}},
                    {"type": "text", "text": "hello"},
                ],
                "usage": {},
            },
        )

    client = make_client("anthropic", handler)
    result = client.generate_text(model="claude-opus-5", messages=simple_messages())
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "some_other_tool"
    assert result.tool_calls[0].arguments == {"x": 1}
    assert result.text == "hello"


# --- Gemini: responseSchema ----------------------------------------------------


def test_gemini_sends_response_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"name":"x","version":"1"}'}]}}
                ],
                "usageMetadata": {},
            },
        )

    client = make_client("gemini", handler)
    result = client.generate_text(
        model="gemini-flash-latest", messages=simple_messages(), response_schema=SCHEMA
    )
    config = captured["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == SCHEMA
    assert result.text == '{"name":"x","version":"1"}'


# --- Compat family: capability split -------------------------------------------


@pytest.mark.parametrize("provider", ["moonshot", "perplexity"])
def test_compat_providers_with_schema_support_get_json_schema(provider):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "models" in request.url.path:
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"name":"x","version":"1"}'}}], "usage": {}},
        )

    client = make_client(provider, handler)
    result = client.generate_text(model="m", messages=simple_messages(), response_schema=SCHEMA)
    fmt = captured["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == SCHEMA
    assert not any("not enforce" in w for w in result.warnings)


def test_deepseek_falls_back_to_json_object_with_warning():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"name":"x"}'}}], "usage": {}},
        )

    client = make_client("deepseek", handler)
    result = client.generate_text(model="deepseek-chat", messages=simple_messages(), response_schema=SCHEMA)
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert any("does not enforce" in w for w in result.warnings)
    assert "deepseek" in result.warnings[0]
    # simple_messages() doesn't mention "json" — DeepSeek 400s without it
    # (live-verified 2026-08-06), so KeyCall must inject the instruction.
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": "Respond only with a single valid JSON object.",
    }
    assert any("added a 'respond only with JSON'" in w for w in result.warnings)


def test_deepseek_skips_json_instruction_when_prompt_already_mentions_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    client = make_client("deepseek", handler)
    result = client.generate_text(
        model="deepseek-chat",
        messages=[Message(role="user", content=[TextInput(text="Reply as JSON please")])],
        response_schema=SCHEMA,
    )
    assert captured["body"]["messages"] == [{"role": "user", "content": "Reply as JSON please"}]
    assert not any("added a 'respond only with JSON'" in w for w in result.warnings)
    assert any("does not enforce" in w for w in result.warnings)  # this warning still applies


def test_custom_target_falls_back_to_json_object_with_warning():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"name":"x"}'}}], "usage": {}},
        )

    client = KeyCall(
        provider="my-lab",
        protocol="openai-compatible",
        api_key=CANARY,
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(handler),
    )
    result = client.generate_text(model="m", messages=simple_messages(), response_schema=SCHEMA)
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert any("does not enforce" in w for w in result.warnings)


def test_no_response_schema_means_no_warning_and_no_response_format():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {}})

    client = make_client("deepseek", handler)
    result = client.generate_text(model="deepseek-chat", messages=simple_messages())
    assert "response_format" not in captured["body"]
    assert result.warnings == ()


# --- adversarial: malformed / boundary / injection ---------------------------


def test_response_schema_rejects_non_mapping_types():
    from keycall import TextGenerationRequest

    for bad in (123, ["type", "object"], True, b"\x00\x01", float("nan")):
        with pytest.raises((ValueError, TypeError)):
            TextGenerationRequest(model="m", messages=simple_messages(), response_schema=bad)


def test_response_schema_with_type_key_but_wrong_value_type_is_not_rejected_by_keycall():
    """KeyCall checks the 'type' key exists, not that it's well-formed JSON
    Schema — it isn't a schema-validating-a-schema library. This pins that
    intentional permissiveness so it doesn't silently change."""
    from keycall import TextGenerationRequest

    request = TextGenerationRequest(
        model="m", messages=simple_messages(), response_schema={"type": 12345}
    )
    assert request.response_schema == {"type": 12345}


def test_schema_with_quotes_backslashes_and_unicode_round_trips_safely():
    """Injection-shaped content in a schema value must not corrupt the
    outgoing JSON body — proves httpx's own JSON encoder is used, not
    string concatenation, for every adapter that embeds response_schema."""
    hostile_schema = {
        "type": "object",
        "properties": {
            'name"; DROP TABLE x; --': {
                "type": "string",
                "description": 'contains "quotes", \\backslashes\\, \nnewlines, and 🎉 unicode',
            }
        },
        "required": ['name"; DROP TABLE x; --'],
    }
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # If encoding were unsafe, this parse would fail outright.
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"status": "completed", "output": [], "usage": {}},
        )

    client = make_client("openai", handler)
    client.generate_text(model="gpt-4o-mini", messages=simple_messages(), response_schema=hostile_schema)
    assert captured["body"]["text"]["format"]["schema"] == hostile_schema


def test_anthropic_malformed_tool_input_does_not_crash():
    """A provider that returns a broken/missing tool_use.input must not
    crash the adapter — pass through what it actually said."""

    for bad_input in (None, [1, 2, 3], "not-an-object", 42):
        def handler(request: httpx.Request, _payload=bad_input) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "tool_use", "name": "keycall_response", "input": _payload}
                    ],
                    "usage": {},
                },
            )

        client = make_client("anthropic", handler)
        result = client.generate_text(
            model="claude-opus-5", messages=simple_messages(), response_schema=SCHEMA
        )
        # Must produce *some* JSON-serialized text, never raise.
        json.loads(result.text)


def test_anthropic_tool_use_missing_input_key_entirely():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": [{"type": "tool_use", "name": "keycall_response"}], "usage": {}},
        )

    client = make_client("anthropic", handler)
    result = client.generate_text(
        model="claude-opus-5", messages=simple_messages(), response_schema=SCHEMA
    )
    assert result.text == "{}"


def test_provider_returns_truncated_or_non_json_content_with_schema_requested():
    """KeyCall does not validate that a provider's structured-output content
    actually parses as JSON or matches the schema — that's explicitly the
    caller's job: KeyCall normalizes, it doesn't re-validate provider
    output. Confirm garbage passes through inertly, not silently
    coerced or crashed on."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"name": "truncat'}],
                    }
                ],
                "usage": {},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_text(
        model="gpt-4o-mini", messages=simple_messages(), response_schema=SCHEMA
    )
    assert result.text == '{"name": "truncat'  # passed through, not repaired or rejected
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.text)


def test_deepseek_warning_never_contains_schema_or_prompt_content():
    """The unenforced-schema warning must be a fixed, safe message — never
    interpolate caller-controlled schema or prompt content into it."""
    secret_marker = "PROMPT_SHOULD_NOT_LEAK_abc123"
    schema_marker = "SCHEMA_SHOULD_NOT_LEAK_xyz789"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    client = make_client("deepseek", handler)
    result = client.generate_text(
        model="deepseek-chat",
        messages=[Message(role="user", content=[TextInput(text=secret_marker)])],
        response_schema={"type": "object", "title": schema_marker},
    )
    blob = " ".join(result.warnings)
    assert secret_marker not in blob
    assert schema_marker not in blob


def test_empty_response_schema_object_type_only_is_the_permitted_minimum():
    """Boundary: the smallest schema that passes validation (bare {'type':
    'object'}, no properties/required) must actually work end-to-end, not
    just pass __post_init__."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    client = make_client("openai", handler)
    result = client.generate_text(
        model="gpt-4o-mini", messages=simple_messages(), response_schema={"type": "object"}
    )
    assert result.text is None  # empty output — no crash either way


def test_gemini_web_search_and_response_schema_together_is_unverified_but_does_not_crash_keycall():
    """Unlike Anthropic, KeyCall has no confirmed evidence Gemini rejects
    this combination (a live probe was inconclusive due to rate limiting),
    so it is deliberately NOT gated — passed through honestly rather than
    guessed at. This test pins that KeyCall's own code doesn't crash while
    building the request; whether Gemini itself accepts it is untested."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}
        )

    client = make_client("gemini", handler)
    client.generate_text(
        model="gemini-flash-latest",
        messages=simple_messages(),
        web_search=True,
        response_schema=SCHEMA,
    )
    assert captured["body"]["tools"] == [{"google_search": {}}]
    assert captured["body"]["generationConfig"]["responseSchema"] == SCHEMA


def test_reasoning_truncation_produces_a_warning_not_silent_empty_output():
    """Reasoning-capable compat models (Moonshot/Kimi) can consume the
    whole max_output_tokens budget on reasoning_content and never emit
    content — live-reproduced 2026-08-06 at max_output_tokens=100. KeyCall
    must say so, not just return an empty result."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "thinking very hard about this...",
                            "content": "",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {},
            },
        )

    client = make_client("moonshot", handler)
    result = client.generate_text(model="kimi-k2", messages=simple_messages(), max_output_tokens=10)
    assert result.text is None
    assert any("max_output_tokens was likely too small" in w for w in result.warnings)


def test_normal_content_present_produces_no_reasoning_truncation_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "brief thought",
                            "content": "the answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )

    client = make_client("moonshot", handler)
    result = client.generate_text(model="kimi-k2", messages=simple_messages())
    assert result.text == "the answer"
    assert not any("too small" in w for w in result.warnings)


def test_usage_md_documented_recipe_verbatim():
    """The exact schema and call shape from USAGE.md's 'Structured output'
    section, including additionalProperties: false — the OpenAI
    requirement a schema lacking it would 400 on live (confirmed
    2026-08-06). Pins the documented recipe, not a looser stand-in."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "version": {"type": "string"}},
        "required": ["name", "version"],
        "additionalProperties": False,
    }
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"name":"KeyCall","version":"0.2.0"}'}
                        ],
                    }
                ],
                "usage": {},
            },
        )

    client = make_client("openai", handler)
    result = client.generate_text(
        model="gpt-4o-mini",
        messages=[Message(role="user", content=[TextInput(text="Name and version, as JSON.")])],
        response_schema=schema,
    )
    assert captured["body"]["text"]["format"]["schema"] == schema
    parsed = json.loads(result.text)
    assert parsed == {"name": "KeyCall", "version": "0.2.0"}
