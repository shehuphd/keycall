"""Tool calling: wire shapes both directions, gates, and hostile inputs.

Fixture shapes mirror the live rounds captured 2026-08-08.
"""

import json

import httpx
import pytest

from keycall import (
    ErrorCode,
    KeyCall,
    KeyCallError,
    Message,
    TextInput,
    Tool,
    ToolCall,
    ToolResult,
)

CANARY = "sk-canary-tools-key"

WEATHER = Tool(
    name="get_weather",
    description="Get current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider,
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def user(text="What's the weather in London?"):
    return Message(role="user", content=[TextInput(text=text)])


def capture(response_json):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["count"] = captured.get("count", 0) + 1
        return httpx.Response(200, json=response_json)

    return handler, captured


OPENAI_CALL_RESPONSE = {
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "id": "fc_abc",
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"London"}',
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}

ANTHROPIC_CALL_RESPONSE = {
    "model": "claude-opus-5",
    "stop_reason": "tool_use",
    "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
         "input": {"city": "London"}, "caller": {"type": "direct"}},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

GEMINI_CALL_RESPONSE = {
    "modelVersion": "gemini-flash-latest",
    "responseId": "r1",
    "candidates": [{
        "content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "London"}, "id": "fc9"},
             "thoughtSignature": "SIG=="},
        ]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
}

COMPAT_CALL_RESPONSE = {
    "model": "deepseek-v4-flash",
    "choices": [{
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            ],
        },
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


# --- request validation (before any network) --------------------------------


def test_tool_requires_schema_with_type():
    with pytest.raises(ValueError):
        Tool(name="x", description="d", input_schema={"properties": {}})
    with pytest.raises(ValueError):
        Tool(name="", description="d", input_schema={"type": "object"})


def test_tool_choice_validation():
    with pytest.raises(ValueError):
        KeyCall(provider="openai", api_key=CANARY).generate_text(
            model="m", messages=[user()], tools=[WEATHER], tool_choice="forced"
        )
    with pytest.raises(ValueError):
        KeyCall(provider="openai", api_key=CANARY).generate_text(
            model="m", messages=[user()], tool_choice="auto"
        )
    with pytest.raises(TypeError):
        KeyCall(provider="openai", api_key=CANARY).generate_text(
            model="m", messages=[user()], tools=[{"name": "not-a-tool"}]
        )


def test_tool_parts_are_role_checked():
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    client = make_client("openai", handler)
    call = ToolCall(id="c1", name="t", arguments={})
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="m", messages=[Message(role="user", content=[call])], tools=[WEATHER]
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    result_part = ToolResult(tool_call_id="c1", name="t", content="x")
    with pytest.raises(KeyCallError):
        client.generate_text(
            model="m",
            messages=[Message(role="assistant", content=[result_part])],
            tools=[WEATHER],
        )
    assert "count" not in captured  # nothing reached the network


def test_perplexity_tools_gated_before_network():
    handler, captured = capture({})
    client = make_client("perplexity", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model="sonar", messages=[user()], tools=[WEATHER])
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured


def test_anthropic_tools_with_schema_gated():
    handler, captured = capture({})
    client = make_client("anthropic", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="claude-opus-5",
            messages=[user()],
            tools=[WEATHER],
            response_schema={"type": "object"},
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured


def test_streaming_with_tools_gated():
    from keycall import TextGenerationRequest
    from keycall._client import TextStream

    client = make_client("openai", lambda r: httpx.Response(500))
    request = TextGenerationRequest(model="m", messages=[user()], tools=[WEATHER])

    with pytest.raises(KeyCallError) as excinfo:
        TextStream(client, request)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


# --- provider payload parsing (hostile first) -------------------------------


def test_malformed_arguments_from_provider_is_typed_error():
    bad = json.loads(json.dumps(OPENAI_CALL_RESPONSE))
    bad["output"][0]["arguments"] = "{not json"
    handler, _ = capture(bad)
    with pytest.raises(KeyCallError) as excinfo:
        make_client("openai", handler).generate_text(
            model="gpt-4o-mini", messages=[user()], tools=[WEATHER]
        )
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_non_object_arguments_is_typed_error():
    bad = json.loads(json.dumps(COMPAT_CALL_RESPONSE))
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '["list"]'
    handler, _ = capture(bad)
    with pytest.raises(KeyCallError) as excinfo:
        make_client("deepseek", handler).generate_text(
            model="deepseek-v4-flash", messages=[user()], tools=[WEATHER]
        )
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_openai_call_parsed_with_echo_id():
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-4o-mini", messages=[user()], tools=[WEATHER]
    )
    assert captured["body"]["tools"][0] == {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": dict(WEATHER.input_schema),
    }
    (call,) = result.tool_calls
    assert call.id == "call_1"
    assert call.arguments == {"city": "London"}
    assert json.loads(call.opaque) == {"id": "fc_abc"}


def test_compat_parallel_calls_parsed():
    handler, _ = capture(COMPAT_CALL_RESPONSE)
    result = make_client("deepseek", handler).generate_text(
        model="deepseek-v4-flash", messages=[user()], tools=[WEATHER]
    )
    assert [c.arguments["city"] for c in result.tool_calls] == ["London", "Paris"]
    assert result.finish_reason == "tool_calls"


def test_gemini_call_parsed_with_signature():
    handler, captured = capture(GEMINI_CALL_RESPONSE)
    result = make_client("gemini", handler).generate_text(
        model="gemini-flash-latest", messages=[user()], tools=[WEATHER]
    )
    assert captured["body"]["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    (call,) = result.tool_calls
    assert call.arguments == {"city": "London"}
    assert json.loads(call.opaque)["thoughtSignature"] == "SIG=="


# --- replay: our history back onto each provider's wire ---------------------


def round_two_messages(call):
    return [
        user(),
        Message(role="assistant", content=[call]),
        Message(role="user", content=[
            ToolResult(tool_call_id=call.id, name=call.name,
                       content=json.dumps({"temp_c": 14})),
        ]),
    ]


def test_openai_replay_shapes():
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "London"},
                    opaque=json.dumps({"id": "fc_abc"}))
    make_client("openai", handler).generate_text(
        model="gpt-4o-mini", messages=round_two_messages(call), tools=[WEATHER]
    )
    items = captured["body"]["input"]
    kinds = [item.get("type") or item.get("role") for item in items]
    assert kinds == ["user", "function_call", "function_call_output"]
    assert items[1]["call_id"] == "call_1"
    assert items[1]["id"] == "fc_abc"  # echoed provider item id
    assert json.loads(items[1]["arguments"]) == {"city": "London"}
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": json.dumps({"temp_c": 14}),
    }


def test_anthropic_replay_shapes():
    handler, captured = capture(ANTHROPIC_CALL_RESPONSE)
    call = ToolCall(id="toolu_1", name="get_weather", arguments={"city": "London"})
    make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=round_two_messages(call), tools=[WEATHER]
    )
    messages = captured["body"]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0] == {
        "type": "tool_use", "id": "toolu_1", "name": "get_weather",
        "input": {"city": "London"},
    }
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert captured["body"]["tools"][0]["input_schema"] == dict(WEATHER.input_schema)


def test_gemini_replay_echoes_thought_signature():
    handler, captured = capture(GEMINI_CALL_RESPONSE)
    call = ToolCall(id="fc9", name="get_weather", arguments={"city": "London"},
                    opaque=json.dumps({"thoughtSignature": "SIG==", "id": "fc9"}))
    make_client("gemini", handler).generate_text(
        model="gemini-flash-latest", messages=round_two_messages(call), tools=[WEATHER]
    )
    contents = captured["body"]["contents"]
    assert contents[1]["role"] == "model"
    wire = contents[1]["parts"][0]
    assert wire["thoughtSignature"] == "SIG=="
    assert wire["functionCall"] == {"name": "get_weather", "args": {"city": "London"}, "id": "fc9"}
    # String JSON content becomes the object Gemini requires.
    assert contents[2]["parts"][0]["functionResponse"] == {
        "name": "get_weather", "response": {"temp_c": 14},
    }


def test_gemini_non_json_result_content_is_wrapped():
    handler, captured = capture(GEMINI_CALL_RESPONSE)
    call = ToolCall(id="fc9", name="get_weather", arguments={})
    make_client("gemini", handler).generate_text(
        model="gemini-flash-latest",
        messages=[
            user(),
            Message(role="assistant", content=[call]),
            Message(role="user", content=[
                ToolResult(tool_call_id="fc9", name="get_weather", content="plain words"),
            ]),
        ],
        tools=[WEATHER],
    )
    assert captured["body"]["contents"][2]["parts"][0]["functionResponse"]["response"] == {
        "output": "plain words"
    }


def test_compat_replay_shapes():
    handler, captured = capture(COMPAT_CALL_RESPONSE)
    call = ToolCall(id="call_a", name="get_weather", arguments={"city": "London"})
    make_client("deepseek", handler).generate_text(
        model="deepseek-v4-flash", messages=round_two_messages(call), tools=[WEATHER]
    )
    messages = captured["body"]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert messages[2] == {
        "role": "tool", "tool_call_id": "call_a", "content": json.dumps({"temp_c": 14}),
    }
    assert captured["body"]["tools"][0]["function"]["name"] == "get_weather"


# --- tool_choice and combos --------------------------------------------------


def test_tool_choice_mappings():
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    make_client("openai", handler).generate_text(
        model="m", messages=[user()], tools=[WEATHER], tool_choice="required"
    )
    assert captured["body"]["tool_choice"] == "required"

    handler, captured = capture(ANTHROPIC_CALL_RESPONSE)
    make_client("anthropic", handler).generate_text(
        model="m", messages=[user()], tools=[WEATHER], tool_choice="required"
    )
    assert captured["body"]["tool_choice"] == {"type": "any"}

    handler, captured = capture(GEMINI_CALL_RESPONSE)
    make_client("gemini", handler).generate_text(
        model="m", messages=[user()], tools=[WEATHER], tool_choice="none"
    )
    assert captured["body"]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}


def test_gemini_tools_plus_web_search_sets_combo_flag():
    handler, captured = capture(GEMINI_CALL_RESPONSE)
    make_client("gemini", handler).generate_text(
        model="m", messages=[user()], tools=[WEATHER], web_search=True
    )
    body = captured["body"]
    assert body["toolConfig"]["includeServerSideToolInvocations"] is True
    assert {"google_search": {}} in body["tools"]
    assert any("functionDeclarations" in entry for entry in body["tools"])


def test_openai_tools_merge_with_web_search():
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    make_client("openai", handler).generate_text(
        model="m", messages=[user()], tools=[WEATHER], web_search=True
    )
    kinds = [entry["type"] for entry in captured["body"]["tools"]]
    assert kinds == ["function", "web_search"]


def test_custom_target_passes_through_with_warning():
    handler, captured = capture(COMPAT_CALL_RESPONSE)
    client = KeyCall(
        provider="my-lab",
        api_key=CANARY,
        protocol="openai-compatible",
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(handler),
    )
    result = client.generate_text(model="some-model", messages=[user()], tools=[WEATHER])
    assert captured["body"]["tools"][0]["function"]["name"] == "get_weather"
    assert any("unverified" in w for w in result.warnings)
    assert len(result.tool_calls) == 2


# --- conveniences ------------------------------------------------------------


def test_to_assistant_message_carries_calls_and_text():
    handler, _ = capture(
        {
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {"type": "message",
                 "content": [{"type": "output_text", "text": "Checking."}]},
                {"id": "fc_abc", "type": "function_call", "call_id": "call_1",
                 "name": "get_weather", "arguments": '{"city":"London"}'},
            ],
            "usage": {},
        }
    )
    result = make_client("openai", handler).generate_text(
        model="gpt-4o-mini", messages=[user()], tools=[WEATHER]
    )
    replay = result.to_assistant_message()
    assert replay.role == "assistant"
    kinds = [type(p).__name__ for p in replay.content]
    assert kinds == ["TextInput", "ToolCall"]
    assert replay.content[1].opaque is not None
