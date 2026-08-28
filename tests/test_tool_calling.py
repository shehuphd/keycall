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


# --- apply_patch: a provider-owned tool, not a caller-defined one -----------
#
# Fixture responses mirror the live round captured 2026-08-22
# (project/tool-calling-probe-findings.md): OpenAI only, no caller-supplied
# schema, ToolCall/ToolResult with name "apply_patch" carry the fixed
# operation form instead of arbitrary caller JSON.

APPLY_PATCH_CREATE_RESPONSE = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        {
            "id": "apc_1",
            "type": "apply_patch_call",
            "status": "completed",
            "call_id": "call_p1",
            "operation": {
                "type": "create_file",
                "diff": '+print("hello")\n',
                "path": "hello.py",
            },
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}

APPLY_PATCH_DELETE_RESPONSE = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        {
            "id": "apc_2",
            "type": "apply_patch_call",
            "status": "completed",
            "call_id": "call_p2",
            "operation": {"type": "delete_file", "path": "scratch.tmp"},
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


def test_openai_apply_patch_tool_appended():
    handler, captured = capture(APPLY_PATCH_CREATE_RESPONSE)
    make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=[user()], apply_patch=True
    )
    assert {"type": "apply_patch"} in captured["body"]["tools"]


def test_openai_apply_patch_create_call_parsed():
    handler, _ = capture(APPLY_PATCH_CREATE_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=[user()], apply_patch=True
    )
    (call,) = result.tool_calls
    assert call.id == "call_p1"
    assert call.name == "apply_patch"
    assert call.arguments == {
        "type": "create_file",
        "diff": '+print("hello")\n',
        "path": "hello.py",
    }
    assert json.loads(call.opaque) == {"id": "apc_1"}


def test_openai_apply_patch_delete_call_has_no_diff():
    handler, _ = capture(APPLY_PATCH_DELETE_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=[user()], apply_patch=True
    )
    (call,) = result.tool_calls
    assert call.arguments == {"type": "delete_file", "path": "scratch.tmp"}
    assert "diff" not in call.arguments


def test_openai_apply_patch_replay_shapes():
    handler, captured = capture(APPLY_PATCH_CREATE_RESPONSE)
    call = ToolCall(
        id="call_p1",
        name="apply_patch",
        arguments={"type": "create_file", "diff": '+print("hello")\n', "path": "hello.py"},
        opaque=json.dumps({"id": "apc_1"}),
    )
    make_client("openai", handler).generate_text(
        model="gpt-5.1",
        messages=[
            user(),
            Message(role="assistant", content=[call]),
            Message(role="user", content=[
                ToolResult(tool_call_id=call.id, name=call.name,
                           content={"status": "failed", "output": "disk full"}),
            ]),
        ],
        apply_patch=True,
    )
    items = captured["body"]["input"]
    kinds = [item.get("type") or item.get("role") for item in items]
    assert kinds == ["user", "apply_patch_call", "apply_patch_call_output"]
    assert items[1]["call_id"] == "call_p1"
    assert items[1]["id"] == "apc_1"  # echoed provider item id
    assert items[1]["status"] == "completed"
    assert items[1]["operation"] == {
        "type": "create_file", "diff": '+print("hello")\n', "path": "hello.py",
    }
    assert items[2] == {
        "type": "apply_patch_call_output",
        "call_id": "call_p1",
        "status": "failed",
        "output": "disk full",
    }


def test_openai_apply_patch_replay_defaults_status_for_plain_string_content():
    handler, captured = capture(APPLY_PATCH_CREATE_RESPONSE)
    call = ToolCall(
        id="call_p1", name="apply_patch",
        arguments={"type": "create_file", "diff": "+x\n", "path": "x.py"},
    )
    make_client("openai", handler).generate_text(
        model="gpt-5.1",
        messages=[
            user(),
            Message(role="assistant", content=[call]),
            Message(role="user", content=[
                ToolResult(tool_call_id=call.id, name=call.name, content="patched"),
            ]),
        ],
        apply_patch=True,
    )
    assert captured["body"]["input"][-1] == {
        "type": "apply_patch_call_output",
        "call_id": "call_p1",
        "status": "completed",
        "output": "patched",
    }


def test_apply_patch_gated_before_network():
    handler, captured = capture({})
    with pytest.raises(KeyCallError) as excinfo:
        make_client("anthropic", handler).generate_text(
            model="claude-opus-5", messages=[user()], apply_patch=True
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured


def test_apply_patch_reserved_name_collision_gated_before_network():
    handler, captured = capture({})
    collide = Tool(name="apply_patch", description="x", input_schema={"type": "object"})
    with pytest.raises(KeyCallError) as excinfo:
        make_client("openai", handler).generate_text(
            model="gpt-5.1", messages=[user()], apply_patch=True, tools=[collide]
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured


# --- custom tool: a caller-defined tool with no JSON Schema (OpenAI only) --
#
# Fixture responses mirror the live round captured 2026-08-22: input_schema
# is None, the model's call arrives as a plain string instead of parsed
# arguments, carried in ToolCall.arguments["input"].

WRITE_POEM = Tool(name="write_poem", description="Records a poem", input_schema=None)

CUSTOM_TOOL_RESPONSE = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        {
            "id": "ctc_1",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "call_c1",
            "name": "write_poem",
            "input": "Roses are red.",
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


def test_openai_custom_tool_appended_without_schema():
    handler, captured = capture(CUSTOM_TOOL_RESPONSE)
    make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=[user()], tools=[WRITE_POEM]
    )
    assert {"type": "custom", "name": "write_poem", "description": "Records a poem"} in (
        captured["body"]["tools"]
    )


def test_openai_custom_tool_call_parsed_as_plain_string():
    handler, _ = capture(CUSTOM_TOOL_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=[user()], tools=[WRITE_POEM]
    )
    (call,) = result.tool_calls
    assert call.id == "call_c1"
    assert call.name == "write_poem"
    assert call.arguments == {"input": "Roses are red."}


def test_openai_custom_tool_replay_shapes():
    handler, captured = capture(CUSTOM_TOOL_RESPONSE)
    call = ToolCall(id="call_c1", name="write_poem", arguments={"input": "Roses are red."})
    make_client("openai", handler).generate_text(
        model="gpt-5.1",
        messages=[
            user(),
            Message(role="assistant", content=[call]),
            Message(role="user", content=[
                ToolResult(tool_call_id=call.id, name=call.name, content="Recorded."),
            ]),
        ],
        tools=[WRITE_POEM],
    )
    items = captured["body"]["input"]
    kinds = [item.get("type") or item.get("role") for item in items]
    assert kinds == ["user", "custom_tool_call", "custom_tool_call_output"]
    assert items[1] == {
        "type": "custom_tool_call",
        "call_id": "call_c1",
        "name": "write_poem",
        "input": "Roses are red.",
    }
    assert items[2] == {
        "type": "custom_tool_call_output",
        "call_id": "call_c1",
        "output": "Recorded.",
    }


def test_custom_tool_rejected_for_providers_without_it():
    handler, captured = capture({})
    with pytest.raises(KeyCallError) as excinfo:
        make_client("anthropic", handler).generate_text(
            model="claude-opus-5", messages=[user()], tools=[WRITE_POEM]
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "custom" in excinfo.value.message
    assert "count" not in captured


def test_tool_still_requires_a_json_schema_when_declared():
    with pytest.raises(ValueError, match="JSON Schema"):
        Tool(name="bad", description="x", input_schema={"no_type": "here"})


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


REASONING_ITEM = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": "gAAAAAB-opaque-blob",
}

OPENAI_REASONING_CALL_RESPONSE = {
    "model": "gpt-5.3-chat-latest",
    "status": "completed",
    "output": [
        REASONING_ITEM,
        {
            "id": "fc_abc",
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"London"}',
        },
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


def test_openai_reasoning_item_travels_with_the_call():
    """Replaying a function_call without the reasoning item that preceded
    it is a 400 on reasoning models (verified live 2026-08-09, 3/3)."""
    handler, captured = capture(OPENAI_REASONING_CALL_RESPONSE)
    client = make_client("openai", handler)
    result = client.generate_text(model="gpt-5.3-chat-latest", messages=[user()], tools=[WEATHER])

    echo = json.loads(result.tool_calls[0].opaque)
    assert echo["id"] == "fc_abc"
    assert echo["reasoning"] == REASONING_ITEM

    client.generate_text(
        model="gpt-5.3-chat-latest",
        messages=[
            user(),
            result.to_assistant_message(),
            Message(
                role="user",
                content=[ToolResult(tool_call_id="call_1", name="get_weather", content="14C")],
            ),
        ],
        tools=[WEATHER],
    )
    items = captured["body"]["input"]
    kinds = [item.get("type") for item in items]
    assert "reasoning" in kinds, "the reasoning item must be replayed"
    # It has to arrive before the call it belongs to, and unmodified.
    assert kinds.index("reasoning") < kinds.index("function_call")
    assert items[kinds.index("reasoning")] == REASONING_ITEM
    # The echo blob itself must not leak into the function_call item.
    call_item = items[kinds.index("function_call")]
    assert "reasoning" not in call_item


def test_openai_parallel_calls_replay_one_shared_reasoning_item():
    payload = json.loads(json.dumps(OPENAI_REASONING_CALL_RESPONSE))
    payload["output"].append(
        {
            "id": "fc_def",
            "type": "function_call",
            "call_id": "call_2",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    )
    handler, captured = capture(payload)
    client = make_client("openai", handler)
    result = client.generate_text(model="gpt-5.3-chat-latest", messages=[user()], tools=[WEATHER])
    assert len(result.tool_calls) == 2

    client.generate_text(
        model="gpt-5.3-chat-latest",
        messages=[user(), result.to_assistant_message()],
        tools=[WEATHER],
    )
    kinds = [item.get("type") for item in captured["body"]["input"]]
    assert kinds.count("reasoning") == 1, "one shared item, replayed once"
    assert kinds.count("function_call") == 2


def test_openai_call_without_reasoning_replays_unchanged():
    """Reasoning items appear only when the model reasons, so the ordinary
    shape must stay exactly as it was."""
    handler, captured = capture(OPENAI_CALL_RESPONSE)
    client = make_client("openai", handler)
    result = client.generate_text(model="gpt-4o-mini", messages=[user()], tools=[WEATHER])
    assert json.loads(result.tool_calls[0].opaque) == {"id": "fc_abc"}

    client.generate_text(
        model="gpt-4o-mini",
        messages=[user(), result.to_assistant_message()],
        tools=[WEATHER],
    )
    kinds = [item.get("type") for item in captured["body"]["input"]]
    assert "reasoning" not in kinds
    assert kinds.count("function_call") == 1


# --- tool search: defer_loading, request-size optimization only -----------
#
# Fixture responses mirror the live rounds captured 2026-08-22: a deferred
# tool's discovered call is an ordinary function_call/tool_use, identical to
# a non-deferred one — tool_search_call/tool_search_output and
# server_tool_use/tool_search_tool_result are traces of server-side work,
# never output content.

DEFERRED_WEATHER = Tool(
    name="get_weather",
    description="Get the weather at a specific location",
    input_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    defer_loading=True,
)

OPENAI_TOOL_SEARCH_RESPONSE = {
    "model": "gpt-5.4",
    "status": "completed",
    "output": [
        {"id": "tsc_1", "type": "tool_search_call", "status": "completed",
         "arguments": {"paths": ["get_weather"]}, "call_id": None, "execution": "server"},
        {"id": "tso_1", "type": "tool_search_output", "status": "completed",
         "call_id": None, "execution": "server", "tools": []},
        {"id": "fc_1", "type": "function_call", "status": "completed",
         "arguments": '{"location":"San Francisco"}', "call_id": "call_ts1",
         "name": "get_weather", "namespace": "get_weather"},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


def test_openai_tool_search_appended_when_any_tool_defers():
    handler, captured = capture(OPENAI_TOOL_SEARCH_RESPONSE)
    make_client("openai", handler).generate_text(
        model="gpt-5.4", messages=[user()], tools=[DEFERRED_WEATHER]
    )
    assert {"type": "tool_search"} in captured["body"]["tools"]
    (function_tool,) = [t for t in captured["body"]["tools"] if t.get("type") == "function"]
    assert function_tool["defer_loading"] is True


def test_openai_tool_search_traces_skipped_call_parsed_normally():
    handler, _ = capture(OPENAI_TOOL_SEARCH_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.4", messages=[user()], tools=[DEFERRED_WEATHER]
    )
    (call,) = result.tool_calls
    assert call.id == "call_ts1"
    assert call.name == "get_weather"
    assert call.arguments == {"location": "San Francisco"}
    assert not [p for p in result.parts if p.kind == "unknown"]


def test_openai_tool_search_reply_is_ordinary_function_call_output():
    handler, captured = capture(OPENAI_TOOL_SEARCH_RESPONSE)
    call = ToolCall(id="call_ts1", name="get_weather", arguments={"location": "San Francisco"})
    make_client("openai", handler).generate_text(
        model="gpt-5.4",
        messages=[
            user(),
            Message(role="assistant", content=[call]),
            Message(role="user", content=[
                ToolResult(tool_call_id=call.id, name=call.name, content="68F, sunny"),
            ]),
        ],
        tools=[DEFERRED_WEATHER],
    )
    items = captured["body"]["input"]
    kinds = [item.get("type") or item.get("role") for item in items]
    assert kinds == ["user", "function_call", "function_call_output"]
    assert "defer_loading" not in items[1]
    assert "namespace" not in items[1]


def test_untouched_tools_do_not_trigger_tool_search():
    handler, captured = capture(OPENAI_TOOL_SEARCH_RESPONSE)
    make_client("openai", handler).generate_text(
        model="gpt-5.4", messages=[user()], tools=[WEATHER]
    )
    assert {"type": "tool_search"} not in captured["body"]["tools"]


ANTHROPIC_TOOL_SEARCH_RESPONSE = {
    "model": "claude-opus-5",
    "content": [
        {"type": "text", "text": "Searching for a weather tool."},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search_tool_bm25",
         "input": {"query": "weather forecast"}},
        {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1",
         "content": {"type": "tool_search_tool_search_result",
                     "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}]}},
        {"type": "tool_use", "id": "toolu_ts1", "name": "get_weather",
         "input": {"location": "San Francisco"}},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def test_anthropic_tool_search_appended_when_any_tool_defers():
    handler, captured = capture(ANTHROPIC_TOOL_SEARCH_RESPONSE)
    make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=[user()], tools=[DEFERRED_WEATHER]
    )
    assert {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"} in (
        captured["body"]["tools"]
    )
    (function_tool,) = [t for t in captured["body"]["tools"] if "input_schema" in t]
    assert function_tool["defer_loading"] is True


def test_anthropic_tool_search_traces_skipped_call_parsed_normally():
    handler, _ = capture(ANTHROPIC_TOOL_SEARCH_RESPONSE)
    result = make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=[user()], tools=[DEFERRED_WEATHER]
    )
    (call,) = result.tool_calls
    assert call.id == "toolu_ts1"
    assert call.name == "get_weather"
    assert result.text == "Searching for a weather tool."
    assert not [p for p in result.parts if p.kind == "unknown"]


def test_tool_search_rejected_for_providers_without_it():
    handler, captured = capture({})
    with pytest.raises(KeyCallError) as excinfo:
        make_client("gemini", handler).generate_text(
            model="gemini-flash-latest", messages=[user()], tools=[DEFERRED_WEATHER]
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured
