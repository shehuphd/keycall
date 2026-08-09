"""Streamed tool calls: per-protocol assembly, gates, and parity with the
non-streamed path.

Fixture streams mirror the shapes captured live 2026-08-08: OpenAI splits
arguments across function_call_arguments deltas, Anthropic across
input_json_delta on a tool_use block, the compat family across index-keyed
delta.tool_calls entries, and Gemini sends each call whole.
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
    ToolCallArgumentsDelta,
    ToolCallComplete,
    ToolCallStarted,
    ToolResult,
)

CANARY = "sk-canary-streaming-tools"

WEATHER = Tool(
    name="get_weather",
    description="Get current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def sse(*events: tuple[str | None, dict | str]) -> bytes:
    lines = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        payload = data if isinstance(data, str) else json.dumps(data)
        lines.append(f"data: {payload}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider,
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def messages():
    return [Message(role="user", content=[TextInput(text="Weather in London?")])]


def stream_response(body: bytes):
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def serve(body: bytes):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["count"] = captured.get("count", 0) + 1
        return stream_response(body)

    return handler, captured


def kinds(events):
    return [event.kind for event in events]


# --- fixture streams --------------------------------------------------------


def openai_tool_stream(*, arguments=('{"city":', '"London"}'), terminal=True):
    events = [
        (None, {"type": "response.created", "response": {"model": "gpt-4o-mini"}}),
        (
            None,
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_abc",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                },
            },
        ),
    ]
    for fragment in arguments:
        events.append(
            (
                None,
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_abc",
                    "delta": fragment,
                },
            )
        )
    events.append(
        (
            None,
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_abc",
                "arguments": "".join(arguments),
            },
        )
    )
    if terminal:
        events.append(
            (
                None,
                {
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-4o-mini",
                        "status": "completed",
                        "output": [
                            {
                                "id": "fc_abc",
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "get_weather",
                                "arguments": "".join(arguments),
                            }
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                    },
                },
            )
        )
    return sse(*events)


def anthropic_tool_stream(*, fragments=('{"city":', ' "London"}'), terminal=True):
    events = [
        (
            "message_start",
            {"type": "message_start", "message": {"model": "claude-opus-5", "usage": {}}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
            },
        ),
    ]
    for fragment in fragments:
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
        )
    events.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9},
            },
        )
    )
    if terminal:
        events.append(("message_stop", {"type": "message_stop"}))
    return sse(*events)


def gemini_tool_stream(*, signature="SIG=="):
    call_part = {
        "functionCall": {"name": "get_weather", "args": {"city": "London"}, "id": "gcall_1"},
        "thoughtSignature": signature,
    }
    return sse(
        (
            None,
            {
                "modelVersion": "gemini-2.5-flash",
                "candidates": [{"content": {"parts": [call_part]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            },
        )
    )


def compat_tool_stream(*, finish="tool_calls", done=True):
    events = [
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                },
                            ]
                        }
                    }
                ],
            },
        ),
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city":'}},
                                {"index": 1, "function": {"arguments": '{"city":'}},
                            ]
                        }
                    }
                ],
            },
        ),
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"London"}'}},
                                {"index": 1, "function": {"arguments": '"Paris"}'}},
                            ]
                        }
                    }
                ],
            },
        ),
    ]
    events.append(
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [{"delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )
    )
    if done:
        events.append((None, "[DONE]"))
    return sse(*events)


# --- gates and hostile inputs (first) ---------------------------------------


def test_perplexity_tools_still_gated_before_streaming():
    handler, captured = serve(sse((None, "[DONE]")))
    client = make_client("perplexity", handler)

    with pytest.raises(KeyCallError) as excinfo:
        client.stream_text(model="sonar", messages=messages(), tools=[WEATHER]).__enter__()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured, "gate must fire before the network call"


def test_anthropic_tools_with_schema_still_gated_streaming():
    handler, captured = serve(anthropic_tool_stream())
    client = make_client("anthropic", handler)

    with pytest.raises(KeyCallError) as excinfo:
        client.stream_text(
            model="claude-opus-5",
            messages=messages(),
            tools=[WEATHER],
            response_schema={"type": "object"},
        ).__enter__()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "count" not in captured


def test_malformed_streamed_arguments_is_typed_error():
    handler, _ = serve(anthropic_tool_stream(fragments=("{not json",)))

    with (
        pytest.raises(KeyCallError) as excinfo,
        make_client("anthropic", handler).stream_text(
            model="claude-opus-5", messages=messages(), tools=[WEATHER]
        ) as stream,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_result_unavailable_after_malformed_arguments():
    handler, _ = serve(anthropic_tool_stream(fragments=("{not json",)))

    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages(), tools=[WEATHER]
    ) as stream:
        with pytest.raises(KeyCallError):
            list(stream)
        with pytest.raises(KeyCallError) as excinfo:
            stream.result()
    assert excinfo.value.code is ErrorCode.NETWORK_ERROR


def test_stream_truncated_mid_tool_call_reports_no_result():
    """A call whose arguments never finished must not surface as a complete
    call; the truncation is the error."""
    handler, _ = serve(anthropic_tool_stream(terminal=False))

    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages(), tools=[WEATHER]
    ) as stream:
        with pytest.raises(KeyCallError) as excinfo:
            list(stream)
        assert excinfo.value.code is ErrorCode.NETWORK_ERROR
        with pytest.raises(KeyCallError):
            stream.result()


def test_tool_call_with_no_argument_fragments_yields_empty_arguments():
    """A no-argument tool sends a start and a stop with nothing between."""
    handler, _ = serve(anthropic_tool_stream(fragments=()))

    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()
    assert kinds(events) == [
        "stream_start",
        "tool_call_started",
        "tool_call_complete",
        "stream_finish",
    ]
    assert result.tool_calls[0].arguments == {}


# --- per-protocol assembly --------------------------------------------------


def test_openai_streams_started_deltas_and_complete():
    handler, captured = serve(openai_tool_stream())

    with make_client("openai", handler).stream_text(
        model="gpt-4o-mini", messages=messages(), tools=[WEATHER], tool_choice="auto"
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert kinds(events) == [
        "stream_start",
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_arguments_delta",
        "tool_call_complete",
        "stream_finish",
    ]
    started = next(e for e in events if isinstance(e, ToolCallStarted))
    assert (started.id, started.name) == ("call_1", "get_weather")
    fragments = [e.fragment for e in events if isinstance(e, ToolCallArgumentsDelta)]
    assert fragments == ['{"city":', '"London"}']
    assert json.loads("".join(fragments)) == {"city": "London"}
    complete = next(e for e in events if isinstance(e, ToolCallComplete))
    assert complete.tool_call.arguments == {"city": "London"}
    assert result.tool_calls[0].id == "call_1"
    assert captured["body"]["stream"] is True
    assert captured["body"]["tools"][0]["name"] == "get_weather"


def test_anthropic_streams_tool_use_block():
    handler, _ = serve(anthropic_tool_stream())

    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert kinds(events) == [
        "stream_start",
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_arguments_delta",
        "tool_call_complete",
        "stream_finish",
    ]
    call = result.tool_calls[0]
    assert (call.id, call.name, dict(call.arguments)) == (
        "toolu_1",
        "get_weather",
        {"city": "London"},
    )
    assert result.finish_reason == "tool_use"


def test_gemini_whole_call_emits_no_argument_deltas():
    handler, _ = serve(gemini_tool_stream())

    with make_client("gemini", handler).stream_text(
        model="gemini-2.5-flash", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert kinds(events) == [
        "stream_start",
        "tool_call_started",
        "tool_call_complete",
        "stream_finish",
    ]
    call = result.tool_calls[0]
    assert dict(call.arguments) == {"city": "London"}
    assert json.loads(call.opaque)["thoughtSignature"] == "SIG=="


def test_gemini_streamed_call_replays_with_signature():
    """The echo data has to survive the streaming path too, or the next
    request 400s."""
    handler, _ = serve(gemini_tool_stream())
    with make_client("gemini", handler).stream_text(
        model="gemini-2.5-flash", messages=messages(), tools=[WEATHER]
    ) as stream:
        list(stream)
        result = stream.result()

    replay_handler, captured = serve(gemini_tool_stream())
    call = result.tool_calls[0]
    with make_client("gemini", replay_handler).stream_text(
        model="gemini-2.5-flash",
        messages=[
            *messages(),
            result.to_assistant_message(),
            Message(
                role="user",
                content=[
                    ToolResult(tool_call_id=call.id, name=call.name, content={"c": 12})
                ],
            ),
        ],
        tools=[WEATHER],
    ) as stream:
        list(stream)

    model_turn = captured["body"]["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["thoughtSignature"] == "SIG=="


def test_compat_parallel_calls_assemble_by_index():
    handler, _ = serve(compat_tool_stream())

    with make_client("deepseek", handler).stream_text(
        model="deepseek-chat", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    assert len(completes) == 2
    assert [c.tool_call.id for c in completes] == ["call_a", "call_b"]
    assert [dict(c.tool_call.arguments) for c in completes] == [
        {"city": "London"},
        {"city": "Paris"},
    ]
    # Both calls close before the stream does.
    assert kinds(events)[-1] == "stream_finish"
    assert kinds(events).index("tool_call_complete") < len(kinds(events)) - 1
    assert len(result.tool_calls) == 2
    assert result.usage.total_tokens == 9


def test_compat_calls_close_even_without_finish_reason():
    """Defensive: a target that goes straight to [DONE] must still produce
    complete calls rather than dropping them."""
    handler, _ = serve(compat_tool_stream(finish=None))

    with make_client("deepseek", handler).stream_text(
        model="deepseek-chat", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert len([e for e in events if isinstance(e, ToolCallComplete)]) == 2
    assert len(result.tool_calls) == 2


# --- parity with the non-streamed path --------------------------------------


def test_streamed_and_non_streamed_tool_calls_match():
    """The whole point of the design: the same request answered either way
    gives the caller the same calls to dispatch."""
    stream_handler, _ = serve(anthropic_tool_stream())
    with make_client("anthropic", stream_handler).stream_text(
        model="claude-opus-5", messages=messages(), tools=[WEATHER]
    ) as stream:
        list(stream)
        streamed = stream.result()

    non_streamed_body = {
        "model": "claude-opus-5",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "London"},
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 9},
    }
    client = make_client("anthropic", lambda r: httpx.Response(200, json=non_streamed_body))
    plain = client.generate_text(model="claude-opus-5", messages=messages(), tools=[WEATHER])

    assert streamed.tool_calls == plain.tool_calls
    assert streamed.finish_reason == plain.finish_reason


def test_streamed_result_replays_through_to_assistant_message():
    handler, _ = serve(compat_tool_stream())
    with make_client("deepseek", handler).stream_text(
        model="deepseek-chat", messages=messages(), tools=[WEATHER]
    ) as stream:
        list(stream)
        result = stream.result()

    replay_handler, captured = serve(compat_tool_stream())
    with make_client("deepseek", replay_handler).stream_text(
        model="deepseek-chat",
        messages=[
            *messages(),
            result.to_assistant_message(),
            Message(
                role="user",
                content=[
                    ToolResult(tool_call_id=c.id, name=c.name, content="12C")
                    for c in result.tool_calls
                ],
            ),
        ],
        tools=[WEATHER],
    ) as stream:
        list(stream)

    sent = captured["body"]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_a", "call_b"]
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["call_a", "call_b"]


def test_text_and_tool_calls_stream_together():
    """Compat providers emit text alongside calls; both must survive."""
    body = sse(
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [{"delta": {"content": "Checking. "}}],
            },
        ),
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"London"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
        ),
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
        (None, "[DONE]"),
    )
    handler, _ = serve(body)

    with make_client("deepseek", handler).stream_text(
        model="deepseek-chat", messages=messages(), tools=[WEATHER]
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert kinds(events) == [
        "stream_start",
        "text_delta",
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_complete",
        "stream_finish",
    ]
    assert result.text == "Checking. "
    assert len(result.tool_calls) == 1


def test_custom_target_streaming_warns_tools_unverified():
    handler, _ = serve(compat_tool_stream())
    client = make_client(
        "custom",
        handler,
        base_url="https://tools.example.com/v1",
        protocol="openai-compatible",
    )

    with client.stream_text(model="m", messages=messages(), tools=[WEATHER]) as stream:
        list(stream)
        result = stream.result()

    assert any("unverified" in w for w in result.warnings)


def test_streaming_without_tools_emits_no_tool_events():
    """Mutation guard: the tool paths must stay inert on ordinary streams."""
    body = sse(
        (None, {"model": "deepseek-chat", "choices": [{"delta": {"content": "hi"}}]}),
        (
            None,
            {
                "model": "deepseek-chat",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
        (None, "[DONE]"),
    )
    handler, _ = serve(body)

    with make_client("deepseek", handler).stream_text(
        model="deepseek-chat", messages=messages()
    ) as stream:
        events = list(stream)
        result = stream.result()

    assert kinds(events) == ["stream_start", "text_delta", "stream_finish"]
    assert result.tool_calls == ()
    assert result.text == "hi"
