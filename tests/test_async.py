"""Async client parity.

AsyncKeyCall is half the public API and shares the adapters, so the risk is
not that its request building differs — it is that a feature ships verified
on the sync path only and nobody drives the async one. Every capability
added since 0.5.0 (tools, streamed tool events, image input) is exercised
here against the async client specifically.
"""

import base64
import inspect
import json

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    ErrorCode,
    ImageInput,
    KeyCall,
    KeyCallError,
    Message,
    TextInput,
    Tool,
    ToolResult,
)

CANARY = "sk-canary-async-key"

WEATHER = Tool(
    name="get_weather",
    description="Get current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def client(handler, provider="openai", **kwargs):
    return AsyncKeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


def ask(text="hi"):
    return [Message(role="user", content=[TextInput(text=text)])]


def sse(*events) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events)).encode()


def stream_response(body: bytes):
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


OPENAI_TOOL_CALL = {
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


def test_public_surface_matches_between_the_clients():
    """A parameter added to one client and not the other is the drift this
    file exists to catch, and a signature check finds it without a live
    call."""
    for name in (
        "list_models",
        "generate_text",
        "invoke",
        "stream_text",
        "generate_image",
        "generate_speech",
        "embed",
        "start_video",
        "check_video",
        "fetch_video",
        "generate_video",
        "realtime",
        "close",
    ):
        sync_params = list(inspect.signature(getattr(KeyCall, name)).parameters)
        async_params = list(inspect.signature(getattr(AsyncKeyCall, name)).parameters)
        assert sync_params == async_params, f"{name} differs between the clients"

    # Anything public on one should exist on the other.
    def public(cls):
        return {name for name in dir(cls) if not name.startswith("_")}

    assert public(KeyCall) - public(AsyncKeyCall) == set()


@pytest.mark.anyio
async def test_async_tool_round_trip():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("bodies", []).append(json.loads(request.content))
        replied = any(
            item.get("type") == "function_call_output"
            for item in captured["bodies"][-1].get("input", [])
        )
        if replied:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "14C"}]}
                    ],
                    "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
                },
            )
        return httpx.Response(200, json=OPENAI_TOOL_CALL)

    async with client(handler) as c:
        first = await c.generate_text(model="gpt-4o-mini", messages=ask(), tools=[WEATHER])
        assert [call.name for call in first.tool_calls] == ["get_weather"]
        call = first.tool_calls[0]

        final = await c.generate_text(
            model="gpt-4o-mini",
            messages=[
                *ask(),
                first.to_assistant_message(),
                Message(
                    role="user",
                    content=[
                        ToolResult(tool_call_id=call.id, name=call.name, content='{"temp_c": 14}')
                    ],
                ),
            ],
            tools=[WEATHER],
        )

    assert final.text == "14C"
    replay = captured["bodies"][1]["input"]
    assert any(item.get("type") == "function_call" for item in replay)
    assert any(item.get("type") == "function_call_output" for item in replay)


@pytest.mark.anyio
async def test_async_streamed_tool_call_events():
    body = sse(
        {"type": "response.created", "response": {"model": "gpt-4o-mini"}},
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_abc",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_abc",
            "delta": '{"city":"London"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_abc",
            "arguments": '{"city":"London"}',
        },
        {"type": "response.completed", "response": OPENAI_TOOL_CALL},
    )

    async with client(lambda r: stream_response(body)) as c, c.stream_text(
        model="gpt-4o-mini", messages=ask(), tools=[WEATHER]
    ) as stream:
        events = [event async for event in stream]
        result = stream.result()

    kinds = [event.kind for event in events]
    assert kinds == [
        "stream_start",
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_complete",
        "stream_finish",
    ]
    assert dict(result.tool_calls[0].arguments) == {"city": "London"}


@pytest.mark.anyio
async def test_async_image_input():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "blue"}]}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            },
        )

    async with client(handler) as c:
        result = await c.generate_text(
            model="gpt-4o-mini",
            messages=[
                Message(
                    role="user",
                    content=[TextInput(text="what colour?"), ImageInput(data=PNG_BYTES)],
                )
            ],
        )

    assert result.text == "blue"
    content = captured["body"]["input"][0]["content"]
    image = next(c for c in content if c["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(image["image_url"].split(",")[1]) == PNG_BYTES


@pytest.mark.anyio
async def test_async_gates_fire_before_the_network():
    """The pre-network refusals have to hold on this path too, or an async
    caller pays for a request the sync caller never makes."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    async with client(handler, provider="perplexity") as c:
        with pytest.raises(KeyCallError) as excinfo:
            await c.generate_text(model="sonar", messages=ask(), tools=[WEATHER])
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION

    async with client(handler, provider="deepseek") as c:
        with pytest.raises(KeyCallError) as excinfo:
            await c.generate_text(
                model="deepseek-chat",
                messages=[Message(role="user", content=[ImageInput(data=PNG_BYTES)])],
            )
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION

    async with client(handler, provider="moonshot") as c:
        with pytest.raises(KeyCallError) as excinfo:
            await c.generate_text(model="kimi-k3", messages=ask(), temperature=0.2)
        assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE

    assert not calls, "every gate must refuse before any request goes out"


@pytest.mark.anyio
async def test_async_truncated_stream_raises_and_withholds_the_result():
    partial = sse(
        {"type": "response.created", "response": {"model": "gpt-4o-mini"}},
        {"type": "response.output_text.delta", "delta": "half"},
    )

    async with (
        client(lambda r: stream_response(partial)) as c,
        c.stream_text(model="gpt-4o-mini", messages=ask()) as stream,
    ):
        with pytest.raises(KeyCallError) as excinfo:
            [event async for event in stream]
        assert excinfo.value.code is ErrorCode.NETWORK_ERROR
        with pytest.raises(KeyCallError):
            stream.result()


@pytest.mark.anyio
async def test_async_stream_duration_includes_time_to_first_byte():
    import time

    latency = 0.05

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(latency)
        return stream_response(
            sse(
                {"type": "response.created", "response": {"model": "gpt-4o-mini"}},
                {"type": "response.output_text.delta", "delta": "ok"},
                {
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-4o-mini",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "ok"}],
                            }
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                },
            )
        )

    async with (
        client(handler) as c,
        c.stream_text(model="gpt-4o-mini", messages=ask()) as stream,
    ):
        [event async for event in stream]
        result = stream.result()

    assert result.round_trip_duration_ms >= latency * 1000


@pytest.mark.anyio
async def test_async_closed_client_refuses_further_calls():
    c = client(lambda r: httpx.Response(500))
    await c.close()
    assert c.closed
    with pytest.raises(RuntimeError):
        await c.generate_text(model="gpt-4o-mini", messages=ask())
