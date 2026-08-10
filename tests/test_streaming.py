"""Streaming: per-protocol assembly, truncation, caps, scrubbing, parity.

Fixture streams mirror the shapes captured live 2026-08-08.
"""

import json
import time

import httpx
import pytest

from keycall import (
    CitationFound,
    ErrorCode,
    KeyCall,
    KeyCallError,
    Message,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextInput,
    UnknownStreamEvent,
)

CANARY = "sk-canary-streaming-key"


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
    return [Message(role="user", content=[TextInput(text="hi")])]


def stream_response(body: bytes, headers=None):
    base = {"content-type": "text/event-stream"}
    return httpx.Response(200, content=body, headers={**base, **(headers or {})})


# --- OpenAI -----------------------------------------------------------------


def openai_stream_body(include_terminal=True):
    events = [
        (None, {"type": "response.created", "response": {"model": "gpt-4o-mini"}}),
        (None, {"type": "response.output_text.delta", "delta": "hel"}),
        (None, {"type": "response.output_text.delta", "delta": "lo"}),
    ]
    if include_terminal:
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
                                "type": "message",
                                "content": [{"type": "output_text", "text": "hello"}],
                            }
                        ],
                        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    },
                },
            )
        )
    return sse(*events)


def test_stream_duration_includes_time_to_first_byte():
    """Providers that buffer before the first byte spend most of the round
    trip there; measuring only from the response headers reported ~1% of
    the elapsed duration on Anthropic."""
    latency_s = 0.05

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(latency_s)
        return stream_response(openai_stream_body())

    with make_client("openai", handler).stream_text(
        model="gpt-4o-mini", messages=messages()
    ) as stream:
        list(stream)
        result = stream.result()

    assert result.round_trip_duration_ms >= latency_s * 1000


def test_openai_stream_events_and_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return stream_response(openai_stream_body(), headers={"x-request-id": "req-1"})

    with make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream:
        events = list(stream)
        result = stream.result()
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["StreamStart", "TextDelta", "TextDelta", "StreamFinish"]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "hello"
    assert result.text == "hello"
    assert result.usage.total_tokens == 5
    assert result.provider_request_id == "req-1"
    assert result.finish_reason == "completed"
    assert result.round_trip_duration_ms >= 0


def test_openai_stream_failed_event_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse(
                (None, {"type": "response.created", "response": {"model": "gpt-4o-mini"}}),
                (
                    None,
                    {
                        "type": "response.failed",
                        "response": {"error": {"message": f"boom for {CANARY}"}},
                    },
                ),
            )
        )

    with (
        make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert CANARY not in str(excinfo.value)
    with pytest.raises(KeyCallError):
        stream.result()


# --- Anthropic --------------------------------------------------------------


def anthropic_events(terminal=True):
    events = [
        ("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "usage": {"input_tokens": 4}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        ("ping", {"type": "ping"}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
    ]
    if terminal:
        events.append(("message_stop", {"type": "message_stop"}))
    return events


def test_anthropic_stream_usage_split_across_events():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(sse(*anthropic_events()))

    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages()
    ) as stream:
        events = list(stream)
        result = stream.result()
    assert isinstance(events[0], StreamStart)
    assert result.text == "ok"
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 1
    assert result.finish_reason == "end_turn"
    assert not any("no usage" in w for w in result.warnings)


def test_anthropic_forced_tool_schema_streams_json_fragments():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse(
                ("message_start", {"type": "message_start", "message": {"model": "claude-opus-5", "usage": {"input_tokens": 4}}}),
                ("content_block_start", {"type": "content_block_start", "index": 0,
                                          "content_block": {"type": "tool_use", "name": "keycall_response"}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "input_json_delta", "partial_json": '{"word":'}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "input_json_delta", "partial_json": ' "ok"}'}}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
                ("message_stop", {"type": "message_stop"}),
            )
        )

    schema = {"type": "object", "properties": {"word": {"type": "string"}}}
    with make_client("anthropic", handler).stream_text(
        model="claude-opus-5", messages=messages(), response_schema=schema
    ) as stream:
        list(stream)
        result = stream.result()
    assert json.loads(result.text) == {"word": "ok"}


def test_anthropic_inband_error_scrubbed():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse(
                ("message_start", {"type": "message_start", "message": {"model": "m", "usage": {}}}),
                ("error", {"type": "error", "error": {"type": "overloaded_error", "message": f"overloaded {CANARY}"}}),
            )
        )

    with (
        make_client("anthropic", handler).stream_text(model="m", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert excinfo.value.retryable
    assert CANARY not in str(excinfo.value)


# --- Gemini -----------------------------------------------------------------


def test_gemini_stream_finish_reason_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(":streamGenerateContent")
        assert request.url.params.get("alt") == "sse"
        return stream_response(
            sse(
                (None, {"modelVersion": "gemini-flash-latest", "responseId": "r-9",
                        "candidates": [{"content": {"parts": [{"text": "o"}]}}]}),
                (None, {"candidates": [{"content": {"parts": [{"text": "k"}]}, "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}}),
            )
        )

    with make_client("gemini", handler).stream_text(
        model="gemini-flash-latest", messages=messages()
    ) as stream:
        events = list(stream)
        result = stream.result()
    assert result.text == "ok"
    assert result.finish_reason == "STOP"
    assert result.usage.total_tokens == 5
    assert result.provider_request_id == "r-9"
    assert isinstance(events[-1], StreamFinish)


# --- Compat and Perplexity --------------------------------------------------


def compat_chunk(content=None, finish=None, usage=None, **extra):
    delta = {"content": content} if content is not None else {}
    chunk = {
        "object": "chat.completion.chunk",
        "model": "deepseek-v4-flash",
        "choices": [{"delta": delta, "finish_reason": finish}],
        **extra,
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def test_compat_stream_done_terminal_and_stream_options():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return stream_response(
            sse(
                (None, compat_chunk("o")),
                (None, compat_chunk("k", finish="stop",
                                     usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})),
                (None, "[DONE]"),
            )
        )

    with make_client("deepseek", handler).stream_text(
        model="deepseek-v4-flash", messages=messages()
    ) as stream:
        list(stream)
        result = stream.result()
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert result.text == "ok"
    assert result.usage.total_tokens == 5
    assert result.finish_reason == "stop"


def test_custom_target_gets_no_stream_options():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return stream_response(sse((None, compat_chunk("x", finish="stop")), (None, "[DONE]")))

    client = KeyCall(
        provider="my-lab",
        api_key=CANARY,
        protocol="openai-compatible",
        base_url="https://llm.example.edu/v1",
        httpx_transport=httpx.MockTransport(handler),
    )
    with client.stream_text(model="some-model", messages=messages()) as stream:
        list(stream)
        result = stream.result()
    assert "stream_options" not in captured["body"]
    assert any("no usage" in w for w in result.warnings)


def test_perplexity_done_object_terminal_with_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        chunk = {
            "object": "chat.completion.chunk",
            "model": "sonar",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": None}],
            "search_results": [{"url": "https://example.com/a", "title": "A"}],
        }
        done = {
            "object": "chat.completion.done",
            "model": "sonar",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "search_results": [{"url": "https://example.com/a", "title": "A"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        return stream_response(sse((None, chunk), (None, done)))

    with make_client("perplexity", handler).stream_text(
        model="sonar", messages=messages(), max_output_tokens=16
    ) as stream:
        events = list(stream)
        result = stream.result()
    citation_events = [e for e in events if isinstance(e, CitationFound)]
    assert len(citation_events) == 1  # deduplicated across chunks
    assert result.citations[0].url == "https://example.com/a"
    assert result.usage.total_tokens == 4


# --- failure handling -------------------------------------------------------


def test_truncated_stream_raises_and_result_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(openai_stream_body(include_terminal=False))

    with (
        make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.NETWORK_ERROR
    assert "incomplete" in excinfo.value.message
    with pytest.raises(KeyCallError):
        stream.result()


def test_compat_close_without_done_is_truncation():
    def handler(request: httpx.Request) -> httpx.Response:
        # finish_reason present but no [DONE]: still a truncation.
        return stream_response(sse((None, compat_chunk("x", finish="stop"))))

    with (
        make_client("deepseek", handler).stream_text(
            model="deepseek-v4-flash", messages=messages()
        ) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.NETWORK_ERROR


def test_malformed_stream_json_is_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(b"data: {not json}\n\n")

    with (
        make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_single_oversized_event_hits_per_event_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        huge = "x" * (1024 * 1024 + 10)
        return stream_response(f'data: {{"type": "response.created", "pad": "{huge}"}}\n\n'.encode())

    with (
        make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "byte limit" in excinfo.value.message


def test_cumulative_cap_applies_to_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        body = sse(*(((None, {"type": "response.output_text.delta", "delta": "x" * 50}),) * 100))
        return stream_response(body)

    client = make_client("openai", handler, max_response_bytes=2000)
    with (
        client.stream_text(model="gpt-4o-mini", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert "byte limit" in excinfo.value.message


def test_pre_stream_http_error_classifies_normally():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    stream = make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages())
    with pytest.raises(KeyCallError) as excinfo:
        stream.__enter__()
    assert excinfo.value.code is ErrorCode.INVALID_API_KEY


def test_result_before_finish_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(openai_stream_body())

    with make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream:
        with pytest.raises(KeyCallError):
            stream.result()
        list(stream)
        assert stream.result().text == "hello"


def test_early_break_closes_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(openai_stream_body())

    with make_client("openai", handler).stream_text(model="gpt-4o-mini", messages=messages()) as stream:
        for event in stream:
            if isinstance(event, TextDelta):
                break
    # The context manager closed the connection; the unfinished stream has
    # no result.
    with pytest.raises(KeyCallError):
        stream.result()


def test_unknown_event_type_surfaces_bounded():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse(
                (None, {"type": "response.created", "response": {"model": "m"}}),
                (None, {"type": "response.novel_thing", "blob": "z" * 500}),
                (None, {"type": "response.completed", "response": {"model": "m", "status": "completed",
                                                                    "output": [], "usage": {}}}),
            )
        )

    with make_client("openai", handler).stream_text(model="m", messages=messages()) as stream:
        events = list(stream)
    unknown = [e for e in events if isinstance(e, UnknownStreamEvent)]
    assert len(unknown) == 1
    assert unknown[0].provider_kind == "response.novel_thing"
    assert "z" not in unknown[0].provider_kind


def test_anthropic_web_search_with_schema_still_blocked_streaming():
    client = KeyCall(provider="anthropic", api_key=CANARY, httpx_transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(KeyCallError) as excinfo:
        client.stream_text(
            model="claude-opus-5",
            messages=messages(),
            web_search=True,
            response_schema={"type": "object"},
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


# --- async parity -----------------------------------------------------------


@pytest.mark.anyio
async def test_async_stream_parity():
    from keycall import AsyncKeyCall

    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(openai_stream_body(), headers={"x-request-id": "req-a"})

    client = AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    async with client.stream_text(model="gpt-4o-mini", messages=messages()) as stream:
        events = [event async for event in stream]
        result = stream.result()
    await client.close()
    assert [type(e).__name__ for e in events] == [
        "StreamStart",
        "TextDelta",
        "TextDelta",
        "StreamFinish",
    ]
    assert result.text == "hello"
    assert result.provider_request_id == "req-a"


def test_gemini_usage_in_trailing_chunk_after_finish_reason():
    """The final usageMetadata can arrive in a chunk after the one carrying
    finishReason; the close completes the stream with that usage."""

    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse(
                (None, {"candidates": [{"content": {"parts": [{"text": "ok"}]},
                                         "finishReason": "MAX_TOKENS"}]}),
                (None, {"candidates": [{}],
                        "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 9}}),
            )
        )

    with make_client("gemini", handler).stream_text(model="m", messages=messages()) as stream:
        events = list(stream)
        result = stream.result()
    assert isinstance(events[-1], StreamFinish)
    assert result.finish_reason == "MAX_TOKENS"
    assert result.usage.total_tokens == 9
    assert not any("no usage" in w for w in result.warnings)


def test_gemini_close_without_finish_reason_is_truncation():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            sse((None, {"candidates": [{"content": {"parts": [{"text": "o"}]}}]}))
        )

    with (
        make_client("gemini", handler).stream_text(model="m", messages=messages()) as stream,
        pytest.raises(KeyCallError) as excinfo,
    ):
        list(stream)
    assert excinfo.value.code is ErrorCode.NETWORK_ERROR
