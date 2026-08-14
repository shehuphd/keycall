"""web_search flag and citation normalization, per provider.

Fixtures mirror live responses captured 2026-08-05 (OpenAI Responses
annotations, Anthropic per-block citations, Gemini groundingMetadata,
Perplexity citations/search_results).
"""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-websearch-key"


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


def simple_messages():
    return [Message(role="user", content=[TextInput(text="What's new?")])]


# --- request-side gating ----------------------------------------------------


@pytest.mark.parametrize("provider", ["deepseek"])
def test_web_search_rejected_for_providers_without_native_search(provider):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client(provider, handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model="some-model", messages=simple_messages(), web_search=True)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "web search" in excinfo.value.message


def test_web_search_rejected_for_custom_targets():
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
        client.generate_text(model="m", messages=simple_messages(), web_search=True)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_web_search_off_sends_no_tools():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "completed", "output": [], "usage": {}})

    make_client("openai", handler).generate_text(model="gpt-4o-mini", messages=simple_messages())
    assert "tools" not in captured["body"]


# --- OpenAI -----------------------------------------------------------------


def test_openai_web_search_tool_and_citations():
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
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search", "query": "python version"},
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Python 3.14 is current.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://www.python.org/downloads/",
                                        "title": "Download Python",
                                        "start_index": 0,
                                        "end_index": 23,
                                    }
                                ],
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
            },
        )

    result = make_client("openai", handler).generate_text(
        model="gpt-4o-mini", messages=simple_messages(), web_search=True
    )
    assert captured["body"]["tools"] == [{"type": "web_search"}]
    assert result.text == "Python 3.14 is current."
    assert len(result.citations) == 1
    assert result.citations[0].url == "https://www.python.org/downloads/"
    assert result.citations[0].title == "Download Python"
    # the web_search_call trace item is not output content
    assert all(p.kind == "text" for p in result.parts)


# --- Anthropic --------------------------------------------------------------


def test_anthropic_web_search_tool_and_citations():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-5",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "server_tool_use", "id": "srv_1", "name": "web_search"},
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv_1",
                        "content": [],
                    },
                    {
                        "type": "text",
                        "text": "Python 3.14 is current.",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://devguide.python.org/versions/",
                                "title": "Status of Python versions",
                                "cited_text": "3.14 is the current stable release.",
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 12000,
                    "output_tokens": 50,
                    "server_tool_use": {"web_search_requests": 1},
                },
            },
        )

    result = make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=simple_messages(), web_search=True
    )
    assert captured["body"]["tools"] == [
        {"type": "web_search_20250305", "name": "web_search"}
    ]
    assert result.text == "Python 3.14 is current."
    assert len(result.citations) == 1
    assert result.citations[0].cited_text == "3.14 is the current stable release."
    assert all(p.kind == "text" for p in result.parts)


# --- Gemini -----------------------------------------------------------------


def test_gemini_google_search_tool_and_grounding_citations():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "responseId": "resp_1",
                "modelVersion": "gemini-flash-latest",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Python 3.14 is current."}]},
                        "finishReason": "STOP",
                        "groundingMetadata": {
                            "webSearchQueries": ["current python version"],
                            "groundingChunks": [
                                {
                                    "web": {
                                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                                        "title": "python.org",
                                    }
                                }
                            ],
                            "groundingSupports": [],
                        },
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 8},
            },
        )

    result = make_client("gemini", handler).generate_text(
        model="gemini-flash-latest", messages=simple_messages(), web_search=True
    )
    assert captured["body"]["tools"] == [{"google_search": {}}]
    assert len(result.citations) == 1
    assert result.citations[0].url.startswith("https://vertexaisearch.cloud.google.com/")
    assert result.citations[0].title == "python.org"


# --- Perplexity -------------------------------------------------------------


def test_perplexity_citations_and_search_results_merged():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "sonar",
                "choices": [
                    {"message": {"content": "Python 3.14."}, "finish_reason": "stop"}
                ],
                "citations": [
                    "https://www.python.org/downloads/",
                    "https://devguide.python.org/versions/",
                ],
                "search_results": [
                    {
                        "url": "https://www.python.org/downloads/",
                        "title": "Download Python",
                        "snippet": "Python 3.14.6 June 10, 2026",
                        "date": "2026-06-10",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    result = make_client("perplexity", handler).generate_text(
        model="sonar", messages=simple_messages(), max_output_tokens=100
    )
    # search_results entries are richer; bare citation URLs fill the rest.
    assert len(result.citations) == 2
    rich = next(c for c in result.citations if c.url == "https://www.python.org/downloads/")
    assert rich.title == "Download Python"
    assert rich.cited_text == "Python 3.14.6 June 10, 2026"
    bare = next(c for c in result.citations if c.url == "https://devguide.python.org/versions/")
    assert bare.title is None


def test_perplexity_accepts_web_search_flag_as_noop():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}},
        )

    make_client("perplexity", handler).generate_text(
        model="sonar", messages=simple_messages(), web_search=True, max_output_tokens=100
    )
    # Sonar always searches; the flag must not inject a tools param it
    # doesn't understand.
    assert "tools" not in captured["body"]


def test_identical_citations_collapse_but_per_claim_ones_survive():
    """OpenAI repeats a source once per claim with no excerpt, which is
    pure duplication; Anthropic repeats it with distinct cited_text, which
    is the attribution and must survive."""
    from keycall._types import Citation
    from keycall.adapters._base import dedupe_citations

    node = "https://nodejs.org/en/download"
    identical = [
        Citation(url=node, title="Download Node.js"),
        Citation(url=node, title="Download Node.js"),
        Citation(url=node, title="Download Node.js"),
    ]
    assert len(dedupe_citations(identical)) == 1

    per_claim = [
        Citation(url=node, title="Download Node.js", cited_text="LTS is 22.x"),
        Citation(url=node, title="Download Node.js", cited_text="ships with npm"),
    ]
    assert len(dedupe_citations(per_claim)) == 2

    # Order is the provider's, first occurrence wins.
    mixed = [*identical, *per_claim]
    assert [c.cited_text for c in dedupe_citations(mixed)] == [None, "LTS is 22.x", "ships with npm"]


def test_openai_repeated_url_citations_collapse():
    payload = {
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Node 22 is current.",
                        "annotations": [
                            {"type": "url_citation", "url": "https://nodejs.org/en/download",
                             "title": "Download Node.js"},
                            {"type": "url_citation", "url": "https://nodejs.org/en/download",
                             "title": "Download Node.js"},
                        ],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = client.generate_text(
        model="gpt-4o-mini",
        messages=[Message(role="user", content=[TextInput(text="node version?")])],
        web_search=True,
    )
    client.close()
    assert [c.url for c in result.citations] == ["https://nodejs.org/en/download"]


def test_gemini_citations_match_between_streamed_and_non_streamed():
    """Gemini deduped on the streaming path only, so the same request
    returned different citations depending on how it was called."""
    chunk = {
        "modelVersion": "gemini-flash-latest",
        "candidates": [
            {
                "content": {"parts": [{"text": "ok"}]},
                "groundingMetadata": {
                    "groundingChunks": [
                        {"web": {"uri": "https://example.com/a", "title": "A"}},
                        {"web": {"uri": "https://example.com/a", "title": "A"}},
                        {"web": {"uri": "https://example.com/b", "title": "B"}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    }

    plain = KeyCall(
        provider="gemini",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=chunk)),
    )
    ask = [Message(role="user", content=[TextInput(text="hi")])]
    non_streamed = plain.generate_text(model="gemini-flash-latest", messages=ask, web_search=True)
    plain.close()

    body = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    streaming = KeyCall(
        provider="gemini",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        ),
    )
    with streaming.stream_text(
        model="gemini-flash-latest", messages=ask, web_search=True
    ) as stream:
        events = list(stream)
        streamed = stream.result()
    streaming.close()

    assert [c.url for c in non_streamed.citations] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert streamed.citations == non_streamed.citations
    # The events a caller saw match the result they get back.
    found = [e.citation for e in events if e.kind == "citation"]
    assert tuple(found) == streamed.citations


def test_xai_web_search_reroutes_to_the_responses_surface():
    """Grok's search lives on POST /v1/responses (the OpenAI Responses
    shape), while plain generation stays on chat completions — both
    verified live 2026-08-14. One flag must switch route, body shape,
    and parser together."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "grok-4.6",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": []},
                    {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Today is Friday, August 14, 2026.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://www.datetoday.net/",
                                        "title": "Date Today",
                                    }
                                ],
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 6054, "output_tokens": 2033, "total_tokens": 8087},
            },
        )

    client = make_client("xai", handler)
    result = client.generate_text(
        model="grok-4.6", messages=simple_messages(), web_search=True
    )
    client.close()

    assert captured["path"] == "/v1/responses"
    assert {"type": "web_search"} in captured["body"]["tools"]
    assert "messages" not in captured["body"], "the responses surface takes input items"
    assert result.text == "Today is Friday, August 14, 2026."
    assert [c.url for c in result.citations] == ["https://www.datetoday.net/"]
    assert result.usage.total_tokens == 8087


def _moonshot_search_round(call_id="ws_1"):
    """Round one of Moonshot's builtin flow: the search already ran
    server-side; the model asks for its echo (shape observed live
    2026-08-14 on kimi-k2.6)."""
    return {
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "$web_search",
                                "arguments": json.dumps(
                                    {
                                        "search_result": {"search_id": "s-123"},
                                        "usage": {"total_tokens": 6053},
                                    }
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 27, "completion_tokens": 1, "total_tokens": 28},
    }


def test_moonshot_web_search_runs_the_echo_loop():
    """The builtin's round trip is KeyCall's to complete: one call from
    the caller's seat, two on the wire, with the echo carrying the
    arguments back verbatim and usage summed across rounds."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json=_moonshot_search_round())
        return httpx.Response(
            200,
            json={
                "model": "kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "It is August 14, 2026."},
                    }
                ],
                "usage": {"prompt_tokens": 8279, "completion_tokens": 1648, "total_tokens": 9927},
            },
        )

    client = make_client("moonshot", handler)
    result = client.generate_text(
        model="kimi-k2.6", messages=simple_messages(), web_search=True
    )
    client.close()

    assert len(bodies) == 2
    assert {"type": "builtin_function", "function": {"name": "$web_search"}} in bodies[0]["tools"]
    # Round two replays the call and echoes its arguments as the tool answer.
    tool_message = next(m for m in bodies[1]["messages"] if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "ws_1"
    assert json.loads(tool_message["content"])["search_result"] == {"search_id": "s-123"}
    assert {"type": "builtin_function", "function": {"name": "$web_search"}} in bodies[1]["tools"]

    assert result.text == "It is August 14, 2026."
    # 28 + 9927: both rounds were billed and both are reported.
    assert result.usage.total_tokens == 9955
    # The handshake's tool call is not part of the answer's parts.
    assert all(p.kind == "text" for p in result.parts)


def test_moonshot_refuses_an_unbounded_echo_loop():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_moonshot_search_round())

    client = make_client("moonshot", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(model="kimi-k2.6", messages=simple_messages(), web_search=True)
    client.close()
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "budget" in excinfo.value.message


def test_moonshot_without_web_search_sends_no_builtin():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    client = make_client("moonshot", handler)
    client.generate_text(model="kimi-k2.6", messages=simple_messages())
    client.close()
    assert "tools" not in captured["body"]


def test_xai_without_web_search_stays_on_chat_completions():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "model": "grok-4.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = make_client("xai", handler)
    result = client.generate_text(model="grok-4.6", messages=simple_messages())
    client.close()
    assert captured["path"] == "/v1/chat/completions"
    assert result.text == "Hi."
