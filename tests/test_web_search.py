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


@pytest.mark.parametrize("provider", ["deepseek", "moonshot"])
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
