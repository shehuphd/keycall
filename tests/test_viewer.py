"""Viewer: API layer, HTTP server, auth, and the credential-leak canary."""

import json
import threading
import urllib.request

import httpx
import pytest

from keycall._sources import Target
from keycall.viewer import Token
from keycall.viewer._api import (
    browse_models,
    check_target,
    generate,
    generate_image,
    generate_stream_events,
    list_targets,
    verify_target,
)
from keycall.viewer._registry import Registry
from keycall.viewer._server import _Server

CANARY = "sk-canary-viewer-key-do-not-leak"


def _sse_body(events) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events)).encode()


def _openai_stream_events(include_terminal=True):
    events = [
        {"type": "response.created", "response": {"model": "gpt-4o-mini"}},
        {"type": "response.output_text.delta", "delta": "hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
    ]
    if include_terminal:
        events.append(
            {
                "type": "response.completed",
                "response": {
                    "model": "gpt-4o-mini",
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            }
        )
    return events


def openai_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={"data": [{"id": "gpt-4o-mini"}, {"id": "text-embedding-3-small"}]},
        )
    if request.content and json.loads(request.content).get("stream"):
        return httpx.Response(
            200,
            content=_sse_body(_openai_stream_events()),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        },
    )


def make_registry():
    targets = [
        Target(provider="openai", key=CANARY, name="my-openai"),
        Target(provider="anthropic", key=CANARY + "-2", name="my-claude"),
    ]
    return Registry(targets, httpx_transport=httpx.MockTransport(openai_handler))


# --- API layer --------------------------------------------------------------


def test_list_targets_never_exposes_key():
    reg = make_registry()
    try:
        body = list_targets(reg)
        blob = json.dumps(body)
        assert CANARY not in blob
        assert body["targets"][0]["name"] == "my-openai"
        assert "key" not in body["targets"][0]
        assert body["targets"][0]["id"] == 0
    finally:
        reg.close()


def test_check_target_lists_all_categories():
    reg = make_registry()
    try:
        body = check_target(reg, 0)
        ids = [m["id"] for m in body["models"]]
        assert "gpt-4o-mini" in ids
        assert "text-embedding-3-small" in ids  # all categories, not just text
    finally:
        reg.close()


def test_browse_models_filters_by_category():
    reg = make_registry()
    try:
        body = browse_models(reg, 0, category="text_generation", refresh=True)
        ids = [m["id"] for m in body["models"]]
        assert ids == ["gpt-4o-mini"]
    finally:
        reg.close()


def test_generate_returns_result_without_key():
    reg = make_registry()
    try:
        body = generate(reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"})
        assert body["text"] == "hello"
        assert CANARY not in json.dumps(body)
    finally:
        reg.close()


def test_generate_bad_input():
    reg = make_registry()
    try:
        assert "error" in generate(reg, 0, {"target": 0, "model": "", "prompt": "hi"})
        assert "error" in generate(reg, 0, {"target": 0, "model": "m", "prompt": ""})
    finally:
        reg.close()


def test_verify_target_structured_result():
    reg = make_registry()
    try:
        body = verify_target(reg, 0, generate=True)
        assert body["listed_ok"] is True
        assert body["outcome"] == "generated"
        assert body["attempts"][0]["ok"] is True
        assert CANARY not in json.dumps(body)
    finally:
        reg.close()


def test_unknown_target_id():
    reg = make_registry()
    try:
        assert check_target(reg, 99)["error"]["code"] == "not_found"
        assert generate(reg, 99, {"model": "m", "prompt": "p"})["error"]["code"] == "not_found"
    finally:
        reg.close()


# --- HTTP server ------------------------------------------------------------


@pytest.fixture
def server():
    reg = make_registry()
    token = Token()
    srv = _Server(("127.0.0.1", 0), reg, token)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}", token.value
    srv.shutdown()
    reg.close()


def _get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-KeyCall-Token", token)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url, body, token=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-KeyCall-Token", token)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_requires_token(server):
    base, _ = server
    status, body = _get(f"{base}/api/targets")  # no token
    assert status == 403
    assert body["error"]["code"] == "unauthorized"


def test_api_rejects_wrong_token(server):
    base, _ = server
    status, _ = _get(f"{base}/api/targets", token="wrong-token")
    assert status == 403


def test_api_accepts_valid_token(server):
    base, token = server
    status, body = _get(f"{base}/api/targets", token=token)
    assert status == 200
    assert len(body["targets"]) == 2


def test_health_ok_with_token(server):
    base, token = server
    status, body = _get(f"{base}/api/health", token=token)
    assert status == 200
    assert body["status"] == "ok"
    assert body["targets"] == 2


def test_static_index_served_without_token(server):
    base, _ = server
    req = urllib.request.Request(f"{base}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert b"KeyCall" in resp.read()


def test_generate_over_http_no_key_leak(server):
    base, token = server
    status, body = _post(
        f"{base}/api/generate",
        {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"},
        token=token,
    )
    assert status == 200
    assert body["text"] == "hello"
    assert CANARY not in json.dumps(body)


def test_token_query_param_also_works(server):
    base, token = server
    status, body = _get(f"{base}/api/targets?token={token}")
    assert status == 200
    assert len(body["targets"]) == 2


def test_concurrent_browse_models_single_flight():
    """Dashboard and playground booting together must share one upstream
    fetch, not race duplicate live calls."""
    import threading as _threading

    calls = {"count": 0}
    gate = _threading.Event()

    def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            calls["count"] += 1
            gate.wait(timeout=5)  # hold the first fetch open
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        return httpx.Response(200, json={})

    targets = [Target(provider="openai", key=CANARY, name="t")]
    reg = Registry(targets, httpx_transport=httpx.MockTransport(slow_handler))
    try:
        results = []

        def worker():
            results.append(browse_models(reg, 0, category=None, refresh=False))

        threads = [_threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(timeout=10)

        assert calls["count"] == 1  # one upstream call, shared by all three
        assert all("models" in r for r in results)
    finally:
        reg.close()


def test_empty_registry_then_add_source(tmp_path):
    """Viewer can start with zero targets; a source loads via the API."""
    from keycall.viewer._api import add_source

    reg = Registry([], httpx_transport=httpx.MockTransport(openai_handler))
    try:
        assert list_targets(reg)["targets"] == []

        keyfile = tmp_path / "keys.toml"
        keyfile.write_text(
            f'[[targets]]\nprovider = "openai"\nkey = "{CANARY}"\nname = "added"\n',
            encoding="utf-8",
        )
        body = add_source(reg, {"path": str(keyfile)})
        assert "error" not in body
        assert [t["name"] for t in body["targets"]] == ["added"]
        assert CANARY not in json.dumps(body)
    finally:
        reg.close()


def test_add_source_rejects_bad_input(tmp_path):
    from keycall.viewer._api import add_source

    reg = Registry([], httpx_transport=httpx.MockTransport(openai_handler))
    try:
        assert add_source(reg, {})["error"]["code"] == "bad_request"
        assert add_source(reg, {"path": "-"})["error"]["code"] == "bad_request"
        assert add_source(reg, {"path": "/nonexistent.toml"})["error"]["code"] == "bad_source"
    finally:
        reg.close()


def test_add_source_over_http_token_gated(server, tmp_path):
    base, token = server
    keyfile = tmp_path / "keys.toml"
    keyfile.write_text(
        f'[[targets]]\nprovider = "openai"\nkey = "{CANARY}-3"\nname = "http-added"\n',
        encoding="utf-8",
    )
    status, _ = _post(f"{base}/api/source", {"path": str(keyfile)})  # no token
    assert status == 403
    status, body = _post(f"{base}/api/source", {"path": str(keyfile)}, token=token)
    assert status == 200
    assert any(t["name"] == "http-added" for t in body["targets"])


# --- registry hardening -----------------------------------------------------


def test_add_targets_closes_opened_clients_on_later_failure(monkeypatch):
    from keycall import ErrorCode, KeyCallError
    from keycall.viewer import _registry as registry_module

    instances = []

    class StubClient:
        def __init__(self, *, provider, api_key, protocol=None, base_url=None,
                     httpx_transport=None):
            if provider == "bad":
                raise KeyCallError("unknown provider", code=ErrorCode.UNSUPPORTED_PROVIDER)
            self.provider = provider
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(registry_module, "KeyCall", StubClient)
    reg = Registry([])
    with pytest.raises(KeyCallError):
        reg.add_targets(
            [
                Target(provider="openai", key=CANARY),
                Target(provider="bad", key=CANARY),
            ]
        )
    assert len(instances) == 1
    assert instances[0].closed
    assert reg.views() == []


def test_cached_discovery_expires_after_ttl():
    reg = make_registry()
    assert "error" not in check_target(reg, 0)
    assert reg.cached_discovery(0) is not None
    # Age the entry past the TTL instead of sleeping through it.
    with reg._lock:
        reg._entries[0].discovery_at -= 301.0
    assert reg.cached_discovery(0) is None
    reg.close()


# --- server input validation ------------------------------------------------


def test_verify_attempts_must_be_bounded_integer(server):
    base, token = server
    for bad in ("abc", 0, -1, 1000, True, None):
        status, body = _post(
            f"{base}/api/verify", {"target": 0, "attempts": bad}, token=token
        )
        assert status == 400, bad
        assert body["error"]["code"] == "bad_request"


def test_non_object_json_body_rejected(server):
    base, token = server
    req = urllib.request.Request(
        f"{base}/api/verify", data=b"[1, 2, 3]", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("X-KeyCall-Token", token)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status == 400


def test_bad_content_length_rejected(server):
    import http.client
    from urllib.parse import urlparse

    base, token = server
    parsed = urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        conn.putrequest("POST", "/api/verify", skip_host=True)
        conn.putheader("Host", parsed.netloc)
        conn.putheader("X-KeyCall-Token", token)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 400
    finally:
        conn.close()


# --- streamed generation ----------------------------------------------------


def _post_sse(url, body, token=None):
    """POST and read the raw SSE response into a list of parsed events."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-KeyCall-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as e:
        return e.code, [json.loads(e.read())]
    events = [
        json.loads(frame[len("data:"):])
        for frame in raw.split("\n\n")
        if frame.startswith("data:")
    ]
    return status, events


def test_generate_stream_events_direct():
    reg = make_registry()
    events = list(
        generate_stream_events(reg, 0, {"model": "gpt-4o-mini", "prompt": "hi"})
    )
    kinds = [e.get("kind") for e in events]
    assert kinds == ["stream_start", "text_delta", "text_delta", "result"]
    assert "".join(e["text"] for e in events if e.get("kind") == "text_delta") == "hello"
    result = events[-1]
    assert result["text"] == "hello"
    assert result["usage"]["total_tokens"] == 5
    assert CANARY not in json.dumps(events)
    reg.close()


def test_generate_stream_bad_input_yields_single_error():
    reg = make_registry()
    assert list(generate_stream_events(reg, 99, {"model": "m", "prompt": "p"})) == [
        {"error": {"code": "not_found", "message": "unknown target id"}}
    ]
    events = list(generate_stream_events(reg, 0, {"prompt": "no model"}))
    assert events[0]["error"]["code"] == "bad_request"
    reg.close()


def test_stream_endpoint_over_http(server):
    base, token = server
    status, events = _post_sse(
        f"{base}/api/generate/stream",
        {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"},
        token=token,
    )
    assert status == 200
    assert [e.get("kind") for e in events] == [
        "stream_start",
        "text_delta",
        "text_delta",
        "result",
    ]
    assert events[-1]["text"] == "hello"
    assert CANARY not in json.dumps(events)


def test_stream_endpoint_requires_token(server):
    base, _ = server
    status, events = _post_sse(
        f"{base}/api/generate/stream",
        {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"},
    )
    assert status == 403
    assert events[0]["error"]["code"] == "unauthorized"


def test_stream_endpoint_truncation_surfaces_error_event():
    """A provider stream that dies without its terminal event must reach the
    browser as an error payload, never a hung or silently-closed response."""

    def truncated_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        return httpx.Response(
            200,
            content=_sse_body(_openai_stream_events(include_terminal=False)),
            headers={"content-type": "text/event-stream"},
        )

    reg = Registry(
        [Target(provider="openai", key=CANARY, name="trunc")],
        httpx_transport=httpx.MockTransport(truncated_handler),
    )
    token = Token()
    srv = _Server(("127.0.0.1", 0), reg, token)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        port = srv.server_address[1]
        status, events = _post_sse(
            f"http://127.0.0.1:{port}/api/generate/stream",
            {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"},
            token=token.value,
        )
        assert status == 200
        assert events[-1]["error"]["code"] == "network_error"
        assert "incomplete" in events[-1]["error"]["message"]
        assert CANARY not in json.dumps(events)
    finally:
        srv.shutdown()
        reg.close()


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _tool_call_registry():
    """A target whose model asks for a tool, then answers once it has the
    result. The second call is distinguished by the replayed history."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        body = json.loads(request.content)
        seen.append(body)
        replied = any(
            item.get("type") == "function_call_output" for item in body.get("input", [])
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
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"London"}',
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            },
        )

    targets = [Target(provider="openai", key=CANARY, name="my-openai")]
    return Registry(targets, httpx_transport=httpx.MockTransport(handler)), seen


def test_generate_surfaces_tool_calls_with_their_echo_data():
    reg, seen = _tool_call_registry()
    try:
        body = generate(
            reg,
            0,
            {"target": 0, "model": "gpt-4o-mini", "prompt": "weather?", "tools": [WEATHER_TOOL]},
        )
        assert [c["name"] for c in body["tool_calls"]] == ["get_weather"]
        call = body["tool_calls"][0]
        assert call["arguments"] == {"city": "London"}
        # The browser has to hand this back untouched on the next turn.
        assert call["opaque"]
        assert seen[0]["tools"][0]["name"] == "get_weather"
        assert CANARY not in json.dumps(body)
    finally:
        reg.close()


def test_generate_continues_from_replayed_history():
    """The Playground owns the loop, so a continuation arrives as history
    the server rebuilds into messages."""
    reg, seen = _tool_call_registry()
    try:
        first = generate(
            reg,
            0,
            {"target": 0, "model": "gpt-4o-mini", "prompt": "weather?", "tools": [WEATHER_TOOL]},
        )
        call = first["tool_calls"][0]
        second = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "weather?",
                "tools": [WEATHER_TOOL],
                "history": [
                    {"role": "assistant", "parts": [call]},
                    {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "tool_result",
                                "tool_call_id": call["id"],
                                "name": call["name"],
                                "content": '{"temp_c": 14}',
                            }
                        ],
                    },
                ],
            },
        )
        assert second["text"] == "14C"
        assert not second["tool_calls"]
        # The replay carried the call and its result to the provider.
        replay = seen[1]["input"]
        assert any(item.get("type") == "function_call" for item in replay)
        assert any(item.get("type") == "function_call_output" for item in replay)
    finally:
        reg.close()


def test_malformed_tools_and_history_are_named_bad_requests():
    reg = make_registry()
    try:
        bad_tool = generate(
            reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "hi", "tools": [{"name": "x"}]}
        )
        assert bad_tool["error"]["code"] == "bad_request"
        assert "input_schema" in bad_tool["error"]["message"]

        not_a_list = generate(
            reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "hi", "tools": "nope"}
        )
        assert not_a_list["error"]["code"] == "bad_request"

        bad_history = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "hi",
                "history": [{"role": "assistant", "parts": [{"kind": "mystery"}]}],
            },
        )
        assert bad_history["error"]["code"] == "bad_request"
        assert "mystery" in bad_history["error"]["message"]
    finally:
        reg.close()


def test_stream_emits_tool_call_events():
    events = [
        {"type": "response.created", "response": {"model": "gpt-4o-mini"}},
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"city":"London"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "arguments": '{"city":"London"}',
        },
        {
            "type": "response.completed",
            "response": {
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"London"}',
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        return httpx.Response(
            200, content=_sse_body(events), headers={"content-type": "text/event-stream"}
        )

    reg = Registry(
        [Target(provider="openai", key=CANARY, name="my-openai")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        streamed = list(
            generate_stream_events(
                reg,
                0,
                {"target": 0, "model": "gpt-4o-mini", "prompt": "weather?", "tools": [WEATHER_TOOL]},
            )
        )
    finally:
        reg.close()

    kinds = [e.get("kind") for e in streamed]
    assert "tool_call_started" in kinds
    assert "tool_call_complete" in kinds
    final = streamed[-1]
    assert final["kind"] == "result"
    assert final["tool_calls"][0]["arguments"] == {"city": "London"}
    assert CANARY not in json.dumps(streamed)


PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_playground_image_reaches_the_provider_as_bytes():
    """The browser holds the file, so it posts base64; the server decodes
    it into the same ImageInput a library caller would build."""
    import base64

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        seen["body"] = json.loads(request.content)
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

    reg = Registry(
        [Target(provider="openai", key=CANARY, name="my-openai")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "what colour?",
                "images": [{"data_base64": base64.b64encode(PNG_1PX).decode()}],
            },
        )
    finally:
        reg.close()

    assert body["text"] == "blue"
    content = seen["body"]["input"][0]["content"]
    image = next(c for c in content if c["type"] == "input_image")
    # Media type comes from the bytes, not from whatever the browser said.
    assert image["image_url"].startswith("data:image/png;base64,")


def test_playground_image_can_carry_a_turn_without_a_prompt():
    import base64

    reg = make_registry()
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "",
                "images": [{"data_base64": base64.b64encode(PNG_1PX).decode()}],
            },
        )
    finally:
        reg.close()
    assert "error" not in body


def test_malformed_playground_images_are_named_bad_requests():
    reg = make_registry()
    try:
        both = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "hi",
                "images": [{"url": "https://x/i.png", "data_base64": "AAAA"}],
            },
        )
        assert both["error"]["code"] == "bad_request"
        assert "exactly one" in both["error"]["message"]

        not_base64 = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "hi",
                "images": [{"data_base64": "this is not base64!!"}],
            },
        )
        assert not_base64["error"]["code"] == "bad_request"
    finally:
        reg.close()


def test_generate_image_returns_the_picture_and_its_media_type():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-image-1"}]})
        return httpx.Response(
            200,
            json={
                "output_format": "png",
                "data": [{"b64_json": "QUJD"}],
                "usage": {"input_tokens": 5, "output_tokens": 100, "total_tokens": 105},
            },
        )

    reg = Registry(
        [Target(provider="openai", key=CANARY, name="my-openai")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = generate_image(
            reg, 0, {"target": 0, "model": "gpt-image-1", "prompt": "a blue circle"}
        )
    finally:
        reg.close()

    assert body["images"] == [{"base64_data": "QUJD", "media_type": "image/png"}]
    assert body["operation"] == "image_generation"
    assert CANARY not in json.dumps(body)


def test_generate_image_reports_a_provider_that_cannot():
    reg = make_registry()
    try:
        body = generate_image(
            reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "a blue circle"}
        )
        missing = generate_image(reg, 0, {"target": 0, "model": "", "prompt": ""})
    finally:
        reg.close()
    # make_registry() is an OpenAI target, so the call reaches the provider
    # and the mock refuses it; the shape is what matters here.
    assert "error" in body or "images" in body
    assert missing["error"]["code"] == "bad_request"


WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 24
PDF_BYTES = b"%PDF-1.4\n1 0 obj\nendobj\ntrailer\n%%EOF\n"


def test_playground_sound_file_reaches_the_provider_as_bytes():
    """Audio takes the same route images do: base64 over the wire, decoded
    into the AudioInput a library caller would build. Gemini is the only
    provider that accepts one, so the test targets Gemini."""
    import base64

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"models": [{"name": "models/gemini-2.5-flash"}]})
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "a chime"}], "role": "model"}},
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
            },
        )

    reg = Registry(
        [Target(provider="gemini", key=CANARY, name="my-gemini")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gemini-2.5-flash",
                "prompt": "what do you hear?",
                "audio": [{"data_base64": base64.b64encode(WAV_BYTES).decode()}],
            },
        )
    finally:
        reg.close()

    assert body["text"] == "a chime"
    parts = seen["body"]["contents"][0]["parts"]
    audio = next(p for p in parts if "inlineData" in p)
    # Sniffed from the bytes, not taken from whatever the browser labelled it.
    assert audio["inlineData"]["mimeType"] == "audio/wav"


def test_playground_document_keeps_the_filename_the_user_picked():
    """Providers show a document's name to the model, so a picked PDF has
    to arrive under its own name rather than an invented one."""
    import base64

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "read it"}]}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
        )

    reg = Registry(
        [Target(provider="openai", key=CANARY, name="my-openai")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "summarise this",
                "files": [
                    {
                        "data_base64": base64.b64encode(PDF_BYTES).decode(),
                        "filename": "quarterly report.pdf",
                    }
                ],
            },
        )
    finally:
        reg.close()

    assert body["text"] == "read it"
    content = seen["body"]["input"][0]["content"]
    document = next(c for c in content if c["type"] == "input_file")
    assert document["filename"] == "quarterly report.pdf"


def test_a_sound_file_can_carry_a_turn_without_a_prompt():
    """The prompt is only required when nothing else carries the turn, and
    an attachment of any kind counts, not images alone."""
    import base64

    reg = make_registry()
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "",
                "files": [{"data_base64": base64.b64encode(PDF_BYTES).decode()}],
            },
        )
    finally:
        reg.close()
    assert "error" not in body or body["error"]["code"] != "bad_request"


def test_malformed_sound_and_document_payloads_are_named_bad_requests():
    """The error says which attachment and which index, so a person can
    tell a broken PDF from a broken recording."""
    reg = make_registry()
    try:
        both = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "hi",
                "audio": [{"url": "https://x/a.wav", "data_base64": "AAAA"}],
            },
        )
        junk = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "hi",
                "files": [{"data_base64": "not base64 at all!!"}],
            },
        )
    finally:
        reg.close()
    assert both["error"]["code"] == "bad_request"
    assert "audio 0" in both["error"]["message"]
    assert junk["error"]["code"] == "bad_request"
    assert "file 0" in junk["error"]["message"]


def test_browsed_models_lead_with_the_one_the_walk_would_try_first():
    """The Playground picks whichever model the browser lists first, so
    listing them in raw provider order meant defaulting to Gemini's first
    advertised model — one the walk already skips. A first generation then
    failed on a key that works. Ordering here is the same rule verify uses,
    imported rather than repeated."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": f"models/{name}",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                    # Provider order: the withdrawn one first, the maintained
                    # alias buried, exactly as Gemini serves it.
                    for name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")
                ]
            },
        )

    reg = Registry(
        [Target(provider="gemini", key=CANARY, name="my-gemini")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = browse_models(reg, 0, category="text_generation", refresh=True)
    finally:
        reg.close()

    listed = [m["id"] for m in body["models"]]
    assert listed[0] == "gemini-flash-latest", (
        "the picker defaults to the first entry, so it must be the candidate "
        f"the walk would try first; got {listed}"
    )
    # Nothing is dropped: a model the walk deprioritises is still selectable.
    assert set(listed) == {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"}


def test_targets_tell_the_browser_what_each_key_can_accept():
    """The Playground disables an attachment the selected key can never
    satisfy. That gate is only honest if it reads the same catalog the
    adapters gate on, so the shape is asserted per provider."""
    reg = Registry(
        [
            Target(provider="openai", key=CANARY, name="my-openai"),
            Target(provider="gemini", key=CANARY + "-2", name="my-gemini"),
            Target(provider="deepseek", key=CANARY + "-3", name="my-deepseek"),
        ],
        httpx_transport=httpx.MockTransport(openai_handler),
    )
    try:
        body = list_targets(reg)
    finally:
        reg.close()

    accepts = {t["provider"]: t["accepts"] for t in body["targets"]}
    # OpenAI reads pictures and documents but not sound.
    assert accepts["openai"]["image"] == {"bytes": True, "url": True}
    assert accepts["openai"]["file"] == {"bytes": True, "url": False}
    assert accepts["openai"]["audio"] == {"bytes": False, "url": False}
    # Gemini is the only one that listens, and only to bytes.
    assert accepts["gemini"]["audio"] == {"bytes": True, "url": False}
    # DeepSeek takes no attachment of any kind.
    assert all(form == {"bytes": False, "url": False} for form in accepts["deepseek"].values())
    # And the suggestion of where to turn instead names valid providers.
    assert body["providers_accepting"]["audio"] == ["gemini"]
    assert "deepseek" not in body["providers_accepting"]["image"]
    assert CANARY not in json.dumps(body)
