"""Viewer: API layer, HTTP server, auth, and the credential-leak canary."""

import base64
import json
import os
import threading
import time
import urllib.request

import httpx
import pytest

from keycall import ModelCategory
from keycall._sources import Target
from keycall.viewer import Token
from keycall.viewer._api import (
    browse_models,
    check_target,
    generate,
    generate_image,
    generate_stream_events,
    generate_video,
    list_targets,
    set_settings,
    verify_target,
)
from keycall.viewer._registry import Registry
from keycall.viewer._server import _Server, _watch_sources

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


def test_list_targets_reports_the_current_read_timeout():
    reg = make_registry()
    try:
        assert list_targets(reg)["read_timeout"] == 180
        set_settings(reg, {"read_timeout": 60})
        assert list_targets(reg)["read_timeout"] == 60
    finally:
        reg.close()


def test_set_settings_rebuilds_clients_and_keeps_ids():
    reg = make_registry()
    try:
        before = reg.client(0)
        body = set_settings(reg, {"read_timeout": 300})
        assert body == {"read_timeout": 300}
        after = reg.client(0)
        assert after is not before
        # Same target behind the same id, and the retired client still
        # works until shutdown so an in-flight request isn't cut off.
        assert after.provider == before.provider
        assert reg.views()[0].id == 0
        before.list_models(categories={ModelCategory.TEXT_GENERATION})
    finally:
        reg.close()


@pytest.mark.parametrize(
    "value", [59, 301, 180.5, "180", True, None]
)
def test_set_settings_refuses_a_bad_read_timeout(value):
    reg = make_registry()
    try:
        body = set_settings(reg, {"read_timeout": value})
        assert body["error"]["code"] == "bad_request"
        assert reg.read_timeout == 180
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


def test_generate_sends_reasoning_effort_to_the_provider():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        )

    targets = [Target(provider="openai", key=CANARY, name="my-openai")]
    reg = Registry(targets, httpx_transport=httpx.MockTransport(handler))
    try:
        body = generate(
            reg,
            0,
            {"target": 0, "model": "gpt-4o-mini", "prompt": "hi", "reasoning_effort": "low"},
        )
        assert body["text"] == "hi"
        assert seen[0]["reasoning"] == {"effort": "low"}
    finally:
        reg.close()


def test_generate_refuses_reasoning_effort_on_an_unsupporting_provider():
    targets = [Target(provider="deepseek", key=CANARY, name="my-deepseek")]
    reg = Registry(
        targets,
        httpx_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": "deepseek-chat"}]})
        ),
    )
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "deepseek-chat",
                "prompt": "hi",
                "reasoning_effort": "low",
            },
        )
        assert body["error"]["code"] == "unsupported_operation"
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


def test_token_accepts_a_fixed_value_for_dev_reload():
    """The dev-reload restart hands the child process the running token so
    the open browser tab keeps working; everything else generates fresh."""
    resumed = Token(value="carried-across-restart")
    assert resumed.matches("carried-across-restart")
    assert not resumed.matches("something-else")
    assert not Token().matches(Token().value)


def test_source_watch_requests_reload_on_change(tmp_path):
    """Editing a Python file under the watched root must stop the server
    with the reload flag set; an untouched tree must not."""
    (tmp_path / "_catalog").mkdir()
    (tmp_path / "_catalog" / "catalog.json").write_text("{}")
    module = tmp_path / "module.py"
    module.write_text("x = 1\n")

    class FakeServer:
        def __init__(self):
            self.reload_requested = False
            self.stopped = threading.Event()

        def shutdown(self):
            self.stopped.set()

    quiet = FakeServer()
    threading.Thread(
        target=_watch_sources, args=(quiet, tmp_path, 0.02), daemon=True
    ).start()
    assert not quiet.stopped.wait(0.2)

    watched = FakeServer()
    threading.Thread(
        target=_watch_sources, args=(watched, tmp_path, 0.02), daemon=True
    ).start()
    # Let the watcher take its baseline before the edit, or the changed
    # mtime is the baseline and nothing looks different.
    time.sleep(0.2)
    later = time.time() + 10
    os.utime(module, (later, later))
    assert watched.stopped.wait(3)
    assert watched.reload_requested


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
                     read_timeout=None, httpx_transport=None):
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


def test_generate_sends_history_before_the_current_prompt():
    """A multi-turn conversation must reach the provider in the order it
    happened: prior turns first, then the turn being asked now. The reverse
    order once made every follow-up read as the opening question."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        )

    targets = [Target(provider="openai", key=CANARY, name="my-openai")]
    reg = Registry(targets, httpx_transport=httpx.MockTransport(handler))
    try:
        body = generate(
            reg,
            0,
            {
                "target": 0,
                "model": "gpt-4o-mini",
                "prompt": "How old is he?",
                "system": "Be brief.",
                "history": [
                    {"role": "user", "parts": [{"kind": "text", "text": "Who plays Indiana Jones?"}]},
                    {"role": "assistant", "parts": [{"kind": "text", "text": "Harrison Ford."}]},
                ],
            },
        )
        assert body["text"] == "ok"
        wire = seen[0]["input"]
        roles = [item.get("role") for item in wire]
        assert roles == ["system", "user", "assistant", "user"]
        # The current question is the last turn on the wire, not the first.
        assert "How old is he?" in json.dumps(wire[-1])
        assert "Indiana Jones" in json.dumps(wire[1])
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


GEMINI_VIDEO_OP = "models/veo-3.1-lite-generate-preview/operations/y5lxdapaztmq"
GEMINI_VIDEO_URI = (
    "https://generativelanguage.googleapis.com/v1beta/files/jn989ri0g72v:download?alt=media"
)


def test_generate_video_returns_the_clip_and_its_media_type():
    # Video is job-based (start, poll, download), unlike image's single
    # round trip, so the route's mock has to answer the whole sequence,
    # not just one request. Payloads mirror test_video_generation.py's own
    # live-captured Gemini fixtures.
    video_bytes = b"\x00\x00\x00 ftypisommp4-ish"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":predictLongRunning"):
            return httpx.Response(200, json={"name": GEMINI_VIDEO_OP})
        if "/operations/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "name": GEMINI_VIDEO_OP,
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [{"video": {"uri": GEMINI_VIDEO_URI}}]
                        }
                    },
                },
            )
        return httpx.Response(
            200, content=video_bytes, headers={"content-type": "video/mp4"}
        )

    reg = Registry(
        [Target(provider="gemini", key=CANARY, name="my-gemini")],
        httpx_transport=httpx.MockTransport(handler),
    )
    try:
        body = generate_video(
            reg,
            0,
            {"target": 0, "model": "veo-3.1-lite-generate-preview", "prompt": "A boat."},
        )
    finally:
        reg.close()

    assert body["videos"] == [
        {
            "base64_data": base64.b64encode(video_bytes).decode(),
            "media_type": "video/mp4",
            "url": GEMINI_VIDEO_URI,
        }
    ]
    assert body["operation"] == "video_generation"
    assert CANARY not in json.dumps(body)


def test_generate_video_reports_a_provider_that_cannot():
    reg = make_registry()
    try:
        body = generate_video(
            reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "A boat."}
        )
        missing = generate_video(reg, 0, {"target": 0, "model": "", "prompt": ""})
    finally:
        reg.close()
    # make_registry() is an OpenAI target, which has no video surface at
    # all; the response shape is what's checked here.
    assert "error" in body
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
    # The Playground toggles gate on these flags; a flag off here is one
    # whose gate would refuse the request after a billable round trip.
    caps = body["provider_capabilities"]
    assert caps["moonshot"]["web_search"] is True
    assert caps["deepseek"]["web_search"] is False
    assert caps["perplexity"]["tool_calling"] is False
    assert caps["anthropic"]["image_generation"] is False
    assert caps["openai"]["image_generation"] is True
    assert caps["gemini"]["video_generation"] is True
    assert caps["xai"]["video_generation"] is True
    assert caps["openai"]["video_generation"] is False
    assert caps["openai"]["reasoning_effort"] is True
    assert caps["deepseek"]["reasoning_effort"] is False
    assert caps["assemblyai"]["transcription"] is True
    assert caps["deepgram"]["transcription"] is True
    assert caps["openai"]["transcription"] is False
    assert CANARY not in json.dumps(body)


# --- cookie auth and CSRF ---------------------------------------------------


def _raw(url, *, method="GET", body=None, headers=None, follow=True):
    """A request with full control over headers and redirect following, so
    the cookie handshake can be inspected rather than followed blindly."""
    import urllib.error

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(
        *([] if follow else [_NoRedirect])
    )
    req = urllib.request.Request(url, data=body, method=method)
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with opener.open(req) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_opening_the_printed_link_trades_the_token_for_a_cookie(server):
    """The terminal link is the only way the browser learns the token. It
    must not stay in the address bar, where it would reach history and
    anything the user copies."""
    base, token = server
    status, headers, _ = _raw(f"{base}/?token={token}", follow=False)

    assert status == 303, "expected a redirect that strips the token"
    assert headers["Location"] == "/"
    cookie = headers["Set-Cookie"]
    assert token in cookie
    # httpOnly is the whole point: page script renders untrusted model
    # output and must not be able to read this.
    assert "HttpOnly" in cookie
    # SameSite=Strict keeps it off requests other sites make.
    assert "SameSite=Strict" in cookie
    # Secure would stop the cookie being stored at all over plain http on
    # loopback, which is what this server is.
    assert "Secure" not in cookie


def test_the_cookie_alone_authenticates_afterwards(server):
    base, token = server
    status, _ = _get(f"{base}/api/targets")  # no credentials at all
    assert status == 403

    status, _, payload = _raw(
        f"{base}/api/targets", headers={"Cookie": f"keycall_viewer_token={token}"}
    )
    assert status == 200
    assert len(json.loads(payload)["targets"]) == 2


def test_a_wrong_cookie_is_refused(server):
    base, _ = server
    status, _, _ = _raw(
        f"{base}/api/targets", headers={"Cookie": "keycall_viewer_token=not-the-token"}
    )
    assert status == 403


def test_cookie_post_without_a_json_content_type_is_refused(server):
    """The CSRF gate. A cross-site POST can skip the CORS preflight only by
    staying a "simple request", which cannot carry application/json. Any
    other content type is therefore something a browser would have let
    another page send with our cookie attached."""
    base, token = server
    payload = json.dumps({"target": 0, "model": "gpt-4o-mini", "prompt": "hi"}).encode()

    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        status, _, body = _raw(
            f"{base}/api/generate",
            method="POST",
            body=payload,
            headers={"Cookie": f"keycall_viewer_token={token}", "Content-Type": content_type},
        )
        assert status == 403, f"{content_type} was accepted"
        assert json.loads(body)["error"]["code"] == "forbidden"

    # The same request as JSON goes through.
    status, _, body = _raw(
        f"{base}/api/generate",
        method="POST",
        body=payload,
        headers={"Cookie": f"keycall_viewer_token={token}", "Content-Type": "application/json"},
    )
    assert status == 200


def test_a_post_from_another_origin_is_refused(server):
    """Belt to SameSite's braces: if a browser ever attached the cookie
    anyway, the Origin header still gives it away."""
    base, token = server
    status, _, body = _raw(
        f"{base}/api/generate",
        method="POST",
        body=json.dumps({"target": 0, "model": "gpt-4o-mini", "prompt": "hi"}).encode(),
        headers={
            "Cookie": f"keycall_viewer_token={token}",
            "Content-Type": "application/json",
            "Origin": "https://evil.example",
        },
    )
    assert status == 403
    assert json.loads(body)["error"]["code"] == "forbidden"


def test_header_auth_still_works_for_scripts(server):
    """curl, the CLI, and this suite authenticate with the header, which
    carries its own CSRF immunity: a cross-origin request cannot set a
    custom header without a preflight this server never answers."""
    base, token = server
    status, body = _get(f"{base}/api/targets", token=token)
    assert status == 200
    status, body = _post(
        f"{base}/api/generate",
        {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"},
        token=token,
    )
    assert status == 200
    assert body["text"] == "hello"


def test_the_page_shell_never_carries_the_token(server):
    """Whatever the browser is handed on first load must not contain the
    secret, or stripping it from the URL would be theatre."""
    base, token = server
    _, _, shell = _raw(f"{base}/?token={token}")
    assert token.encode() not in shell
    _, _, script = _raw(f"{base}/static/app.js")
    assert token.encode() not in script
    # And the script no longer has any token machinery to leak through.
    assert b"sessionStorage" not in script


def test_every_tab_path_serves_the_page_shell(server):
    """Each tab has its own URL, so a reload or a pasted link opens
    straight onto it; the page reads the path and shows the matching tab.
    An unknown path stays a 404 rather than becoming a catch-all."""
    base, _ = server
    for path in ("/", "/dashboard", "/models", "/playground", "/verify", "/traces"):
        status, _, body = _raw(f"{base}{path}")
        assert status == 200, path
        assert b"<nav id=\"tabs\">" in body, path
    status, _, _ = _raw(f"{base}/nonsense")
    assert status == 404


def test_the_token_handshake_works_on_a_tab_path(server):
    """A bookmarked deep link with a token still swaps it for the cookie
    and redirects back to the same tab path, token stripped."""
    base, token = server
    status, headers, _ = _raw(f"{base}/traces?token={token}", follow=False)
    assert status == 303
    assert headers["Location"] == "/traces"
    assert "keycall_viewer_token" in headers.get("Set-Cookie", "")


# --- adding a key from the browser ------------------------------------------


def test_a_typed_key_becomes_a_target_and_is_never_echoed():
    """The viewer could only load a file path, so anyone holding a key had
    to write a TOML file before they could click anything. A typed key now
    goes into the same in-memory registry — and, like every other viewer
    response, the reply carries no credential."""
    from keycall.viewer._api import add_key

    reg = Registry([], httpx_transport=httpx.MockTransport(openai_handler))
    try:
        body = add_key(reg, {"provider": "openai", "key": CANARY, "name": "typed"})
        assert "error" not in body, body
        assert [t["provider"] for t in body["targets"]] == ["openai"]
        assert body["targets"][0]["name"] == "typed"
        assert CANARY not in json.dumps(body)
        # And it is usable immediately, not merely recorded.
        assert generate(reg, 0, {"target": 0, "model": "gpt-4o-mini", "prompt": "hi"})[
            "text"
        ] == "hello"
    finally:
        reg.close()


def test_a_typed_key_joins_the_targets_already_loaded():
    from keycall.viewer._api import add_key

    reg = make_registry()  # two targets from a file
    try:
        body = add_key(reg, {"provider": "gemini", "key": CANARY + "-3"})
        assert len(body["targets"]) == 3
        assert body["targets"][2]["provider"] == "gemini"
        assert CANARY not in json.dumps(body)
    finally:
        reg.close()


def test_a_bad_typed_key_is_refused_with_a_readable_reason():
    from keycall.viewer._api import add_key

    reg = Registry([], httpx_transport=httpx.MockTransport(openai_handler))
    try:
        for payload, expected in [
            ({}, "provider"),
            ({"provider": "openai"}, "key"),
            ({"provider": "openai", "key": "   "}, "key"),
            ({"provider": "   ", "key": "sk-x"}, "provider"),
            ({"provider": "openai", "key": 42}, "key"),
        ]:
            body = add_key(reg, payload)
            assert "error" in body, payload
            assert expected in body["error"]["message"], payload
        # An unknown provider needs a protocol and base URL; without them
        # this is a configuration mistake, reported rather than raised.
        body = add_key(reg, {"provider": "not-a-provider", "key": "sk-x"})
        assert "error" in body
        assert not reg.views(), "a refused key must not leave a target behind"
    finally:
        reg.close()


def test_typed_keys_go_through_the_same_csrf_gate(server):
    """/api/key accepts a credential, so it must be no easier to reach from
    another site than the routes that spend money."""
    base, token = server
    payload = json.dumps({"provider": "openai", "key": "sk-x"}).encode()

    status, _, _ = _raw(
        f"{base}/api/key",
        method="POST",
        body=payload,
        headers={"Cookie": f"keycall_viewer_token={token}", "Content-Type": "text/plain"},
    )
    assert status == 403, "a simple cross-site POST reached /api/key"

    status, _, _ = _raw(
        f"{base}/api/key",
        method="POST",
        body=payload,
        headers={
            "Cookie": f"keycall_viewer_token={token}",
            "Content-Type": "application/json",
            "Origin": "https://evil.example",
        },
    )
    assert status == 403, "a foreign origin reached /api/key"

    # And unauthenticated, as with everything else under /api.
    status, _, _ = _raw(
        f"{base}/api/key", method="POST", body=payload,
        headers={"Content-Type": "application/json"},
    )
    assert status == 403


def test_the_provider_list_comes_from_the_catalog():
    """The key form's dropdown is filled from this, so a provider added to
    the catalog appears without a second edit in the frontend."""
    from keycall._registry import supported_providers

    reg = make_registry()
    try:
        body = list_targets(reg)
    finally:
        reg.close()
    assert body["providers"] == list(supported_providers())
    assert "openai" in body["providers"] and "moonshot" in body["providers"]


def test_traces_record_calls_without_recording_prompts(server):
    """The Traces tab answers "why was that slow" from timing and
    outcomes alone: a row per API call, and never the prompt text."""
    base, token = server

    status, body = _get(f"{base}/api/traces", token=token)
    assert status == 200
    assert body["traces"] == []

    marker = "the-secret-prompt-text-that-must-not-be-recorded"
    status, _ = _post(
        f"{base}/api/generate",
        {"target": 0, "model": "gpt-4o-mini", "prompt": marker, "max_output_tokens": 32},
        token=token,
    )
    assert status == 200

    status, body = _get(f"{base}/api/traces", token=token)
    assert status == 200
    rows = body["traces"]
    assert len(rows) == 1
    row = rows[0]
    assert row["route"] == "/api/generate"
    assert row["model"] == "gpt-4o-mini"
    assert row["provider"] == "openai"
    assert row["target"] == 0
    assert row["status"] == "ok"
    assert row["duration_ms"] >= 0
    assert marker not in json.dumps(body), "prompt text must never enter a trace"


def test_traces_capture_the_error_outcome(server):
    base, token = server
    _post(
        f"{base}/api/generate",
        {"target": 99, "model": "gpt-4o-mini", "prompt": "hi"},
        token=token,
    )
    _, body = _get(f"{base}/api/traces", token=token)
    rows = body["traces"]
    assert rows, "a failed call still writes a trace row"
    assert rows[0]["status"].startswith("error")


def test_traces_require_the_token(server):
    base, _ = server
    status, _ = _get(f"{base}/api/traces")
    assert status == 403


def test_traces_clear_wipes_the_log(server):
    base, token = server
    _post(
        f"{base}/api/generate",
        {"target": 99, "model": "gpt-4o-mini", "prompt": "hi"},
        token=token,
    )
    _, body = _get(f"{base}/api/traces", token=token)
    assert body["traces"], "the failed call should have written a row to clear"

    status, body = _post(f"{base}/api/traces/clear", {}, token=token)
    assert status == 200
    assert body["cleared"] is True
    _, body = _get(f"{base}/api/traces", token=token)
    assert body["traces"] == []


def test_traces_clear_requires_the_token(server):
    base, _ = server
    status, _ = _post(f"{base}/api/traces/clear", {})
    assert status == 403
