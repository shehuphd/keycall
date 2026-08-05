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
    list_targets,
    verify_target,
)
from keycall.viewer._registry import Registry
from keycall.viewer._server import _Server

CANARY = "sk-canary-viewer-key-do-not-leak"


def openai_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={"data": [{"id": "gpt-4o-mini"}, {"id": "text-embedding-3-small"}]},
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
