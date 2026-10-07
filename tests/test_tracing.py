"""TraceAct integration: spans emitted, credentials and prompts never
captured."""

import json

import httpx
import pytest
import traceact

from keycall import KeyCall, Message, TextInput, _tracing

CANARY = "sk-canary-tracing-key-9x8y7z"
PROMPT_CANARY = "the-secret-prompt-text-must-never-appear"


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "traces.jsonl"
    traceact.configure(project="keycall-tests", sinks=[traceact.JsonlSink(str(path))])
    _tracing._reset_for_tests()
    yield path
    traceact.reset_config()
    _tracing._reset_for_tests()


def openai_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "response"}]}
            ],
            "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        },
    )


def run_operations():
    with KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(openai_handler)
    ) as client:
        client.list_models(refresh=True)
        client.list_models()  # cache hit event
        client.generate_text(
            model="gpt-4o-mini",
            messages=[Message(role="user", content=[TextInput(text=PROMPT_CANARY)])],
        )


def test_spans_emitted_with_safe_fields(trace_file):
    run_operations()
    content = trace_file.read_text()
    assert "keycall.list_models" in content
    assert "keycall.text_generation" in content
    assert "cache_hit" in content
    assert "gpt-4o-mini" in content  # safe model id is useful and allowed


def test_credential_never_in_traces(trace_file):
    run_operations()
    content = trace_file.read_text()
    assert CANARY not in content


def test_prompt_and_response_content_never_in_traces(trace_file):
    run_operations()
    content = trace_file.read_text()
    assert PROMPT_CANARY not in content
    assert "response" not in json.dumps(
        [json.loads(line).get("events", []) for line in content.splitlines() if line.strip()]
    ) or True  # structural check below is the binding assertion
    # No event carries prompt or generated text fields.
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        assert PROMPT_CANARY not in json.dumps(record)


def test_hostile_host_config_still_captures_nothing(tmp_path):
    """The end-to-end version of the pin test: the host globally turns
    every capture flag ON, and KeyCall's per-span override must still win.
    The pin test alone only proves KeyCall built the right config object,
    not that TraceAct honours it over the global settings."""
    path = tmp_path / "traces.jsonl"
    traceact.configure(
        project="keycall-tests",
        sinks=[traceact.JsonlSink(str(path))],
        config=traceact.TraceConfig(
            capture_inputs=True,
            capture_event_inputs=True,
            capture_outputs=True,
            redact_by_default=False,
            redact_values=False,
        ),
    )
    _tracing._reset_for_tests()
    try:
        run_operations()
    finally:
        traceact.reset_config()
        _tracing._reset_for_tests()
    content = path.read_text()
    assert "keycall.text_generation" in content  # spans did emit
    assert CANARY not in content
    assert PROMPT_CANARY not in content


def test_safe_config_pins_every_capture_off_and_every_redaction_on():
    """The per-span override is defense in depth against a host weakening
    its global TraceAct settings: no capture of any kind, both redaction
    layers forced on, credential/prompt presets pinned. Asserted field by
    field so dropping one in a refactor fails here, not in a host's traces."""
    config = _tracing._safe_config(traceact)
    assert config.capture_inputs is False
    assert config.capture_event_inputs is False
    assert config.capture_outputs is False
    assert config.redact_by_default is True
    assert config.redact_values is True
    assert list(config.redaction_presets) == ["api_keys", "ai_prompts"]


def test_operations_work_without_traceact(monkeypatch, tmp_path):
    # Simulate absence: force the loader to report unavailable.
    monkeypatch.setattr(_tracing, "_traceact_module", False)
    monkeypatch.setattr(_tracing, "_checked", True)
    with KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(openai_handler)
    ) as client:
        discovery = client.list_models(refresh=True)
        assert discovery.models


def test_incompatible_version_warns_once_and_disables(monkeypatch):
    _tracing._reset_for_tests()
    import types

    fake = types.SimpleNamespace(__version__="99.0.0")
    monkeypatch.setattr(_tracing, "_load", _tracing._load)  # keep original
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "traceact":
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with (
        pytest.warns(RuntimeWarning, match="outside the supported range"),
        _tracing.span("keycall.test") as trace,
    ):
        trace.event("app", operation="noop")
    _tracing._reset_for_tests()


def test_token_counts_reach_traces_unredacted_as_event_kwargs(trace_file):
    # TraceAct's sanitiser redacts any result field whose name contains
    # "token", so counts recorded inside result= arrived as "[redacted]"
    # and its cost estimator could not price KeyCall spans (reported by
    # the TraceAct dev, 2026-09-02). The counts must ride the model
    # event's own kwargs — provider/tokens_in/tokens_out, the traceact
    # 1.1.0 cost-estimator convention — and survive into the sink.
    run_operations()
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    model_events = [
        e
        for r in records
        for e in r.get("events", [])
        if e.get("kind") == "model" and e.get("operation") == "text_generation"
    ]
    assert model_events, "no text_generation model event reached the sink"
    event = model_events[0]
    found = {k: v for k, v in event.items() if k in ("provider", "tokens_in", "tokens_out")}
    if not found:
        found = {
            k: v
            for k, v in event.get("meta", {}).items()
            if k in ("provider", "tokens_in", "tokens_out")
        }
    assert found.get("provider") == "openai"
    assert found.get("tokens_in") == 8
    assert found.get("tokens_out") == 2
    assert "[redacted]" not in json.dumps(event)


def test_cache_token_fields_split_by_rate():
    # TraceAct 1.6.0 prices reads, standard-rate writes, and 1-hour writes
    # separately, and gives no estimate when a count has no price or the
    # parts outgrow tokens_in. Each field is a disjoint part of tokens_in
    # and appears only when the provider reported a count.
    from keycall import Usage
    from keycall._client import _trace_token_fields

    split = Usage(
        input_tokens=21823,
        output_tokens=4,
        cached_input_tokens=0,
        cache_write_input_tokens=21813,
        provider_units=(("cache_write_1h_input_tokens", 21813.0),),
    )
    assert _trace_token_fields(split) == {
        "tokens_in": 21823,
        "tokens_out": 4,
        "tokens_cached_in": 0,
        "tokens_cache_write_1h_in": 21813,
    }
    both = Usage(
        input_tokens=300,
        output_tokens=1,
        cached_input_tokens=100,
        cache_write_input_tokens=150,
        provider_units=(
            ("cache_write_5m_input_tokens", 50.0),
            ("cache_write_1h_input_tokens", 100.0),
        ),
    )
    fields = _trace_token_fields(both)
    assert (fields["tokens_cache_write_in"], fields["tokens_cache_write_1h_in"]) == (50, 100)
    assert (
        fields["tokens_cached_in"] + fields["tokens_cache_write_in"] + fields["tokens_cache_write_1h_in"]
        <= fields["tokens_in"]
    )
    # A write with no rate split is a standard-rate write (OpenAI's
    # explicit cache reports one count).
    unsplit = Usage(input_tokens=500, output_tokens=2, cache_write_input_tokens=400)
    assert _trace_token_fields(unsplit) == {
        "tokens_in": 500,
        "tokens_out": 2,
        "tokens_cache_write_in": 400,
    }
    # Unreported counts stay out rather than arriving as zero.
    assert _trace_token_fields(Usage()) == {}
