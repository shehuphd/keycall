"""Live provider smoke tests.

Deselected by default; select with `pytest -m live`. Credentials load only
when this module's test actually runs, from the target file named by
KEYCALL_LIVE_SOURCE, so ordinary runs never touch the environment for
key-like values.

Mode is a CI concern, not a test concern: `warn` runs this job with
continue-on-error, `strict` blocks the release on any failure. The test
itself distinguishes rate-limit outcomes (a verification-environment
failure: the release stays unverified, but the adapter is not implicated)
from adapter or credential failures.
"""

from __future__ import annotations

import os

import pytest

from keycall._sources import load_targets
from keycall._verify_core import run_verify

pytestmark = pytest.mark.live


def candidates(discovery, limit: int = 8):
    """Walk the provider's maintained aliases first, then its list order —
    the same rule run_verify uses. Without it a Gemini run spends most of
    its budget on models Google has already withdrawn (six of the first
    eight advertised, 2026-08-09), which is what made this suite flaky
    against Gemini rather than any adapter fault."""
    from keycall._verify_core import _is_maintained_alias

    ordered = sorted(discovery.models, key=lambda m: not _is_maintained_alias(m.id))
    return ordered[:limit]


def test_live_smoke_every_target_generates():
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    targets, _ = load_targets(source)

    verified: list[str] = []
    rate_limited: list[str] = []
    failed: list[str] = []
    for target in targets:
        result = run_verify(target, generate=True)
        summary = (
            f"{result.label} ({result.provider}): {result.outcome}, "
            f"digest {result.model_list_digest}, rule v{result.selection_rule_version}, "
            f"{len(result.attempts)} attempt(s)"
        )
        for attempt in result.attempts:
            summary += (
                f"\n    {'ok' if attempt.ok else attempt.error_code}: {attempt.model_id} "
                f"(filtered {attempt.position}, raw {attempt.raw_position}, "
                f"{attempt.classification_source})"
            )
        print(summary)
        if result.generate_ok:
            verified.append(result.label)
        elif result.outcome == "rate_limited_unverified":
            rate_limited.append(summary)
        else:
            failed.append(summary)

    report = []
    if failed:
        report.append("adapter/credential failures:\n" + "\n".join(failed))
    if rate_limited:
        report.append(
            "verification-environment failures (rate limited, provider not "
            "implicated, release still unverified):\n" + "\n".join(rate_limited)
        )
    assert not report, "\n\n".join(report)
    assert verified, "no targets were verified"


def test_live_stream_smoke_every_target():
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
        try:
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            attempt_errors = []
            for model in candidates(discovery):
                try:
                    with client.stream_text(
                        model=model.id,
                        messages=[
                            Message(role="user", content=[TextInput(text="Reply with the single word: ok")])
                        ],
                        max_output_tokens=16,
                    ) as stream:
                        deltas = sum(1 for e in stream if e.kind == "text_delta")
                        result = stream.result()
                    print(
                        f"{target.display_name}: streamed {model.id} "
                        f"({deltas} delta(s), finish {result.finish_reason}, "
                        f"usage {'reported' if result.usage.output_tokens is not None else 'missing'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(f"{target.display_name}: no model streamed\n" + "\n".join(attempt_errors))
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_tool_round_every_supporting_target():
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool, ToolResult
    from keycall._capabilities import TOOL_CALLING_PROVIDERS

    weather = Tool(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    ask = [Message(role="user", content=[
        TextInput(text="What's the weather in London right now? Use the tool."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in TOOL_CALLING_PROVIDERS:
            continue
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
        try:
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            attempt_errors = []
            for model in candidates(discovery):
                try:
                    first = client.generate_text(
                        model=model.id, messages=ask, tools=[weather],
                        max_output_tokens=300,
                    )
                    if not first.tool_calls:
                        attempt_errors.append(f"    {model.id}: no tool call made")
                        continue
                    call = first.tool_calls[0]
                    final = client.generate_text(
                        model=model.id,
                        messages=[
                            *ask,
                            first.to_assistant_message(),
                            Message(role="user", content=[
                                ToolResult(tool_call_id=call.id, name=call.name,
                                           content='{"temp_c": 14, "condition": "rainy"}'),
                            ]),
                        ],
                        tools=[weather],
                        max_output_tokens=300,
                    )
                    print(
                        f"{target.display_name}: tool round on {model.id} "
                        f"({len(first.tool_calls)} call(s), args {dict(call.arguments)}, "
                        f"final text {'yes' if final.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a tool round\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streamed_tool_call_every_supporting_target():
    """The streamed argument shapes are provider-specific and undocumented
    enough to be worth re-verifying every release: a provider that changes
    how it splits arguments would otherwise surface as calls with silently
    empty arguments."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool
    from keycall._capabilities import TOOL_CALLING_PROVIDERS

    weather = Tool(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    ask = [Message(role="user", content=[
        TextInput(text="What's the weather in London right now? Use the tool."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in TOOL_CALLING_PROVIDERS:
            continue
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
        try:
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            attempt_errors = []
            for model in candidates(discovery):
                try:
                    started, fragments = [], []
                    with client.stream_text(
                        model=model.id, messages=ask, tools=[weather],
                        max_output_tokens=300,
                    ) as stream:
                        for event in stream:
                            if event.kind == "tool_call_started":
                                started.append(event.name)
                            elif event.kind == "tool_call_arguments_delta":
                                fragments.append(event.fragment)
                        result = stream.result()
                    if not result.tool_calls:
                        attempt_errors.append(f"    {model.id}: no tool call streamed")
                        continue
                    call = result.tool_calls[0]
                    assert started, f"{model.id}: completed a call with no start event"
                    assert call.arguments, (
                        f"{model.id}: streamed call has empty arguments — the "
                        "provider's argument-fragment shape may have changed"
                    )
                    print(
                        f"{target.display_name}: streamed tool call on {model.id} "
                        f"({len(result.tool_calls)} call(s), {len(fragments)} fragment(s), "
                        f"args {dict(call.arguments)})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a streamed tool call\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_perplexity_tools_gate_still_correct():
    """Capability-drift probe: the Perplexity gate rests on live evidence
    that Sonar rejects tools. If this call stops failing with the known
    rejection, the gate is stale and TOOL_CALLING_PROVIDERS needs updating —
    this test failing IS the notification."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "perplexity"), None)
    if target is None:
        pytest.skip("no perplexity target in the live source")
    response = httpx.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {target.key}"},
        json={
            "model": "sonar",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "noop", "description": "does nothing",
                "parameters": {"type": "object", "properties": {}},
            }}],
        },
        timeout=30,
    )
    assert response.status_code == 400 and "not supported" in response.text.lower(), (
        f"capability drift: perplexity tools returned HTTP {response.status_code} "
        "instead of the known 'not supported' rejection — re-probe and update "
        "TOOL_CALLING_PROVIDERS and this test"
    )
    print("perplexity: tools still rejected (gate evidence current)")
