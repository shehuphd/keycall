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

from keycall._errors import KeyCallError
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
    no_models: list[str] = []
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
        elif result.outcome == "no_text_models" and result.listed_ok:
            # The credential verified: listing succeeded, the account just
            # advertises nothing to invoke. Tinker's OpenAI-compatible
            # endpoint serves your own fine-tuned checkpoints, so an empty
            # model list is its normal state rather than a fault. Reported,
            # not counted as an adapter or credential failure.
            no_models.append(summary)
        else:
            failed.append(summary)

    for summary in no_models:
        print(f"credential valid, no models advertised: {summary}")

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
        try:
            client = KeyCall(
                provider=target.provider,
                api_key=target.key,
                protocol=target.protocol,
                base_url=target.base_url,
            )
        except KeyCallError as exc:
            failures.append(f"{target.display_name}: unusable target — {exc}")
            continue
        try:
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            if not discovery.models:
                # Credential verified by the listing; nothing to stream.
                print(f"{target.display_name}: no models advertised, nothing to stream")
                continue
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


# Solid blue 8x8 PNG, built inline so the suite carries no binary fixture.
def _blue_png() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    size = 8
    raw = b"".join(b"\x00" + bytes((0, 102, 204)) * size for _ in range(size))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_live_image_input_every_supporting_target():
    """Image wire shapes differ per provider and are easy to get subtly
    wrong (a mislabelled media type is a 400 on two of them), so each
    release re-checks that a real image round-trips."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import ImageInput, KeyCall, Message, ModelCategory, TextInput
    from keycall._registry import providers_with

    supporting = providers_with("image_input")
    ask = [
        Message(
            role="user",
            content=[
                TextInput(text="What colour is this image? Answer with one word."),
                ImageInput(data=_blue_png()),
            ],
        )
    ]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in supporting:
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
            for model in candidates(discovery, 6):
                try:
                    result = client.generate_text(
                        model=model.id, messages=ask, max_output_tokens=200
                    )
                    answer = (result.text or "").strip().lower()
                    assert "blue" in answer, (
                        f"{model.id} read the image as {answer[:40]!r}; the bytes may "
                        "be reaching the provider in the wrong shape"
                    )
                    print(f"{target.display_name}: image read by {model.id} -> {answer[:20]!r}")
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model read the image\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)
