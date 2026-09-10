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
from keycall._registry import supported_service_providers
from keycall._sources import load_targets
from keycall._verify_core import run_verify

pytestmark = pytest.mark.live


def candidates(discovery, limit: int = 8):
    """The production candidate order, imported rather than reimplemented so
    this suite exercises the rule users get. Without an order the walk
    spends its budget on withdrawn models, which is what made this suite
    look flaky against Gemini rather than any adapter fault."""
    from keycall._verify_core import order_candidates

    return order_candidates(discovery.models)[:limit]


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
        elif result.outcome == "services_probed" and result.listed_ok:
            # A service key's verification is its category probes, which ran
            # live inside run_verify; print the standings so the log shows
            # what each category reported.
            for service in result.services:
                print(f"    {service.name}: {service.status}")
            verified.append(result.label)
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
    service_names = set(supported_service_providers())
    failures = []
    for target in targets:
        if target.provider in service_names:
            # A service key has no models to stream; its own drift probes
            # cover it.
            continue
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


def test_live_apply_patch_round_every_supporting_target():
    """apply_patch is a provider-owned tool with a fixed schema, not a
    caller-defined one — this exercises the whole round (call, reply,
    final text) the same way test_live_tool_round does for ordinary
    function calls, so a provider-side shape change surfaces here instead
    of silently breaking the ToolCall/ToolResult mapping. Never writes to
    disk: the reply is a canned confirmation, not an executed patch."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, ToolResult
    from keycall._capabilities import APPLY_PATCH_PROVIDERS

    ask = [Message(role="user", content=[
        TextInput(text="Use apply_patch to create hello.py containing only: print(1)"),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in APPLY_PATCH_PROVIDERS:
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
                        model=model.id, messages=ask, apply_patch=True,
                        max_output_tokens=300,
                    )
                    if not first.tool_calls:
                        attempt_errors.append(f"    {model.id}: no tool call made")
                        continue
                    call = first.tool_calls[0]
                    assert call.name == "apply_patch", (
                        f"{model.id}: expected an apply_patch call, got {call.name!r}"
                    )
                    assert call.arguments.get("type") == "create_file", (
                        f"{model.id}: expected a create_file operation, got "
                        f"{call.arguments.get('type')!r}"
                    )
                    final = client.generate_text(
                        model=model.id,
                        messages=[
                            *ask,
                            first.to_assistant_message(),
                            Message(role="user", content=[
                                ToolResult(tool_call_id=call.id, name=call.name,
                                           content={"status": "completed",
                                                    "output": "hello.py created"}),
                            ]),
                        ],
                        apply_patch=True,
                        max_output_tokens=300,
                    )
                    print(
                        f"{target.display_name}: apply_patch round on {model.id} "
                        f"(operation {dict(call.arguments)}, "
                        f"final text {'yes' if final.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed an apply_patch round\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streamed_apply_patch_call_every_supporting_target():
    """apply_patch streams through a dedicated event pair
    (response.apply_patch_call_operation_diff.delta/.done) rather than the
    function-call arguments events — undocumented enough, and different
    enough for delete_file (no diff at all, completes straight from
    response.output_item.done), to deserve its own live check."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput
    from keycall._capabilities import APPLY_PATCH_PROVIDERS

    ask = [Message(role="user", content=[
        TextInput(text="Use apply_patch to create hello.py containing only: print(1)"),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in APPLY_PATCH_PROVIDERS:
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
                        model=model.id, messages=ask, apply_patch=True,
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
                    assert call.name == "apply_patch", (
                        f"{model.id}: expected an apply_patch call, got {call.name!r}"
                    )
                    assert started == ["apply_patch"], (
                        f"{model.id}: expected one apply_patch start event, got {started}"
                    )
                    assert call.arguments.get("diff"), (
                        f"{model.id}: streamed call has no diff — the operation-diff "
                        "delta/done event shape may have changed"
                    )
                    print(
                        f"{target.display_name}: streamed apply_patch call on {model.id} "
                        f"({len(fragments)} diff fragment(s), operation {dict(call.arguments)})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a streamed apply_patch call\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_code_interpreter_every_supporting_target():
    """code_interpreter is provider-run, not caller-run — unlike
    apply_patch, there is no reply round; the model runs code and the
    code/output pair comes straight back in one call. Exercises all four
    supporting providers, whose wire forms diverge more than any other
    normalized tool: OpenAI and xAI report a null-ish outputs field with
    the human answer only in the following text, Gemini pairs
    executableCode/codeExecutionResult directly, and Anthropic maps onto
    an internal bash_code_execution server tool."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput
    from keycall._capabilities import CODE_INTERPRETER_PROVIDERS

    ask = [Message(role="user", content=[
        TextInput(text="Use the code interpreter to compute 17 * 23 and tell me the result."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in CODE_INTERPRETER_PROVIDERS:
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
                    result = client.generate_text(
                        model=model.id, messages=ask, code_interpreter=True,
                        max_output_tokens=500,
                    )
                    if not result.code_executions:
                        attempt_errors.append(f"    {model.id}: no code execution ran")
                        continue
                    execution = result.code_executions[0]
                    assert "17" in execution.code and "23" in execution.code, (
                        f"{model.id}: expected the code to reference the operands, got "
                        f"{execution.code!r}"
                    )
                    print(
                        f"{target.display_name}: code_interpreter round on {model.id} "
                        f"(code {execution.code!r}, output {execution.output!r}, "
                        f"final text {'yes' if result.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a code_interpreter round\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streamed_code_interpreter_every_supporting_target():
    """Streaming coverage for OpenAI, Gemini, and xAI — Anthropic is
    excluded deliberately: its streaming shape for bash_code_execution has
    not been live-probed, so a streamed Anthropic code_interpreter call
    currently completes with the right final text but silently drops its
    code/output (known limitation, catalog.json anthropic.code_interpreter_note)."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput
    from keycall._capabilities import CODE_INTERPRETER_PROVIDERS

    ask = [Message(role="user", content=[
        TextInput(text="Use the code interpreter to compute 17 * 23 and tell me the result."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in CODE_INTERPRETER_PROVIDERS or target.provider == "anthropic":
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
                    with client.stream_text(
                        model=model.id, messages=ask, code_interpreter=True,
                        max_output_tokens=500,
                    ) as stream:
                        events = list(stream)
                        result = stream.result()
                    unknowns = [e for e in events if e.kind == "unknown"]
                    assert not unknowns, (
                        f"{model.id}: streamed code_interpreter emitted unrecognized "
                        f"events {[e.provider_kind for e in unknowns]}"
                    )
                    if not result.code_executions:
                        attempt_errors.append(f"    {model.id}: no code execution streamed")
                        continue
                    print(
                        f"{target.display_name}: streamed code_interpreter on {model.id} "
                        f"({len(result.code_executions)} execution(s), "
                        f"final text {'yes' if result.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a streamed code_interpreter "
                    "call\n" + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_custom_tool_round_every_supporting_target():
    """A custom (freeform) tool has no JSON Schema — the model's call
    arrives as a plain string rather than parsed arguments. This exercises
    the whole round (call, reply, final text) the same way
    test_live_apply_patch_round does, so a change to the plain-string
    convention surfaces here instead of silently breaking
    ToolCall.arguments["input"]."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool, ToolResult
    from keycall._capabilities import CUSTOM_TOOL_PROVIDERS

    write_poem = Tool(name="write_poem", description="Records a poem", input_schema=None)
    ask = [Message(role="user", content=[
        TextInput(text="Use the write_poem tool to write a two-line poem about the moon. "
                        "Call the tool, don't just answer in text."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in CUSTOM_TOOL_PROVIDERS:
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
                        model=model.id, messages=ask, tools=[write_poem],
                        max_output_tokens=300,
                    )
                    if not first.tool_calls:
                        attempt_errors.append(f"    {model.id}: no tool call made")
                        continue
                    call = first.tool_calls[0]
                    assert call.name == "write_poem", (
                        f"{model.id}: expected a write_poem call, got {call.name!r}"
                    )
                    assert isinstance(call.arguments.get("input"), str) and call.arguments[
                        "input"
                    ], (
                        f"{model.id}: expected a non-empty plain-string input, got "
                        f"{call.arguments!r}"
                    )
                    final = client.generate_text(
                        model=model.id,
                        messages=[
                            *ask,
                            first.to_assistant_message(),
                            Message(role="user", content=[
                                ToolResult(tool_call_id=call.id, name=call.name,
                                           content="Recorded."),
                            ]),
                        ],
                        tools=[write_poem],
                        max_output_tokens=300,
                    )
                    print(
                        f"{target.display_name}: custom tool round on {model.id} "
                        f"(input {call.arguments['input']!r}, "
                        f"final text {'yes' if final.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a custom tool round\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streamed_custom_tool_call_every_supporting_target():
    """custom_tool_call_input streams through a dedicated event pair
    (response.custom_tool_call_input.delta/.done), same pattern as
    apply_patch's dedicated diff-delta events."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool
    from keycall._capabilities import CUSTOM_TOOL_PROVIDERS

    write_poem = Tool(name="write_poem", description="Records a poem", input_schema=None)
    ask = [Message(role="user", content=[
        TextInput(text="Use the write_poem tool to write a two-line poem about the sun. "
                        "Call the tool, don't just answer in text."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in CUSTOM_TOOL_PROVIDERS:
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
                        model=model.id, messages=ask, tools=[write_poem],
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
                    assert call.name == "write_poem", (
                        f"{model.id}: expected a write_poem call, got {call.name!r}"
                    )
                    assert started == ["write_poem"], (
                        f"{model.id}: expected one write_poem start event, got {started}"
                    )
                    assert call.arguments.get("input"), (
                        f"{model.id}: streamed call has no input — the "
                        "custom_tool_call_input delta/done event shape may have changed"
                    )
                    print(
                        f"{target.display_name}: streamed custom tool call on {model.id} "
                        f"({len(fragments)} input fragment(s))"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a streamed custom tool call\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_tool_search_round_every_supporting_target():
    """defer_loading=True is a request-size optimization, not a behavior
    change: the discovered tool's call and reply are ordinary
    ToolCall/ToolResult parts, identical to a non-deferred tool's. This
    exercises the whole round the same way test_live_apply_patch_round
    does, so a change to either provider's tool-search convention surfaces
    here instead of silently breaking the deferred-tool round trip."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool, ToolResult
    from keycall._capabilities import TOOL_SEARCH_PROVIDERS

    weather = Tool(
        name="get_weather",
        description="Get the current weather at a specific location",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        defer_loading=True,
    )
    ask = [Message(role="user", content=[
        TextInput(text="What is the weather in San Francisco? Use the get_weather tool."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in TOOL_SEARCH_PROVIDERS:
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
                    assert call.name == "get_weather", (
                        f"{model.id}: expected a get_weather call, got {call.name!r}"
                    )
                    assert not [p for p in first.parts if p.kind == "unknown"], (
                        f"{model.id}: a tool-search trace leaked through as UnknownOutput"
                    )
                    final = client.generate_text(
                        model=model.id,
                        messages=[
                            *ask,
                            first.to_assistant_message(),
                            Message(role="user", content=[
                                ToolResult(tool_call_id=call.id, name=call.name,
                                           content="68F, sunny"),
                            ]),
                        ],
                        tools=[weather],
                        max_output_tokens=300,
                    )
                    print(
                        f"{target.display_name}: tool search round on {model.id} "
                        f"(args {dict(call.arguments)}, "
                        f"final text {'yes' if final.text else 'NO'})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a tool search round\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streamed_tool_search_every_supporting_target():
    """Streaming coverage: tool_search_call/tool_search_output (OpenAI) and
    server_tool_use/tool_search_tool_result (Anthropic) stream as
    already-handled event forms — no dedicated events of their own — so
    this confirms neither leaks an unknown stream event."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput, Tool
    from keycall._capabilities import TOOL_SEARCH_PROVIDERS

    weather = Tool(
        name="get_weather",
        description="Get the current weather at a specific location",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        defer_loading=True,
    )
    ask = [Message(role="user", content=[
        TextInput(text="What is the weather in Tokyo? Use the get_weather tool."),
    ])]

    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in TOOL_SEARCH_PROVIDERS:
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
                    with client.stream_text(
                        model=model.id, messages=ask, tools=[weather],
                        max_output_tokens=300,
                    ) as stream:
                        events = list(stream)
                        result = stream.result()
                    unknowns = [e for e in events if e.kind == "unknown"]
                    assert not unknowns, (
                        f"{model.id}: streamed tool search emitted unrecognized "
                        f"events {[e.provider_kind for e in unknowns]}"
                    )
                    if not result.tool_calls:
                        attempt_errors.append(f"    {model.id}: no tool call streamed")
                        continue
                    print(
                        f"{target.display_name}: streamed tool search on {model.id} "
                        f"(call {result.tool_calls[0].name!r})"
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    attempt_errors.append(f"    {model.id}: {exc}")
            else:
                failures.append(
                    f"{target.display_name}: no model completed a streamed tool search call\n"
                    + "\n".join(attempt_errors)
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_streaming_transcription_every_supporting_target():
    """A whole transcription round against each STT provider: spoken-word
    PCM in, interims, finals with word timings, and the session summary's
    billable-audio duration out. Audio is synthesized on the fly with
    macOS `say`, so the expected words are known and the assertion is on
    recognition content, not just frame plumbing."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import shutil
    import subprocess
    import tempfile
    import threading
    import time as _time

    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("live transcription check needs `say` and `ffmpeg` to synthesize audio")
    from keycall import KeyCall
    from keycall._capabilities import STREAMING_TRANSCRIPTION_PROVIDERS

    with tempfile.TemporaryDirectory() as tmp:
        aiff = f"{tmp}/speech.aiff"
        pcm_path = f"{tmp}/speech.pcm"
        subprocess.run(
            ["say", "-o", aiff, "The quick brown fox jumps over the lazy dog."],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", "-f", "s16le", pcm_path],
            check=True,
            capture_output=True,
        )
        with open(pcm_path, "rb") as f:
            pcm = f.read()

    preferred_model = {"deepgram": "nova-3"}  # its no-model default is a dated base model
    targets, _ = load_targets(source)
    failures = []
    for target in targets:
        if target.provider not in STREAMING_TRANSCRIPTION_PROVIDERS:
            continue
        client = KeyCall(provider=target.provider, api_key=target.key)
        try:
            with client.transcribe_stream(
                model=preferred_model.get(target.provider), sample_rate=16000
            ) as session:

                def feed(feed_session=session):
                    chunk = 3200  # 100 ms of 16 kHz 16-bit mono
                    for i in range(0, len(pcm), chunk):
                        feed_session.send_audio(pcm[i : i + chunk])
                        _time.sleep(0.05)
                    _time.sleep(1.5)
                    feed_session.finish()

                feeder = threading.Thread(target=feed)
                feeder.start()
                finals, unknowns = [], []
                ended = None
                for event in session.events(timeout=30):
                    if event.kind == "final_transcript":
                        finals.append(event)
                    elif event.kind == "unknown":
                        unknowns.append(event.provider_kind)
                    elif event.kind == "session_ended":
                        ended = event
                feeder.join()
            text = " ".join(f.text for f in finals).lower()
            assert "fox" in text and "dog" in text, (
                f"{target.display_name}: transcription missed the spoken words, got {text!r}"
            )
            assert not unknowns, (
                f"{target.display_name}: unrecognized frames {unknowns}"
            )
            assert finals and finals[0].words, (
                f"{target.display_name}: final transcript carries no word timings"
            )
            assert ended is not None, f"{target.display_name}: session never ended"
            if target.provider == "elevenlabs":
                # ElevenLabs sends no billed-duration frame (raw-verified
                # 2026-08-31); a reported duration here would be invented.
                assert ended.audio_duration_seconds is None, (
                    f"{target.display_name}: started reporting a duration — "
                    "update the catalog note and this test"
                )
            else:
                assert ended.audio_duration_seconds, (
                    f"{target.display_name}: session summary reported no billable duration"
                )
            print(
                f"{target.display_name}: transcribed {text!r} "
                f"({len(finals)} final(s), {ended.audio_duration_seconds}s billed)"
            )
        except Exception as exc:  # noqa: BLE001 — reported, not hidden
            failures.append(f"{target.display_name}: {exc}")
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


def test_live_moonshot_search_still_returns_no_citations():
    """Capability-drift probe for an absence claim: Moonshot's $web_search
    injects results without any citation structure in the response
    (verified 2026-08-14), so result.citations is documented as always
    empty there. If citation-shaped fields ever appear, that documentation
    and the adapter are stale — normalize the citations and update the
    catalog note, USAGE.md, and this test. Probed raw so it verifies the
    provider, not KeyCall's own parsing."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")

    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "moonshot"), None)
    if target is None:
        pytest.skip("no moonshot target in the live source")

    tools = [{"type": "builtin_function", "function": {"name": "$web_search"}}]
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Search the web for one tech news headline from this week "
            "and name it with its outlet in one sentence.",
        }
    ]
    searched = False
    final = None
    with httpx.Client(timeout=180) as client:
        for _ in range(4):
            response = client.post(
                "https://api.moonshot.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {target.key}"},
                json={
                    "model": "kimi-k2.6",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 3000,
                },
            )
            assert response.status_code == 200, (
                f"capability drift: the $web_search round answered HTTP "
                f"{response.status_code} — re-probe the builtin flow"
            )
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") != "tool_calls":
                final = body
                break
            searched = True
            message = choice["message"]
            messages.append(message)
            for call in message.get("tool_calls", []):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": call["function"]["arguments"],
                    }
                )
    if not searched:
        pytest.skip("the model declined to search this run; nothing to conclude")
    assert final is not None, "the echo loop never reached a final answer"

    citation_shaped = {
        "citations", "references", "search_results", "annotations",
        "grounding", "groundingMetadata", "sources",
    }
    found = sorted(
        key
        for scope in (final, final["choices"][0], final["choices"][0].get("message", {}))
        for key in scope
        if key in citation_shaped
    )
    assert not found, (
        f"capability drift: a searched Moonshot response now carries {found} — "
        "normalize the citations and update the catalog note, USAGE.md, and "
        "this probe"
    )
    print("moonshot: searched answer still carries no citation structure (evidence current)")


def test_live_seed_and_temperature_support_still_holds():
    """Capability-drift probe for the seed gate and the temperature pins,
    probed raw so a failure is unambiguously the vendor's (evidence
    2026-09-08):

    - Gemini, DeepSeek, Moonshot, and xAI define a seed field: a valid seed
      is accepted, a malformed one rejected. If one stops taking a seed,
      supports_seed / SEED_PROVIDERS and USAGE are stale.
    - OpenAI (Responses API), Anthropic, and Perplexity have no seed field,
      so the gate refuses one before the network. Anthropic rejects an
      unknown top-level field outright; if a seed becomes a validated field
      on any of the three, the gate is wrongly blocking a usable parameter.
    - Anthropic's newest models (opus 4.7+, opus/sonnet 5, fable) and every
      Moonshot kimi model reject a non-default temperature. If one starts
      accepting 0.5, the catalog's sampling_constraints are stale.

    This test failing IS the notification to re-probe and update the
    catalog, _capabilities.SEED_PROVIDERS, USAGE.md, and this probe."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    targets, _ = load_targets(source)
    by_provider = {t.provider: t for t in targets}
    failures: list[str] = []

    chat = {
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "moonshot": "https://api.moonshot.ai/v1/chat/completions",
        "xai": "https://api.x.ai/v1/chat/completions",
    }
    listing = {
        "deepseek": "https://api.deepseek.com/v1/models",
        "moonshot": "https://api.moonshot.ai/v1/models",
        "xai": "https://api.x.ai/v1/models",
    }

    def first_chat_model(provider: str, key: str, client: httpx.Client) -> str | None:
        r = client.get(listing[provider], headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
        # Skip the image/video model families xAI lists alongside chat.
        chat_ids = [i for i in ids if "imagine" not in i and "image" not in i]
        return chat_ids[0] if chat_ids else (ids[0] if ids else None)

    with httpx.Client(timeout=60) as client:
        # --- seed accepted where the catalog says it exists (compat trio) ---
        for provider in ("deepseek", "moonshot", "xai"):
            target = by_provider.get(provider)
            if target is None:
                continue
            model = first_chat_model(provider, target.key, client)
            if model is None:
                failures.append(f"{provider}: no chat model listed to probe")
                continue
            headers = {"Authorization": f"Bearer {target.key}"}
            base = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
            ok = client.post(chat[provider], headers=headers, json={**base, "seed": 7})
            if ok.status_code != 200:
                failures.append(
                    f"{provider}: a valid seed returned HTTP {ok.status_code} on {model} "
                    f"(body {ok.text[:160]}) — seed may no longer be accepted; re-probe "
                    "supports_seed"
                )
            bad = client.post(chat[provider], headers=headers, json={**base, "seed": "not-an-int"})
            if bad.status_code == 200:
                failures.append(
                    f"{provider}: a malformed seed was accepted on {model} — the field may "
                    "no longer be validated; re-probe supports_seed"
                )

        # --- Gemini seed in generationConfig ---
        gem = by_provider.get("gemini")
        if gem is not None:
            page = client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"pageSize": 1000, "key": gem.key},
            )
            page.raise_for_status()
            gen_models = [
                m["name"].removeprefix("models/")
                for m in page.json()["models"]
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and "flash-lite" in m["name"]
            ]
            if gen_models:
                model = gen_models[0]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                        "generationConfig": {"maxOutputTokens": 2, "seed": 7}}
                ok = client.post(url, params={"key": gem.key}, json=body)
                if ok.status_code != 200:
                    failures.append(
                        f"gemini: generationConfig.seed returned HTTP {ok.status_code} on "
                        f"{model} — re-probe supports_seed"
                    )
                bad = client.post(url, params={"key": gem.key}, json={
                    "contents": body["contents"],
                    "generationConfig": {"maxOutputTokens": 2, "seed": "x"},
                })
                if bad.status_code == 200:
                    failures.append("gemini: a malformed seed was accepted — re-probe supports_seed")

        # --- seed absent on Anthropic: an unknown field is refused outright ---
        anth = by_provider.get("anthropic")
        if anth is not None:
            r = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anth.key, "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-5", "max_tokens": 2,
                      "messages": [{"role": "user", "content": "hi"}], "seed": 1},
            )
            if r.status_code == 200:
                failures.append(
                    "anthropic: a seed field was accepted — Anthropic may have added seed; "
                    "flip supports_seed and update the gate"
                )
            # --- temperature pinned on the newest Anthropic models ---
            for model in ("claude-opus-4-8", "claude-fable-5", "claude-sonnet-5"):
                r = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": anth.key, "anthropic-version": "2023-06-01"},
                    json={"model": model, "max_tokens": 2,
                          "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5},
                )
                if r.status_code == 200:
                    failures.append(
                        f"anthropic: {model} accepted temperature=0.5 — the sampling_constraints "
                        "pin is stale"
                    )

        # --- Moonshot kimi still pins temperature ---
        moon = by_provider.get("moonshot")
        if moon is not None:
            model = first_chat_model("moonshot", moon.key, client) or "kimi-k3"
            r = client.post(
                chat["moonshot"], headers={"Authorization": f"Bearer {moon.key}"},
                json={"model": model, "max_tokens": 2,
                      "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5},
            )
            if r.status_code == 200:
                failures.append(
                    f"moonshot: {model} accepted temperature=0.5 — the kimi sampling pin is stale"
                )

    assert not failures, "seed/temperature capability drift:\n" + "\n".join(failures)
    print("seed and temperature support current on every probed provider")


def test_live_grok_voice_dialect_evidence_still_holds():
    """Capability-drift probe for the three pieces of live evidence the
    xAI realtime support rests on (captured 2026-08-14). Probed raw, not
    through KeyCall's translator, so it verifies the provider rather
    than our own code:

    1. The session object speaks the pre-GA shape (`modalities`, not
       `output_modalities`) — this is why XAIAdapter passes
       ga_session=False and puts `voice` at the session top level.
    2. Grok Voice is voice-only: a text-modality update is echoed as
       accepted, yet the answer still arrives as audio plus transcript
       with no output_text deltas — this is why the docs say the words
       arrive as the transcript.
    3. response.done reports no usage — recorded in the catalog note,
       USAGE.md, and RealtimeTurnComplete's docstring.

    Any assertion failing means the dialect moved: re-probe, update
    XAIAdapter.realtime_plan, the catalog realtime_note, and the docs —
    this test failing IS the notification."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import json

    import httpx
    from httpx_ws import connect_ws

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "xai"), None)
    if target is None:
        pytest.skip("no xai target in the live source")

    client = httpx.Client(headers={"Authorization": f"Bearer {target.key}"})
    try:
        with connect_ws("wss://api.x.ai/v1/realtime?model=grok-voice-latest", client) as ws:
            created = json.loads(ws.receive_text(timeout=30))
            assert created["type"] == "session.created"
            session = created.get("session", {})
            assert "modalities" in session and "output_modalities" not in session, (
                "capability drift: the Grok Voice session object no longer speaks "
                f"the pre-GA shape (keys: {sorted(session.keys())}) — revisit "
                "ga_session=False in XAIAdapter.realtime_plan"
            )

            ws.send_text(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text"],
                            "instructions": "Answer in five words or fewer.",
                        },
                    }
                )
            )
            ws.send_text(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Why is the sky blue?"}],
                        },
                    }
                )
            )
            ws.send_text(json.dumps({"type": "response.create"}))

            kinds: set[str] = set()
            usage: dict = {}
            for _ in range(120):
                frame = json.loads(ws.receive_text(timeout=45))
                kinds.add(frame["type"])
                assert frame["type"] != "error", f"grok voice answered an error frame: {frame}"
                if frame["type"] == "response.done":
                    usage = frame.get("response", {}).get("usage") or {}
                    break
            else:
                pytest.fail("grok voice never sent response.done within the frame budget")

            assert "response.output_text.delta" not in kinds, (
                "capability drift: Grok Voice emitted text deltas — it is no longer "
                "voice-only, and a text output modality may now be exposable"
            )
            assert "response.output_audio.delta" in kinds, (
                f"capability drift: no audio deltas in a Grok Voice response ({sorted(kinds)})"
            )
            assert "response.output_audio_transcript.delta" in kinds, (
                "capability drift: no transcript deltas — the words no longer arrive "
                "as the audio transcript"
            )
            assert not usage.get("total_tokens"), (
                f"capability drift: Grok Voice now reports usage ({usage}) — update the "
                "catalog realtime_note, USAGE.md, and RealtimeTurnComplete's docstring"
            )
    finally:
        client.close()
    print("xai: grok voice still pre-GA, voice-only, and usage-free (evidence current)")


@pytest.mark.live
def test_live_xai_schema_enforcement_still_holds():
    """Capability-drift probe for xAI structured output (captured 2026-09-08
    on grok-4.6). Probed raw, not through KeyCall, so it verifies the provider:

    1. A strict json_schema response_format returns 200 with conforming JSON —
       this is why the catalog labels xai schema_enforcement=json_schema, which
       puts it in JSON_SCHEMA_COMPAT_PROVIDERS and makes the compat adapter emit
       the schema instead of the json_object floor.
    2. A malformed schema type 400s ("Schema validation failed") — the
       discriminator that xai parses and validates the schema rather than
       ignoring it.

    Either assertion failing means the enforcement moved: re-probe, revisit the
    catalog schema_enforcement flag and note, and the structured-output docs —
    this test failing IS the notification. A transport failure is not that
    signal and skips instead, since an unreachable host says nothing about how
    xai handles a schema."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import json

    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "xai"), None)
    if target is None:
        pytest.skip("no xai target in the live source")

    headers = {"Authorization": f"Bearer {target.key}", "Content-Type": "application/json"}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["answer", "count"],
        "additionalProperties": False,
    }
    prompt = "Reply in json: a one-word answer to 'capital of France' and a count of its letters."

    def probe() -> str:
        with httpx.Client(headers=headers, timeout=60) as client:
            listing = client.get("https://api.x.ai/v1/models")
            listing.raise_for_status()
            model = next(
                m["id"]
                for m in listing.json()["data"]
                if m["id"].startswith("grok-")
                and "imagine" not in m["id"]
                and "build" not in m["id"]
            )

            good = client.post(
                "https://api.x.ai/v1/chat/completions",
                json={
                    "model": model,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "keycall_response",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                },
            )
            assert good.status_code == 200, (
                f"capability drift: xai rejected a strict json_schema request "
                f"(HTTP {good.status_code}: {good.text[:300]}) — revisit the catalog "
                "schema_enforcement=json_schema flag"
            )
            parsed = json.loads(good.json()["choices"][0]["message"]["content"])
            assert set(parsed) == {"answer", "count"}, (
                f"capability drift: xai no longer conforms output to the schema (got {parsed})"
            )

            bad = client.post(
                "https://api.x.ai/v1/chat/completions",
                json={
                    "model": model,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "x",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "nonsense_type"}},
                                "required": ["answer"],
                            },
                            "strict": True,
                        },
                    },
                },
            )
            assert bad.status_code == 400, (
                f"capability drift: xai no longer validates the schema — a malformed "
                f"schema type answered HTTP {bad.status_code}, not 400 ({bad.text[:300]})"
            )
            return model

    # A TLS or socket failure says nothing about xai's schema handling, so it
    # must not read as capability drift: a runner hit SSL WRONG_VERSION_NUMBER
    # mid-read and failed a release on 2026-09-08. Retry once, then leave the
    # lane unverified for the run, the way a provider-paced batch does.
    for attempt in (1, 2):
        try:
            model = probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"xai unreachable from this runner ({type(exc).__name__}: {exc}); "
                    "schema enforcement unverified this run"
                )
    print(f"xai: strict json_schema still accepted and enforced on {model} (evidence current)")


@pytest.mark.live
def test_live_anthropic_structured_output_still_holds():
    """Capability-drift probe for Anthropic structured output (evidence
    2026-09-10), both halves of the fable-5-1 switch:

    1. output_config.format {type: json_schema} returns schema-conforming
       JSON on the newest listed model. Losing that means the native
       mechanism drifted: re-probe and revisit the adapter, the catalog
       schema_enforcement note, and the structured-output docs.
    2. An absence claim: claude-fable-5-1 refuses tool_choice type "any"
       with a 400, the evidence behind its catalog tool_choice_constraints
       entry. If it starts accepting, the constraint should come out and
       tool_choice='required' work there again.

    This test failing IS the notification. A transport failure says
    nothing about either claim and skips instead."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import json

    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "anthropic"), None)
    if target is None:
        pytest.skip("no anthropic target in the live source")

    headers = {
        "x-api-key": target.key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def probe() -> str:
        with httpx.Client(headers=headers, timeout=60) as client:
            listing = client.get("https://api.anthropic.com/v1/models?limit=100")
            listing.raise_for_status()
            ids = [m["id"] for m in listing.json()["data"]]
            model = ids[0]

            good = client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": model,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Say hi in one word."}],
                    "output_config": {
                        "format": {"type": "json_schema", "schema": schema}
                    },
                },
            )
            assert good.status_code == 200, (
                f"capability drift: anthropic rejected output_config.format on {model} "
                f"(HTTP {good.status_code}: {good.text[:300]}) — revisit the adapter and "
                "the catalog schema_enforcement note"
            )
            blocks = good.json()["content"]
            text = next(b["text"] for b in blocks if b["type"] == "text")
            assert set(json.loads(text)) == {"answer"}, (
                f"capability drift: anthropic no longer conforms output to the schema "
                f"(got {text[:200]})"
            )

            if "claude-fable-5-1" not in ids:
                print(
                    "anthropic: claude-fable-5-1 no longer listed; its tool_choice "
                    "constraint checks nothing this run"
                )
                return model
            forced = client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-fable-5-1",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "tools": [
                        {
                            "name": "t",
                            "description": "d",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    "tool_choice": {"type": "any"},
                },
            )
            assert forced.status_code == 400, (
                f"capability drift: claude-fable-5-1 answered a forced tool_choice with "
                f"HTTP {forced.status_code}, not 400 — drop its tool_choice_constraints "
                "entry from the catalog"
            )
            return model

    for attempt in (1, 2):
        try:
            model = probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"anthropic unreachable from this runner ({type(exc).__name__}: {exc}); "
                    "structured output unverified this run"
                )
    print(
        f"anthropic: output_config.format enforced on {model}, fable-5-1 still refuses "
        "forced tool_choice (evidence current)"
    )


def test_live_compat_reasoning_effort_still_unbound():
    """Capability-drift probe for the providers whose catalog
    `reasoning_effort` is false because the field is accepted and ignored
    (deepseek, moonshot; evidence remeasured on moonshot 2026-09-10).
    KeyCall refuses `reasoning_effort` there rather than letting a caller
    believe a level took effect, so the refusal rests on that evidence.

    Measuring binding every release would mean many samples per level per
    provider, since reasoning token counts swing hugely run to run. The
    cheap discriminator is validation: a provider that parses the field as
    a control validates its values, so an invalid level answering HTTP 200
    means the field is still being ignored. A 400 means the provider began
    parsing it, which is the moment to re-measure whether it now binds and
    to revisit the catalog flag.

    A transport failure says nothing either way and skips."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    from keycall._registry import resolve_provider

    targets, _ = load_targets(source)
    checked = []

    def probe(target) -> None:
        resolved = resolve_provider(target.provider)
        base = resolved.base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {target.key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(headers=headers, timeout=120) as client:
            listing = client.get(f"{base}/models")
            listing.raise_for_status()
            model = next(
                m["id"]
                for m in listing.json()["data"]
                if "vision" not in m["id"] and "embed" not in m["id"]
            )
            answer = client.post(
                f"{base}/chat/completions",
                json={
                    "model": model,
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "reasoning_effort": "zzz-not-a-level",
                },
            )
            assert answer.status_code == 200, (
                f"capability drift: {target.provider} now rejects an invalid "
                f"reasoning_effort (HTTP {answer.status_code}: {answer.text[:300]}), so it "
                "parses the field. Re-measure whether a level binds reasoning token counts "
                "and revisit the catalog reasoning_effort flag and its note"
            )
            checked.append(f"{target.provider}/{model}")

    for provider in ("moonshot",):
        target = next((t for t in targets if t.provider == provider), None)
        if target is None:
            continue
        if resolve_provider(provider).capabilities.reasoning_effort:
            pytest.fail(
                f"{provider} now records reasoning_effort=true; this probe covers the "
                "providers that ignore the field and needs updating alongside that flag"
            )
        for attempt in (1, 2):
            try:
                probe(target)
                break
            except httpx.TransportError as exc:
                if attempt == 2:
                    print(f"{provider} unreachable ({type(exc).__name__}: {exc}); unverified")
    if not checked:
        pytest.skip("no moonshot target in the live source")
    print(f"reasoning_effort still ignored (invalid level accepted) on: {', '.join(checked)}")


def test_live_openai_transcription_wires_still_split_the_same_way():
    """Capability-drift probe for the streaming-only transcription families
    (evidence 2026-09-10). OpenAI's listing mixes models only its realtime
    socket serves in with the ones the stored-file endpoint takes, and
    carries nothing to tell them apart, so KeyCall splits them on the id.

    Checked in both directions against every transcription model the key
    lists, because each failure hurts differently: a family recorded here
    that the endpoint does serve hides a working model, and one missing
    from it reaches a user as a bare 404 from the provider.

    A transport failure says nothing and skips."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import io
    import math
    import struct
    import wave

    import httpx

    from keycall import KeyCall, ModelCategory
    from keycall._registry import resolve_provider

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "openai"), None)
    if target is None:
        pytest.skip("no openai target in the live source")
    families = resolve_provider("openai").capabilities.streaming_only_transcription_families
    assert families, "openai records no streaming-only transcription families; probe and data move together"

    client = KeyCall(provider="openai", api_key=target.key, read_timeout=120)
    try:
        discovery = client.list_models(categories={ModelCategory.TRANSCRIPTION}, refresh=True)
    finally:
        client.close()
    model_ids = [m.id for m in discovery.models]
    assert model_ids, "openai listed no transcription models at all"

    # One second of a tone: valid audio, so a refusal is about the model.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(16000)
            )
        )
    audio = buffer.getvalue()

    def probe() -> list[str]:
        seen = []
        with httpx.Client(
            headers={"Authorization": f"Bearer {target.key}"}, timeout=120
        ) as http:
            for model_id in model_ids:
                answer = http.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    files={"file": ("probe.wav", audio, "audio/wav")},
                    data={"model": model_id},
                )
                recorded = any(family in model_id.lower() for family in families)
                if recorded:
                    assert answer.status_code == 404, (
                        f"capability drift: {model_id} matches a recorded realtime-only "
                        f"family but the stored-file endpoint answered HTTP "
                        f"{answer.status_code}; KeyCall is hiding a model that works. "
                        "Revisit openai's streaming_only_transcription_families"
                    )
                else:
                    assert answer.status_code == 200, (
                        f"capability drift: {model_id} is offered on the stored-file "
                        f"surface but the endpoint answered HTTP {answer.status_code} "
                        f"({answer.text[:200]}); it may need adding to openai's "
                        "streaming_only_transcription_families"
                    )
                seen.append(f"{model_id}{' (realtime-only)' if recorded else ''}")
        return seen

    for attempt in (1, 2):
        try:
            checked = probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"openai unreachable from this runner ({type(exc).__name__}: {exc}); "
                    "transcription wire split unverified this run"
                )
    print(f"openai transcription wire split holds across {len(checked)}: {', '.join(checked)}")


def test_live_deepseek_reasoning_effort_still_binds():
    """Capability-drift probe for DeepSeek's reasoning-effort control
    (evidence 2026-09-10, which replaced a 2026-08-14 note recording the
    field as accepted and ignored). Two claims, both cheap:

    1. The provider validates the level, refusing an unknown one with 400.
       A field parsed as an enum is a field being read.
    2. It binds at the boundary that needs no statistics: effort "none"
       spends zero reasoning tokens and a level spends some. Comparing
       levels against each other would need many samples, since the counts
       swing widely run to run; zero against nonzero does not.

    Failing either means the catalog's `reasoning_effort` flag for deepseek
    has drifted and its note needs re-measuring. A transport failure says
    nothing and skips."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    from keycall._registry import resolve_provider

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "deepseek"), None)
    if target is None:
        pytest.skip("no deepseek target in the live source")
    assert resolve_provider("deepseek").capabilities.reasoning_effort, (
        "deepseek no longer records reasoning_effort=true; this probe and that flag "
        "move together"
    )
    base = resolve_provider("deepseek").base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {target.key}", "Content-Type": "application/json"}
    ask = "A rope burns unevenly in 60 minutes. Measure 45 minutes with two ropes. Explain."

    def probe() -> str:
        with httpx.Client(headers=headers, timeout=300) as client:
            listing = client.get(f"{base}/models")
            listing.raise_for_status()
            model = listing.json()["data"][0]["id"]

            rejected = client.post(
                f"{base}/chat/completions",
                json={
                    "model": model,
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Say ok."}],
                    "reasoning_effort": "zzz-not-a-level",
                },
            )
            assert rejected.status_code == 400, (
                f"capability drift: deepseek accepted an invalid reasoning_effort "
                f"(HTTP {rejected.status_code}), so it may have stopped parsing the field. "
                "Re-measure whether a level still binds and revisit the catalog flag"
            )

            def spend(effort: str) -> int:
                answer = client.post(
                    f"{base}/chat/completions",
                    json={
                        "model": model,
                        "max_tokens": 2000,
                        "messages": [{"role": "user", "content": ask}],
                        "reasoning_effort": effort,
                    },
                )
                answer.raise_for_status()
                usage = answer.json()["usage"]
                return int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0))

            none_spend, high_spend = spend("none"), spend("high")
            assert none_spend == 0 and high_spend > 0, (
                f"capability drift: deepseek's effort levels no longer move reasoning spend "
                f"on {model} (none={none_spend}, high={high_spend}) — re-measure and revisit "
                "the catalog reasoning_effort flag and its note"
            )
            return f"{model} (none={none_spend}, high={high_spend})"

    for attempt in (1, 2):
        try:
            observed = probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"deepseek unreachable from this runner ({type(exc).__name__}: {exc}); "
                    "reasoning-effort binding unverified this run"
                )
    print(f"deepseek: reasoning_effort still validated and still binding on {observed}")


def test_live_google_maps_service_probes_still_hold():
    """Capability-drift probe for the Google Maps service adapter (evidence
    2026-09-10). Probed raw, not through KeyCall, so it verifies the provider:

    1. All three category endpoints take the key in the X-Goog-Api-Key
       header: geocoding on the v4beta surface, places text search, and
       routes. Any of them refusing the header means the auth carve moved
       and the catalog hosts and auth note need re-probing.
    2. An absence claim: the legacy maps.googleapis.com geocode endpoint
       still refuses a header-only key with REQUEST_DENIED. If it starts
       reading the header, the v4beta choice in the catalog geocoding note
       should be revisited.
    3. A bad key answers HTTP 400 with "API key not valid" (google.rpc's
       INVALID_ARGUMENT spelling), the discriminator behind the adapter's
       translate_error carve to INVALID_API_KEY.

    This test failing IS the notification. A transport failure says nothing
    about any claim and skips instead. Each enabled-endpoint probe is one
    billable call at the category's per-call rate."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "google_maps"), None)
    if target is None:
        pytest.skip("no google_maps target in the live source")

    def probe() -> None:
        with httpx.Client(timeout=60) as client:
            key_header = {"X-Goog-Api-Key": target.key}
            category_requests = {
                "geocoding": client.get(
                    "https://geocode.googleapis.com/v4beta/geocode/address",
                    params={"addressQuery": "1600 Amphitheatre Parkway, Mountain View, CA"},
                    headers=key_header,
                ),
                "places": client.post(
                    "https://places.googleapis.com/v1/places:searchText",
                    json={"textQuery": "coffee", "maxResultCount": 1},
                    headers={**key_header, "X-Goog-FieldMask": "places.id"},
                ),
                "directions": client.post(
                    "https://routes.googleapis.com/directions/v2:computeRoutes",
                    json={
                        "origin": {"address": "Victoria Station, London"},
                        "destination": {"address": "London Bridge Station, London"},
                        "travelMode": "TRANSIT",
                    },
                    headers={**key_header, "X-Goog-FieldMask": "routes.duration"},
                ),
            }
            for name, response in category_requests.items():
                # 200 is the enabled answer; 403 is a project with that API
                # switched off, which still proves the header carried the
                # key. Anything else means the auth surface moved.
                assert response.status_code in (200, 403), (
                    f"capability drift: {name} answered HTTP {response.status_code} "
                    f"to a header-keyed request ({response.text[:300]}) — re-probe "
                    "the catalog hosts and auth note"
                )

            legacy = client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": "1600 Amphitheatre Parkway, Mountain View, CA"},
                headers=key_header,
            )
            legacy_status = legacy.json().get("status") if legacy.status_code == 200 else None
            assert legacy_status == "REQUEST_DENIED", (
                f"capability drift: the legacy geocode endpoint no longer refuses a "
                f"header-only key (HTTP {legacy.status_code}, status {legacy_status!r}) "
                "— revisit the v4beta choice in the catalog geocoding note"
            )

            bad = client.get(
                "https://geocode.googleapis.com/v4beta/geocode/address",
                params={"addressQuery": "Mountain View"},
                headers={"X-Goog-Api-Key": "keycall-drift-probe-not-a-key"},
            )
            assert bad.status_code == 400 and "API key not valid" in bad.text, (
                f"capability drift: a bad key no longer answers 400 'API key not "
                f"valid' (HTTP {bad.status_code}: {bad.text[:300]}) — revisit the "
                "adapter's INVALID_API_KEY carve"
            )

    for attempt in (1, 2):
        try:
            probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"google maps unreachable from this runner ({type(exc).__name__}: "
                    f"{exc}); service probes unverified this run"
                )
    print(
        "google_maps: header key accepted on all three category endpoints, legacy "
        "geocode still refuses it, bad key still 400s (evidence current)"
    )


def test_live_livekit_probe_still_holds():
    """Capability-drift probe for the LiveKit service adapter (evidence
    2026-09-10). Minted and probed raw, not through KeyCall, so it verifies
    the provider's wire contract:

    1. A stdlib HS256 token (iss = api_key, api_secret signs, video.roomList
       grant, Bearer) answers RoomService ListRooms with 200.
    2. A wrong secret answers 401 with a body that is not Twirp JSON — the
       plain-text discriminator translate_error reads as a bad signature.
    3. A valid signature without the grant answers 401 with a Twirp JSON
       body naming permissions — the other half of that discriminator.

    This test failing IS the notification. A transport failure says nothing
    about any claim and skips instead."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import base64
    import hashlib
    import hmac
    import json
    import time

    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "livekit"), None)
    if target is None or target.secret is None or target.base_url is None:
        pytest.skip("no livekit target with a key pair and base_url in the live source")
    api_key, api_secret = target.key, target.secret

    # The dashboard offers the wss:// spelling; the API rides https on the
    # same host, the same normalization resolve_provider applies.
    origin = target.base_url.replace("wss://", "https://").rstrip("/")
    url = f"{origin}/twirp/livekit.RoomService/ListRooms"

    def mint(secret: str, grants: dict[str, bool]) -> str:
        def encode(raw: bytes) -> str:
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        now = int(time.time())
        header = encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = encode(
            json.dumps(
                {
                    "iss": api_key,
                    "sub": "keycall-drift-probe",
                    "nbf": now - 10,
                    "exp": now + 600,
                    "video": grants,
                }
            ).encode()
        )
        signing_input = f"{header}.{payload}".encode()
        signature = encode(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
        return f"{header}.{payload}.{signature}"

    def post(token: str) -> httpx.Response:
        return httpx.post(
            url,
            json={},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )

    def probe() -> None:
        good = post(mint(api_secret, {"roomList": True}))
        assert good.status_code == 200, (
            f"capability drift: a granted HS256 token no longer lists rooms "
            f"(HTTP {good.status_code}: {good.text[:300]}) — re-probe the mint "
            "convention against LiveKit Cloud"
        )

        bad_signature = post(mint(api_secret + "x", {"roomList": True}))
        assert bad_signature.status_code == 401, (
            f"capability drift: a bad signature answered HTTP "
            f"{bad_signature.status_code}, not 401 ({bad_signature.text[:300]})"
        )
        try:
            bad_payload = bad_signature.json()
        except ValueError:
            bad_payload = None
        assert not isinstance(bad_payload, dict), (
            f"capability drift: a bad signature now answers Twirp JSON "
            f"({bad_signature.text[:300]}) — the adapter's plain-text-vs-JSON 401 "
            "discriminator no longer separates wrong-secret from missing-grant"
        )

        ungranted = post(mint(api_secret, {}))
        assert ungranted.status_code == 401, (
            f"capability drift: a token without the roomList grant answered HTTP "
            f"{ungranted.status_code}, not 401 ({ungranted.text[:300]})"
        )
        try:
            ungranted_payload = ungranted.json()
        except ValueError:
            ungranted_payload = None
        assert isinstance(ungranted_payload, dict) and "permissions" in str(
            ungranted_payload.get("msg", "")
        ), (
            f"capability drift: a missing grant no longer answers Twirp JSON naming "
            f"permissions ({ungranted.text[:300]}) — revisit the adapter's 401 "
            "discriminator"
        )

    for attempt in (1, 2):
        try:
            probe()
            break
        except httpx.TransportError as exc:
            if attempt == 2:
                pytest.skip(
                    f"livekit unreachable from this runner ({type(exc).__name__}: "
                    f"{exc}); service probe unverified this run"
                )
    print(
        "livekit: granted token lists rooms, wrong secret answers plain-text 401, "
        "missing grant answers Twirp JSON 401 (evidence current)"
    )


@pytest.mark.live
def test_live_streaming_diarization_still_holds():
    """Capability-drift probe for streaming speaker labels (evidence
    2026-09-08), spoken by two macOS voices so more than one speaker exists:

    - AssemblyAI (speaker_labels=true) and Deepgram (diarize=true) label
      each finalized word. Losing that means streaming_diarization and the
      docs are stale.
    - ElevenLabs is an absence claim: its realtime words carry a speaker_id
      key that is always null, which is why diarize=True refuses there. If
      it starts filling the field, the flag should be turned on rather than
      left refusing a feature the provider now has.

    This test failing IS the notification to re-probe and update the
    catalog flags, _capabilities, and the transcription docs."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import base64
    import json
    import queue
    import shutil
    import subprocess
    import tempfile
    import threading
    import time as _time

    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("streaming diarization check needs `say` and `ffmpeg` to synthesize audio")
    import httpx
    from httpx_ws import WebSocketDisconnect, WebSocketNetworkError, connect_ws

    from keycall import KeyCall
    from keycall._capabilities import STREAMING_DIARIZATION_PROVIDERS

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for voice, line in (
            ("Alex", "The quick brown fox jumps over the lazy dog."),
            ("Samantha", "Then the cat curled up and watched it happen."),
        ):
            aiff = f"{tmp}/{voice}.aiff"
            subprocess.run(["say", "-v", voice, "-o", aiff, line], check=True)
            parts.append(aiff)
        subprocess.run(
            ["ffmpeg", "-y", "-i", parts[0], "-i", parts[1], "-filter_complex",
             "[0:a][1:a]concat=n=2:v=0:a=1", "-ar", "16000", "-ac", "1",
             "-f", "s16le", f"{tmp}/out.pcm"],
            check=True, capture_output=True,
        )
        with open(f"{tmp}/out.pcm", "rb") as handle:
            pcm = handle.read()

    targets, _ = load_targets(source)
    by_provider = {t.provider: t for t in targets}
    preferred_model = {"deepgram": "nova-3"}
    failures: list[str] = []

    for provider in sorted(STREAMING_DIARIZATION_PROVIDERS):
        target = by_provider.get(provider)
        if target is None:
            continue
        client = KeyCall(provider=provider, api_key=target.key)
        try:
            with client.transcribe_stream(
                model=preferred_model.get(provider), sample_rate=16000, diarize=True
            ) as session:

                def feed(feed_session=session):
                    for i in range(0, len(pcm), 3200):
                        feed_session.send_audio(pcm[i : i + 3200])
                        _time.sleep(0.05)
                    _time.sleep(1.5)
                    feed_session.finish()

                feeder = threading.Thread(target=feed)
                feeder.start()
                labels = set()
                for event in session.events(timeout=40):
                    if event.kind == "final_transcript":
                        labels |= {w.speaker for w in event.words if w.speaker is not None}
                feeder.join()
            if not labels:
                failures.append(
                    f"capability drift: {provider} returned no speaker label on a "
                    "diarized stream — revisit streaming_diarization and the docs"
                )
            else:
                print(f"{provider}: streaming diarization live, labels {sorted(labels)}")
        finally:
            client.close()

    eleven = by_provider.get("elevenlabs")
    if eleven is not None:
        seen: set[str] = set()
        http = httpx.Client(headers={"xi-api-key": eleven.key})
        try:
            url = (
                "wss://api.elevenlabs.io/v1/speech-to-text/realtime?audio_format=pcm_16000"
                "&commit_strategy=vad&include_timestamps=true&model_id=scribe_v2_realtime"
            )
            with connect_ws(url, http) as ws:
                for i in range(0, len(pcm), 3200):
                    ws.send_text(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(pcm[i : i + 3200]).decode("ascii"),
                        "sample_rate": 16000, "commit": False,
                    }))
                    _time.sleep(0.01)
                ws.send_text(json.dumps({
                    "message_type": "input_audio_chunk", "audio_base_64": "",
                    "sample_rate": 16000, "commit": True,
                }))
                deadline = _time.time() + 30
                while _time.time() < deadline:
                    try:
                        frame = json.loads(ws.receive_text(timeout=10))
                    except (queue.Empty, WebSocketDisconnect, WebSocketNetworkError):
                        break
                    if frame.get("type") == "committed_transcript_with_timestamps":
                        seen = {
                            str(w["speaker_id"])
                            for w in frame.get("words", [])
                            if w.get("speaker_id") is not None
                        }
                        break
        finally:
            http.close()
        if seen:
            failures.append(
                "capability drift: elevenlabs now fills speaker_id on the realtime "
                f"wire ({sorted(seen)}) — turn streaming_diarization on for it, read "
                "the id in ElevenLabsTranslator, and drop the refusal"
            )
        else:
            print("elevenlabs: realtime speaker_id still always null (evidence current)")

    assert not failures, "streaming diarization drift:\n" + "\n".join(failures)


# Solid blue 8x8 PNG, built inline so the suite carries no binary fixture.
def _blue_png() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    # Grok refuses images under 512 total pixels ("below the minimum of
    # 512 pixels", observed live 2026-09-02), so the probe stays a little
    # above that; the other providers accept any size.
    size = 32
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
    release re-checks that an image round-trips."""
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


def test_live_audio_and_file_input():
    """Audio is Gemini-only and documents work on three providers; both
    wire shapes are easy to get subtly wrong, so each release re-reads a
    valid WAV and a valid PDF."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import struct
    import wave
    from io import BytesIO

    from keycall import AudioInput, FileInput, KeyCall, Message, ModelCategory, TextInput
    from keycall._registry import providers_with

    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"".join(struct.pack("<h", (i % 200) * 60) for i in range(2400)))
    wav = buffer.getvalue()

    pdf = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 52 >>\nstream\n"
        b"BT /F1 24 Tf 72 700 Td (KEYCALL TEST DOCUMENT) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n%%EOF\n"
    )

    cases = (
        (
            "audio",
            providers_with("audio_input"),
            [TextInput(text="Describe this sound in one short sentence."), AudioInput(data=wav)],
            None,
        ),
        (
            "document",
            providers_with("file_input"),
            [
                TextInput(text="What does this document say? Quote it."),
                FileInput(data=pdf, filename="test.pdf"),
            ],
            "keycall",
        ),
    )

    targets, _ = load_targets(source)
    failures = []
    for label, supporting, parts, expected in cases:
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
                for model in candidates(discovery, 5):
                    try:
                        result = client.generate_text(
                            model=model.id,
                            messages=[Message(role="user", content=parts)],
                            max_output_tokens=250,
                        )
                        answer = (result.text or "").strip()
                        assert answer, f"{model.id} returned no text for the {label}"
                        if expected:
                            assert expected in answer.lower(), (
                                f"{model.id} did not read the {label}: {answer[:60]!r}"
                            )
                        print(f"{target.display_name}: {label} read by {model.id} -> {answer[:40]!r}")
                        break
                    except Exception as exc:  # noqa: BLE001 — reported, not hidden
                        attempt_errors.append(f"    {model.id}: {exc}")
                else:
                    failures.append(
                        f"{target.display_name}: no model read the {label}\n"
                        + "\n".join(attempt_errors)
                    )
            finally:
                client.close()
    assert not failures, "\n".join(failures)


def test_live_embeddings():
    """Both embedding endpoints batch and must return one vector per input
    in order; a provider quietly changing that would misalign every
    caller's index."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall
    from keycall._registry import providers_with

    # Embedding models aren't what the text walk selects, so name one per
    # provider; a retirement here shows up as a clean model_not_available.
    models = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-001"}
    inputs = ["the first string", "an entirely different second string"]

    targets, _ = load_targets(source)
    supporting = providers_with("embeddings")
    checked = []
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
            result = client.embed(model=models[target.provider], inputs=inputs)
            assert len(result.parts) == len(inputs)
            dims = {len(part.values) for part in result.parts}
            assert len(dims) == 1, f"ragged vector widths: {dims}"
            assert result.parts[0].values != result.parts[1].values, (
                "different inputs produced identical vectors, so the batch "
                "may not be mapping inputs to outputs"
            )
            print(
                f"{target.display_name}: {len(result.parts)} vectors of "
                f"{dims.pop()} dims from {models[target.provider]}"
            )
            checked.append(target.provider)
        finally:
            client.close()
    assert checked, "no embedding-capable target in the live source"


def test_live_image_generation():
    """The providers answer over different wires (OpenAI and xAI a
    dedicated images endpoint, Gemini an inlineData part on
    generateContent), and the bytes have to decode to a valid image, not
    just arrive."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import base64

    from keycall import KeyCall
    from keycall._registry import providers_with

    # Image models aren't what the text walk selects, so name one per
    # provider; a retirement shows up as a clean model_not_available.
    models = {
        "openai": "gpt-image-1",
        "gemini": "gemini-3.1-flash-image",
        "xai": "grok-imagine-image",
    }
    signatures = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"),
                  (b"RIFF", "webp"))

    targets, _ = load_targets(source)
    supporting = providers_with("image_generation")
    checked = []
    for target in targets:
        if target.provider not in supporting:
            continue
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
            read_timeout=180.0,
        )
        try:
            # A provider's own content filter firing is not a KeyCall
            # regression: Gemini's recitation check refuses roughly one in
            # ten runs of this very prompt (measured raw, 2026-09-08), and
            # failing a release on it blocks publishing for a reason no
            # code change can fix. Retry once, then leave this provider
            # unverified for the run rather than calling it a fault.
            for attempt in (1, 2):
                try:
                    result = client.generate_image(
                        model=models[target.provider],
                        prompt=(
                            "A simple flat illustration of a blue circle "
                            "on a white background"
                        ),
                    )
                    break
                except KeyCallError as exc:
                    if "recitation" not in exc.message.lower() or attempt == 2:
                        if "recitation" in exc.message.lower():
                            pytest.skip(
                                f"{target.display_name}: {models[target.provider]} hit the "
                                "provider's recitation filter twice; image generation "
                                "unverified this run"
                            )
                        raise
            assert result.parts, "no image part returned"
            part = result.parts[0]
            raw = base64.b64decode(part.base64_data or "")
            kind = next((name for magic, name in signatures if raw.startswith(magic)), None)
            assert kind is not None, (
                f"{models[target.provider]} returned bytes that are not a known image "
                f"format (first bytes {raw[:8]!r})"
            )
            assert kind in (part.media_type or ""), (
                f"media_type {part.media_type!r} disagrees with the actual {kind} bytes"
            )
            print(
                f"{target.display_name}: {len(raw)} byte {kind} from "
                f"{models[target.provider]}, {result.usage.total_tokens} tokens"
            )
            checked.append(target.provider)
        finally:
            client.close()
    assert checked, "no image-capable target in the live source"


# No provider's list_models response carries a price or tier field (nor
# does keycall.Model), so a cheaper/lighter model can only be recognized
# by name. These are the tier words providers already use today, tried
# in priority order; a provider with none of them falls back to its
# first listed model. Centralized here so a provider renaming its tier
# (e.g. "lite" -> "mini") is a one-line fix instead of a silent miss.
_CHEAP_TIER_HINTS = ("lite", "nano", "mini", "fast", "flash")


def _cheapest_model_id(models):
    for hint in _CHEAP_TIER_HINTS:
        match = next((m.id for m in models if hint in m.id.lower()), None)
        if match is not None:
            return match
    return models[0].id


def test_live_video_generation():
    """Video billing runs well above every other operation this suite
    covers, so this picks the lightest available model per provider (the
    live model list, not a fixed id: a hardcoded model would drift the
    day a lighter tier ships) and the shortest duration each accepts.
    Gemini's Veo only takes 4, 6, or 8 seconds; xAI's Grok Imagine takes
    any whole second from 1. Bytes still have to decode as a valid video,
    not just arrive."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import base64

    from keycall import KeyCall, ModelCategory
    from keycall._registry import providers_with

    shortest_duration = {"gemini": 4, "xai": 1}

    targets, _ = load_targets(source)
    supporting = providers_with("video_generation")
    checked = []
    for target in targets:
        if target.provider not in supporting:
            continue
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
            read_timeout=120.0,
        )
        try:
            discovery = client.list_models(categories={ModelCategory.VIDEO_GENERATION})
            assert discovery.models, f"{target.display_name}: no video models listed"
            model = _cheapest_model_id(discovery.models)
            result = client.generate_video(
                model=model,
                prompt="A simple flat illustration of a blue circle on a white background",
                duration_seconds=shortest_duration[target.provider],
                timeout=120.0,
            )
            assert result.parts, "no video part returned"
            part = result.parts[0]
            raw = base64.b64decode(part.base64_data or "")
            assert raw[4:12] == b"ftypisom" or raw[:4] == b"\x1aE\xdf\xa3", (
                f"{model} returned bytes that are not a recognized video "
                f"format (first bytes {raw[:16]!r})"
            )
            print(f"{target.display_name}: {len(raw)} bytes from {model}")
            checked.append(target.provider)
        finally:
            client.close()
    assert checked, "no video-capable target in the live source"


def test_live_prerecorded_transcription_every_supporting_target():
    """Four providers, two wire forms (three answer in one round trip,
    AssemblyAI runs a job); each release transcribes one synthesized
    clip on every one through transcribe() itself and checks the words
    came back with the timings each wire promises. All wire facts
    live-verified 2026-09-02."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("live transcription check needs `say` and `ffmpeg` to synthesize audio")
    from keycall import KeyCall
    from keycall._registry import providers_with

    with tempfile.TemporaryDirectory() as tmp:
        aiff = str(Path(tmp) / "probe.aiff")
        wav_path = str(Path(tmp) / "probe.wav")
        subprocess.run(
            ["say", "-o", aiff, "The quick brown fox jumps over the lazy dog."],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "16000",
             "-ac", "1", wav_path],
            check=True,
        )
        wav = Path(wav_path).read_bytes()

    plans = {
        "openai": {"model": "whisper-1", "words_promised": True},
        "elevenlabs": {"model": "scribe_v2", "words_promised": True},
        "deepgram": {"model": "nova-3", "words_promised": True},
        "assemblyai": {
            "model": "universal-2", "words_promised": True,
            "timeout": 300.0, "poll_interval": 3.0,
        },
    }
    targets, _ = load_targets(source)
    supporting = providers_with("transcription")
    checked = []
    for target in targets:
        if target.provider not in supporting or target.provider in checked:
            continue
        plan = dict(plans[target.provider])
        words_promised = plan.pop("words_promised")
        model = plan.pop("model")
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
            read_timeout=120.0,
        )
        try:
            result = client.transcribe(model=model, audio=wav, **plan)
            text = result.text.lower()
            for expected in ("quick", "brown", "fox", "lazy"):
                assert expected in text, (
                    f"{target.display_name}: {model} heard {result.text!r}"
                )
            if words_promised:
                assert result.words, f"{target.display_name}: no word timings"
                assert result.words[0].end_ms > 0
            print(
                f"{target.display_name}: {model} -> {len(result.words)} words, "
                f"{result.audio_duration_seconds}s audio"
            )
            checked.append(target.provider)
        finally:
            client.close()
    assert checked, "no transcription-capable target in the live source"


def test_live_openai_transcribe_family_still_refuses_verbose_json():
    """Drift probe for a pinned refusal: the gpt-4o transcribe family
    400s on response_format=verbose_json (observed live 2026-09-02),
    which is why KeyCall asks for word timings on whisper-1 only. If this
    starts succeeding, the adapter should start asking that family for
    words too — probed raw, not through the adapter, so a change is
    unambiguously the vendor's."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "openai"), None)
    if target is None:
        pytest.skip("no openai target in the live source")
    silent_wav = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    ) + b"\x00" * 32000
    with httpx.Client(
        timeout=60, headers={"Authorization": f"Bearer {target.key}"}
    ) as raw:
        response = raw.post(
            "https://api.openai.com/v1/audio/transcriptions",
            files={"file": ("probe.wav", silent_wav, "audio/wav")},
            data={"model": "gpt-4o-mini-transcribe", "response_format": "verbose_json"},
        )
    assert response.status_code == 400, (
        "gpt-4o-mini-transcribe now accepts verbose_json "
        f"(HTTP {response.status_code}): the whisper-1-only word-timing rule in "
        "the openai adapter, its catalog transcription_note, and USAGE's "
        "transcription section can all be widened"
    )
    print("gpt-4o-mini-transcribe still refuses verbose_json (400)")


BATCH_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-flash-latest",
    "moonshot": "kimi-k2.6",
    "xai": "grok-4.3",
}


@pytest.fixture(scope="session", autouse=True)
def _batches_submitted_early(request):
    """Submit the release's batch jobs before the rest of the live suite
    runs, so the providers' queue time overlaps the other tests instead of
    being waited out on its own.

    Batch completion is provider-paced and swings by hours on one provider
    between days: OpenAI answered in under four minutes on 2026-09-02 and
    took 163 minutes on 2026-09-08, both well inside the 24 hours every
    provider reserves. The suite's own runtime is the cheapest waiting
    budget available, so spending it here costs nothing and verifies a lane
    that otherwise skips.

    Autouse so submission happens at session start rather than wherever the
    batch test is ordered in a randomized run, but gated on that test
    being selected: a targeted run must not submit five billable jobs that
    nothing will read."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    wanted = any(
        "test_live_batch_generation_every_supporting_target" in item.nodeid
        for item in request.session.items
    )
    if not source or not wanted:
        yield None
        return

    import time

    from keycall import BatchRequest, KeyCall, Message, TextInput
    from keycall._registry import providers_with

    targets, _ = load_targets(source)
    supporting = providers_with("batch_generation")

    def preferred(provider):
        # The openai-keycall-batch-test key exists because the account's
        # Default project refuses its own batch input files (verified
        # 2026-09-02); prefer a batch-named target where one is present.
        candidates = [t for t in targets if t.provider == provider]
        named = [t for t in candidates if "batch" in (t.name or "")]
        return (named or candidates or [None])[0]

    entries: list[dict] = []
    opened = []
    try:
        for provider in sorted(supporting):
            target = preferred(provider)
            if target is None:
                continue
            client = KeyCall(
                provider=target.provider,
                api_key=target.key,
                protocol=target.protocol,
                base_url=target.base_url,
            )
            opened.append(client)
            requests = [
                BatchRequest(
                    model=BATCH_MODELS[target.provider],
                    messages=[Message(role="user", content=[TextInput(text=prompt)])],
                    max_output_tokens=200,
                )
                for prompt in ("Reply with the word one.", "Reply with the word two.")
            ]
            job = client.start_batch(requests)
            print(f"{target.display_name}: submitted {job.job_id} ({job.provider_status})")
            entries.append({"target": target, "client": client, "job": job})
        # The waiting budget is measured from here, not from whenever the
        # test is ordered, so the overlap with the rest of the suite can
        # only shorten the wait and never shorten the budget.
        yield {"entries": entries, "submitted_at": time.monotonic()}
    finally:
        for entry in entries:
            # Anything still running at session end would bill on its own
            # schedule; stop it before the suite exits.
            if entry["job"].status == "running":
                try:
                    entry["client"].cancel_batch(entry["job"])
                except KeyCallError:
                    pass
        for client in opened:
            client.close()


def test_live_batch_generation_every_supporting_target(_batches_submitted_early):
    """Five providers, five batch dialects. The jobs were submitted by the
    session fixture before the rest of the suite ran, so by the time this
    reads them they have had the whole suite's runtime to complete; this
    only polls out whatever is left. Results are read back in submission
    order regardless of the order the provider answered in.

    The named models double as drift probes: xAI validates batch
    eligibility at the add call, so grok-4.3 losing its batch lane fails
    this test by name, and each other id retiring fails as a create-time
    refusal."""
    submitted = _batches_submitted_early
    if submitted is None:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    entries = submitted["entries"]
    assert entries, "no batch-capable target in the live source"
    import time

    # The budget runs from submission, not from here, so whatever the suite
    # already spent is deducted rather than added: a batch test ordered late
    # waits out only the remainder, and one ordered early still gets the
    # whole budget instead of a shortened one.
    deadline = submitted["submitted_at"] + 2400.0
    in_flight = [entry for entry in entries if entry["job"].status == "running"]
    finished = [entry for entry in entries if entry["job"].status != "running"]
    if finished:
        print(f"{len(finished)} of {len(entries)} batches already done before polling")
    while in_flight and time.monotonic() < deadline:
        time.sleep(15.0)
        still_running = []
        for entry in in_flight:
            entry["job"] = entry["client"].check_batch(entry["job"])
            if entry["job"].status == "running":
                still_running.append(entry)
            else:
                finished.append(entry)
        in_flight = still_running

    for entry in finished:
        target, job = entry["target"], entry["job"]
        assert job.status == "finished", (
            f"{target.display_name}: batch ended {job.status} "
            f"({job.provider_status}): {job.error_message}"
        )
        results = entry["client"].fetch_batch_results(job)
        assert len(results) == 2, f"{target.display_name}: {len(results)} results for 2 requests"
        assert [r.key for r in results] == ["kc-0", "kc-1"]
        for result in results:
            assert result.succeeded, (
                f"{target.display_name}: request {result.index} errored "
                f"({result.error_code}): {result.error_message}"
            )
            assert result.result.text, f"{target.display_name}: empty text in a result"
        print(
            f"{target.display_name}: 2/2 succeeded on {BATCH_MODELS[target.provider]}, "
            f"texts {[r.result.text[:20] for r in results]}"
        )

    if in_flight:
        # Completion is provider-paced (promised within 24h), so a straggler
        # even after the whole suite plus this budget is a verification
        # -environment outcome, not a fault: the provider isn't implicated,
        # and a release must not be held hostage to its batch-queue latency.
        # The batches that did finish above are asserted in full, so a
        # dialect regression still fails; only an unfinished queue
        # skips, leaving that lane unverified this run and named in the skip
        # reason. Build the message by hand — a bare skip arg would let
        # pytest render the entries, Target keys included.
        pytest.skip(
            "batch still processing after the suite plus the poll budget "
            "(provider-paced, provider not implicated, release still "
            "unverified): "
            + ", ".join(
                f"{e['target'].display_name} ({e['job'].job_id}, "
                f"{e['job'].provider_status})"
                for e in in_flight
            )
        )


def test_live_gemini_batch_cancel_path_still_exists():
    """Gemini's :cancel endpoint is the one batch route the 2026-09-02
    probes never exercised (round 1's cancel probe died with its create);
    the adapter carries the documented path, so each release confirms the
    verb answers rather than 404ing. The cancelled batch is one tiny
    request, submitted and stopped in the same breath."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import BatchRequest, KeyCall, Message, TextInput

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider == "gemini"), None)
    if target is None:
        pytest.skip("no gemini target in the live source")
    client = KeyCall(
        provider=target.provider,
        api_key=target.key,
        protocol=target.protocol,
        base_url=target.base_url,
    )
    try:
        job = client.start_batch(
            [
                BatchRequest(
                    model="gemini-flash-latest",
                    messages=[Message(role="user", content=[TextInput(text="Say hi.")])],
                    max_output_tokens=16,
                )
            ]
        )
        cancelled = client.cancel_batch(job)
        assert cancelled.provider_status == "cancelling"
        print(f"{target.display_name}: cancel accepted for {job.job_id}")
    finally:
        client.close()


def test_live_async_client_parity():
    """The async client shares the adapters but has its own transport,
    context managers, and stream iteration. Everything shipped since 0.5.0
    was verified on the sync path only, so one target exercises the async
    one against a live provider each release."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import asyncio

    from keycall import AsyncKeyCall, Message, ModelCategory, TextInput, Tool, ToolResult
    from keycall._capabilities import TOOL_CALLING_PROVIDERS

    targets, _ = load_targets(source)
    target = next((t for t in targets if t.provider in TOOL_CALLING_PROVIDERS), None)
    if target is None:
        pytest.skip("no tool-calling target in the live source")

    weather = Tool(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    async def exercise() -> list[str]:
        notes: list[str] = []
        async with AsyncKeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        ) as client:
            discovery = await client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            models = candidates(discovery, 6)
            assert models, "no text models advertised"

            errors = []
            for model in models:
                ask = [Message(role="user", content=[TextInput(text="Reply with: ok")])]
                try:
                    plain = await client.generate_text(
                        model=model.id, messages=ask, max_output_tokens=200
                    )
                    notes.append(f"generate {model.id}: {(plain.text or '')[:20]!r}")

                    async with client.stream_text(
                        model=model.id, messages=ask, max_output_tokens=200
                    ) as stream:
                        deltas = 0
                        async for event in stream:
                            if event.kind == "text_delta":
                                deltas += 1
                        streamed = stream.result()
                    notes.append(f"stream {model.id}: {deltas} delta(s), {streamed.finish_reason}")

                    tool_ask = [
                        Message(
                            role="user",
                            content=[TextInput(text="Weather in London? Use the tool.")],
                        )
                    ]
                    first = await client.generate_text(
                        model=model.id, messages=tool_ask, tools=[weather], max_output_tokens=300
                    )
                    if first.tool_calls:
                        call = first.tool_calls[0]
                        final = await client.generate_text(
                            model=model.id,
                            messages=[
                                *tool_ask,
                                first.to_assistant_message(),
                                Message(
                                    role="user",
                                    content=[
                                        ToolResult(
                                            tool_call_id=call.id,
                                            name=call.name,
                                            content='{"temp_c": 14}',
                                        )
                                    ],
                                ),
                            ],
                            tools=[weather],
                            max_output_tokens=300,
                        )
                        notes.append(
                            f"tool round {model.id}: args {dict(call.arguments)}, "
                            f"final text {'yes' if final.text else 'NO'}"
                        )
                    return notes
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    errors.append(f"    {model.id}: {exc}")
            raise AssertionError(
                f"{target.display_name}: no model completed the async round\n"
                + "\n".join(errors)
            )

    for note in asyncio.run(exercise()):
        print(f"{target.display_name} async: {note}")


def test_live_candidate_order_has_headroom_before_the_budget():
    """The walk tries DEFAULT_ATTEMPTS models before giving up, and every
    release has assumed a working model appears well inside that budget.
    Nothing tested the assumption, so it could only fail on a user's key.

    It has already drifted twice: Gemini withdrew six of the first eight
    models it advertised (2026-08-09), and OpenAI killed all four of its
    `-chat-latest` aliases (2026-08-10). Both were found by accident.

    Asserting only that some model works would report the problem after
    users hit it. Requiring a margin reports it while there is still room
    to spare, which is what makes this a warning rather than a post-mortem.

    Cost is one generation per provider: a retired model refuses without
    charging, so only the success spends tokens.
    """
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    from keycall import KeyCall, Message, ModelCategory, TextInput
    from keycall._verify_core import DEFAULT_ATTEMPTS, order_candidates

    # Three spare attempts: enough that a provider can retire its top two
    # candidates between releases without a user ever seeing a failure.
    margin = 3
    ceiling = DEFAULT_ATTEMPTS - margin
    ask = [Message(role="user", content=[TextInput(text="Reply with the single word: ok")])]

    targets, _ = load_targets(source)
    service_names = set(supported_service_providers())
    failures = []
    for target in targets:
        if target.provider in service_names:
            continue  # no model walk exists on a service key
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
        try:
            try:
                discovery = client.list_models(
                    categories={ModelCategory.TEXT_GENERATION}, refresh=True
                )
            except KeyCallError as exc:
                # A provider with no list endpoint (Perplexity) has no walk
                # to measure; that is a different, already-reported fact.
                print(f"{target.display_name}: no model list to measure ({exc.code.value})")
                continue
            ordered = order_candidates(discovery.models)[:DEFAULT_ATTEMPTS]
            if not ordered:
                # Tinker serves your own fine-tuned checkpoints, so an empty
                # list is its normal state. No walk, no headroom to measure.
                print(f"{target.display_name}: no models advertised, no walk to measure")
                continue
            dead = []
            position = None
            for index, model in enumerate(ordered, start=1):
                try:
                    client.generate_text(
                        model=model.id, messages=ask, max_output_tokens=200
                    )
                    position = index
                    break
                except Exception as exc:  # noqa: BLE001 — reported, not hidden
                    dead.append(f"    {index}. {model.id}: {str(exc)[:90]}")
            if position is None:
                failures.append(
                    f"{target.display_name}: no model answered within "
                    f"{DEFAULT_ATTEMPTS} attempts — this key now fails verification\n"
                    + "\n".join(dead)
                )
            elif position > ceiling:
                failures.append(
                    f"{target.display_name}: first working model at position "
                    f"{position}, leaving {DEFAULT_ATTEMPTS - position} of "
                    f"{DEFAULT_ATTEMPTS} attempts spare (want {position} <= {ceiling}). "
                    "Candidate ordering needs revisiting for this provider "
                    "before the remaining margin runs out.\n" + "\n".join(dead)
                )
            else:
                print(
                    f"{target.display_name}: working model at position {position} "
                    f"of {DEFAULT_ATTEMPTS} ({ordered[position - 1].id})"
                )
        finally:
            client.close()
    assert not failures, "\n".join(failures)


def test_live_prompt_caching_anthropic_and_openai():
    """TextInput(cacheable=True) is verified two different ways here,
    because the two providers make two different promises.

    OpenAI's explicit breakpoint hit on the very next call in every trial
    (live-verified 2026-08-29) and is held to that bar.

    Anthropic's own docs describe its cache as best-effort with no hit-rate
    guarantee ("regularly analyze cache hit rates and adjust your
    strategy"), and nine live trials here confirmed inconsistent hits: five
    hits, four misses, across delays from 0 to 20 seconds with no
    correlation between delay length and outcome (live-verified
    2026-08-29). KeyCall's job is to send the marker correctly and report
    whatever the provider did correctly — never to guarantee a hit the
    provider itself won't guarantee. Every hit observed across all nine
    trials had cached_input_tokens matching the actual cached
    block's size, which is the claim this test holds Anthropic to: when it
    reports a hit, KeyCall must have read that report correctly. A run
    where every attempt misses is inconclusive, not a failure, and says so.
    """
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import time

    from keycall import KeyCall, Message, TextInput

    _CACHE_PROPAGATION_DELAY_SECONDS = 15
    _MAX_ATTEMPTS = 3
    model = {"anthropic": "claude-opus-5", "openai": "gpt-5.6"}

    def one_attempt(client, provider, attempt):
        # A fresh marker each attempt: reusing one across retries would let
        # an earlier attempt's own write be what a later attempt reads,
        # proving nothing about that attempt's own timing.
        marker = f"run-{time.monotonic_ns()}-{attempt}"
        filler = " ".join(
            f"Fact {i}: the KeyCall live-verification prefix for {marker} pads this block."
            for i in range(1200)
        )
        prefix = TextInput(text=filler, cacheable=True)
        client.generate_text(
            model=model[provider],
            messages=[
                Message(role="system", content=[prefix]),
                Message(role="user", content=[TextInput(text="Reply with: ok")]),
            ],
        )
        time.sleep(_CACHE_PROPAGATION_DELAY_SECONDS)
        second = client.generate_text(
            model=model[provider],
            messages=[
                Message(role="system", content=[prefix]),
                Message(role="user", content=[TextInput(text="Reply with: ok, again")]),
            ],
        )
        return second.usage.cached_input_tokens or 0

    targets, _ = load_targets(source)
    checked = []
    inconclusive = []
    for target in targets:
        if target.provider not in model:
            continue
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
        try:
            results = []
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                cached = one_attempt(client, target.provider, attempt)
                results.append(cached)
                print(f"{target.display_name}: attempt {attempt} cached={cached}")
                if cached > 0:
                    break
            hit = any(r > 0 for r in results)
            if target.provider == "anthropic" and not hit:
                print(
                    f"{target.display_name}: no cache read across {_MAX_ATTEMPTS} attempts — "
                    "inconclusive, not a failure, per Anthropic's own best-effort caching docs"
                )
                inconclusive.append(target.provider)
                continue
            assert hit, (
                f"{target.provider}: the identical cache-marked prefix was resent and no "
                f"cache read was reported across {_MAX_ATTEMPTS} attempts; the marker had "
                "no effect"
            )
            checked.append(target.provider)
        finally:
            client.close()
    # openai never reaches `inconclusive` above (only anthropic's branch
    # appends to it) — a miss there already raised via `assert hit`.
    assert checked or inconclusive, "no anthropic or openai target in the live source"


def test_live_alias_convention_evidence_still_holds():
    """Capability-drift probe for the catalog's alias_conventions evidence,
    which alias_fact() and Model.alias serve to consumers (rates bakes it
    into its ledger at build time):

    - Gemini, maintained=True: a -latest alias must answer a live
      generation, since the catalog says Gemini keeps those aimed at a
      live model (verified 2026-08-09).
    - OpenAI, maintained=False: the -chat-latest family was observed
      retired wholesale (2026-08-10). If a listed -chat-latest id answers
      a generation again, that claim has drifted — update the openai
      alias_conventions entry, the USAGE/README alias notes, and this
      probe. A family absent from the listing altogether is consistent
      with retirement and passes with a printed note.

    Probed through the ordinary client (the claim is about which ids
    answer, not about wire parsing)."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")

    from keycall import KeyCall, Message, ModelCategory, TextInput

    targets, _ = load_targets(source)
    probed = []

    gemini = next((t for t in targets if t.provider == "gemini"), None)
    if gemini is not None:
        client = KeyCall(provider="gemini", api_key=gemini.key, read_timeout=120)
        try:
            try:
                client.generate_text(
                    model="gemini-flash-latest",
                    messages=[Message(role="user", content=[TextInput(text="Say ok.")])],
                    max_output_tokens=200,
                )
                probed.append("gemini")
            except KeyCallError as error:
                # Only a model-is-gone class of refusal is drift
                # evidence; a transient (overload, rate limit, timeout) says
                # nothing about whether the alias is maintained.
                if error.retryable:
                    print(f"gemini probe inconclusive (transient: {error.code.value})")
                else:
                    raise AssertionError(
                        f"capability drift: gemini-flash-latest was refused "
                        f"({error.code.value}) — re-verify the gemini "
                        "alias_conventions entry (maintained=True, 2026-08-09)"
                    ) from error
        finally:
            client.close()

    openai = next((t for t in targets if t.provider == "openai"), None)
    if openai is not None:
        client = KeyCall(provider="openai", api_key=openai.key, read_timeout=120)
        try:
            listed = client.list_models(categories={ModelCategory.TEXT_GENERATION})
            chat_latest = [m.id for m in listed.models if m.id.endswith("-chat-latest")]
            if not chat_latest:
                print("openai lists no -chat-latest ids; consistent with the retirement claim")
            else:
                answered = []
                for model_id in sorted(chat_latest, reverse=True)[:2]:
                    try:
                        client.generate_text(
                            model=model_id,
                            messages=[
                                Message(role="user", content=[TextInput(text="Say ok.")])
                            ],
                            max_output_tokens=16,
                        )
                        answered.append(model_id)
                    except KeyCallError:
                        pass
                assert not answered, (
                    f"capability drift: {answered} answered a generation, but the catalog "
                    "records OpenAI's -chat-latest family as retired (maintained=False, "
                    "2026-08-10) — re-verify and update the openai alias_conventions "
                    "entry, the USAGE/README alias notes, and this probe"
                )
            probed.append("openai")
        finally:
            client.close()

    if not probed:
        pytest.skip("no gemini or openai target in the live source")


def test_live_compat_reasoning_token_reporting_still_holds():
    """Capability-drift probe for reasoning-token reporting on the four
    openai-compatible providers, which Usage.reasoning_tokens normalizes
    from completion_tokens_details.reasoning_tokens:

    - deepseek, moonshot, and xai report the field on their reasoning
      models (verified 2026-08-30: deepseek-v4-pro, kimi-k3, and
      grok-4.20-0309-reasoning all returned positive counts).
    - perplexity reports no reasoning-token count at all: its raw usage
      object carries cost fields and plain token counts, with thinking
      emitted as visible text (verified against the raw API 2026-08-30
      on sonar-reasoning-pro). If a count starts arriving, that claim
      has drifted - update the USAGE reasoning-tokens note and this
      probe.

    A transient error, or a preferred model missing from the live
    listing, is printed as inconclusive; only a live answer with the
    wrong reporting shape is drift evidence."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")

    from keycall import KeyCall, Message, TextInput

    preferred = {
        "deepseek": ("deepseek-v4-pro",),
        "moonshot": ("kimi-k3", "kimi-k2.6"),
        "xai": ("grok-4.20-0309-reasoning", "grok-4.6"),
        "perplexity": ("sonar-reasoning-pro", "sonar-reasoning"),
    }
    reporters = {"deepseek", "moonshot", "xai"}
    targets, _ = load_targets(source)
    probed = []
    for target in targets:
        wanted = preferred.get(target.provider)
        if wanted is None:
            continue
        client = KeyCall(provider=target.provider, api_key=target.key, read_timeout=300)
        try:
            listed = {m.id for m in client.list_models().models}
            model = next((m for m in wanted if m in listed), None)
            if model is None:
                print(
                    f"{target.provider} probe inconclusive: none of {wanted} "
                    "in the live listing - refresh the probe's model ids"
                )
                continue
            try:
                result = client.generate_text(
                    model=model,
                    messages=[Message(role="user", content=[TextInput(text="What is 17*23?")])],
                    max_output_tokens=2048,
                )
            except KeyCallError as error:
                if error.retryable:
                    print(
                        f"{target.provider} probe inconclusive "
                        f"(transient: {error.code.value})"
                    )
                    continue
                raise
            reported = result.usage.reasoning_tokens
            if target.provider in reporters:
                assert isinstance(reported, int) and reported > 0, (
                    f"capability drift: {target.provider} ({model}) answered without a "
                    "reasoning-token count, but the adapter's evidence says it reports "
                    "completion_tokens_details.reasoning_tokens (2026-08-30) - re-verify "
                    "and update the USAGE reasoning-tokens note and this probe"
                )
            else:
                assert reported is None, (
                    f"capability drift: {target.provider} ({model}) reported "
                    f"reasoning_tokens={reported}, but the recorded evidence says it "
                    "sends no reasoning-token count (raw-verified 2026-08-30) - update "
                    "the USAGE reasoning-tokens note and this probe"
                )
            probed.append(target.provider)
        finally:
            client.close()

    if not probed:
        pytest.skip("no compat-provider target in the live source")


def test_live_speech_generation_every_supporting_target():
    """One tiny billable clip per TTS-capable target: voices list, then a
    generation with the first voice. Model choice comes from the live
    model list, not a hardcoded id."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")

    import base64 as b64

    from keycall import KeyCall, ModelCategory
    from keycall._capabilities import providers_with

    supporting = providers_with("speech_generation")
    preferred = {
        "openai": "gpt-4o-mini-tts",
        "elevenlabs": "eleven_flash_v2_5",
    }
    targets, _ = load_targets(source)
    probed = []
    failures = []
    for target in targets:
        if target.provider not in supporting:
            continue
        client = KeyCall(provider=target.provider, api_key=target.key, read_timeout=120)
        try:
            voices = client.list_voices()
            assert voices, f"{target.provider}: list_voices returned nothing"
            listed = client.list_models(
                categories={ModelCategory.SPEECH_GENERATION}, refresh=True
            ).models
            if not listed:
                print(f"{target.provider}: no speech models listed — skipping generation")
                continue
            model = preferred.get(target.provider)
            if model not in {m.id for m in listed}:
                model = listed[0].id
            try:
                result = client.generate_speech(
                    model=model, text="One red circle.", voice=voices[0].id
                )
            except KeyCallError as error:
                if error.retryable:
                    print(f"{target.provider} inconclusive (transient: {error.code.value})")
                    continue
                failures.append(f"{target.provider} ({model}): {error.code.value} — {error}")
                continue
            clip = result.parts[0]
            audio = b64.b64decode(clip.base64_data)
            assert audio, f"{target.provider}: empty audio"
            print(f"{target.provider}: {model} spoke {len(audio)} bytes ({clip.media_type})")
            probed.append(target.provider)
        finally:
            client.close()

    assert not failures, "; ".join(failures)
    if not probed:
        pytest.skip("no speech-capable target in the live source")


def test_live_retired_models_still_refused_by_their_providers():
    """Drift probe behind the retired-model registry: every catalog
    retirement entry (id and each alias spelling) must still be refused by
    its provider, probed raw so a change is unambiguously the vendor's. A
    model answering again means the entry is stale — remove it from the
    provider's retired_models in catalog.json (the pre-flight gate and the
    listing filter both read it), and re-check USAGE's retired-models
    section. Gemini is probed by listing absence (Google removes shut-down
    models from GET /models, verified 2026-09-04); the chat providers are
    probed with a one-token request, which a retired model refuses without
    billing; ElevenLabs with a speech request its refusal never bills."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import concurrent.futures

    import httpx

    from keycall._registry import resolve_provider, supported_providers

    targets, _ = load_targets(source)
    by_provider = {t.provider: t for t in targets}

    failures: list[str] = []
    probes: list[tuple[str, str]] = []
    for provider in supported_providers():
        resolved = resolve_provider(provider)
        if not resolved.retired_models or provider not in by_provider:
            continue
        for entry in resolved.retired_models:
            for spelling in (entry["id"], *entry.get("aliases", ())):
                probes.append((provider, spelling))

    chat_urls = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "perplexity": "https://api.perplexity.ai/chat/completions",
        "moonshot": "https://api.moonshot.ai/v1/chat/completions",
        "xai": "https://api.x.ai/v1/chat/completions",
    }

    gemini_listed: set[str] = set()
    if "gemini" in by_provider and any(p == "gemini" for p, _ in probes):
        with httpx.Client(timeout=60) as raw:
            page = raw.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"pageSize": 1000, "key": by_provider["gemini"].key},
            )
            page.raise_for_status()
            gemini_listed = {
                m["name"].removeprefix("models/") for m in page.json()["models"]
            }

    # One client per provider, shared across its probes: a client per
    # probe floods macOS DNS resolution and fails with spurious
    # ConnectErrors at this volume.
    clients: dict[str, httpx.Client] = {}
    for provider, _model in probes:
        if provider not in clients and provider != "gemini":
            key = by_provider[provider].key
            headers = (
                {"x-api-key": key, "anthropic-version": "2023-06-01"}
                if provider == "anthropic"
                else {"xi-api-key": key}
                if provider == "elevenlabs"
                else {"Authorization": f"Bearer {key}"}
            )
            clients[provider] = httpx.Client(timeout=60, headers=headers)

    def probe(provider: str, model: str) -> str | None:
        fix = (
            f"{provider}/{model} answered again: remove its retired_models "
            "entry from catalog.json and re-check USAGE's retired-models table"
        )
        if provider == "gemini":
            return fix + " (it reappeared in the model listing)" if model in gemini_listed else None
        raw = clients[provider]
        for attempt in range(2):
            try:
                if provider == "anthropic":
                    response = raw.post(
                        "https://api.anthropic.com/v1/messages",
                        json={"model": model, "max_tokens": 1,
                              "messages": [{"role": "user", "content": "x"}]},
                    )
                    return None if response.status_code == 404 else f"{fix} (HTTP {response.status_code})"
                if provider == "elevenlabs":
                    response = raw.post(
                        "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
                        json={"text": "x", "model_id": model},
                    )
                    refused = response.status_code == 400 and "unsupported_model" in response.text
                    return None if refused else f"{fix} (HTTP {response.status_code})"
                response = raw.post(
                    chat_urls[provider],
                    json={"model": model, "max_tokens": 16,
                          "messages": [{"role": "user", "content": "x"}]},
                )
                break
            except httpx.TransportError:
                if attempt:
                    raise
        # A refusal is a 4xx that names the model as gone; a 200 means
        # the model (or a redirect for it) answered and the entry is
        # stale. Other 4xx forms (bad key, rate limit) are
        # inconclusive rather than a pass.
        if response.status_code == 200:
            return fix + " (HTTP 200)"
        text = response.text.lower()
        gone = any(
            marker in text
            for marker in ("not found", "not_found", "deprecated", "invalid model",
                           "does not exist", "no longer available")
        )
        return None if gone else f"{fix} (HTTP {response.status_code}: {response.text[:120]})"

    ran = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(probe, provider, model): (provider, model)
                for provider, model in probes
            }
            for future in concurrent.futures.as_completed(futures):
                ran += 1
                outcome = future.result()
                if outcome:
                    failures.append(outcome)
    finally:
        for client in clients.values():
            client.close()

    if not ran:
        pytest.skip("no live key covers a provider with retirement entries")
    assert not failures, "\n".join(sorted(failures))
    print(f"all {ran} retired-model spellings still refused across their providers")


def test_live_retired_model_replacements_still_exist():
    """The other half of the registry's promise: every recommended
    replacement the refusal names must itself be alive, or the error sends
    the caller on a second failing round trip. Checked against each
    provider's own live listing; entries whose note says the replacement is
    not verifiable in a listing are skipped by that note."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import httpx

    from keycall._registry import resolve_provider, supported_providers

    targets, _ = load_targets(source)
    by_provider = {t.provider: t for t in targets}

    listings: dict[str, set[str]] = {}

    def listed(provider: str) -> set[str] | None:
        if provider in listings:
            return listings[provider]
        target = by_provider.get(provider)
        if target is None:
            return None
        with httpx.Client(timeout=60) as raw:
            if provider == "gemini":
                page = raw.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"pageSize": 1000, "key": target.key},
                )
                page.raise_for_status()
                ids = {m["name"].removeprefix("models/") for m in page.json()["models"]}
            elif provider in ("openai", "moonshot", "xai"):
                base = {
                    "openai": "https://api.openai.com/v1",
                    "moonshot": "https://api.moonshot.ai/v1",
                    "xai": "https://api.x.ai/v1",
                }[provider]
                page = raw.get(f"{base}/models", headers={"Authorization": f"Bearer {target.key}"})
                page.raise_for_status()
                ids = {m["id"] for m in page.json()["data"]}
            elif provider == "anthropic":
                page = raw.get(
                    "https://api.anthropic.com/v1/models",
                    params={"limit": 1000},
                    headers={"x-api-key": target.key, "anthropic-version": "2023-06-01"},
                )
                page.raise_for_status()
                ids = {m["id"] for m in page.json()["data"]}
            else:
                # Perplexity and ElevenLabs have no listing that carries
                # these ids; their replacements are covered by other live
                # tests using them directly.
                listings[provider] = set()
                return listings[provider]
        listings[provider] = ids
        return ids

    failures: list[str] = []
    checked = 0
    for provider in supported_providers():
        resolved = resolve_provider(provider)
        for entry in resolved.retired_models:
            replacement = entry.get("replacement")
            if not replacement or "not verifiable" in entry.get("note", ""):
                continue
            ids = listed(provider)
            if not ids:
                continue
            checked += 1
            if replacement not in ids:
                failures.append(
                    f"{provider}: replacement {replacement} (for retired "
                    f"{entry['id']}) is gone from the live listing — update "
                    "the catalog entry to the provider's current recommendation"
                )
    if not checked:
        pytest.skip("no live key covers a provider with verifiable replacements")
    assert not failures, "\n".join(failures)
    print(f"all {checked} recorded replacements still listed by their providers")
