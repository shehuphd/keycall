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
    """Both providers answer in different shapes (OpenAI a dedicated images
    endpoint, Gemini an inlineData part on generateContent), and the bytes
    have to decode to a valid image, not just arrive."""
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    import base64

    from keycall import KeyCall
    from keycall._registry import providers_with

    # Image models aren't what the text walk selects, so name one per
    # provider; a retirement shows up as a clean model_not_available.
    models = {"openai": "gpt-image-1", "gemini": "gemini-3.1-flash-image"}
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
            result = client.generate_image(
                model=models[target.provider],
                prompt="A simple flat illustration of a blue circle on a white background",
            )
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
    failures = []
    for target in targets:
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
