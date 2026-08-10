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

    # Embedding models are not what the text walk selects, so name one per
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

    # Image models are not what the text walk selects, so name one per
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
