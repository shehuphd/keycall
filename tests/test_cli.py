import json

import httpx

from keycall._cli import main

CANARY = "sk-canary-cli-key-000"


def openai_ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "gpt-4o-mini"}]})
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    )


def patch_transport(monkeypatch, handler):
    from keycall import _client

    original_init = _client.KeyCall.__init__

    def patched(self, **kwargs):
        kwargs.setdefault("httpx_transport", httpx.MockTransport(handler))
        original_init(self, **kwargs)

    monkeypatch.setattr(_client.KeyCall, "__init__", patched)


def test_verify_success_exit_zero(tmp_path, monkeypatch, capsys):
    patch_transport(monkeypatch, openai_ok_handler)
    source = tmp_path / "keys.txt"
    source.write_text(f"provider=openai key={CANARY} name=test-key\n", encoding="utf-8")
    source.chmod(0o600)

    exit_code = main(["verify", "--source", str(source)])
    output = capsys.readouterr()
    assert exit_code == 0
    assert "✓ test-key" in output.out
    assert CANARY not in output.out
    assert CANARY not in output.err


def test_verify_generate_reports_selected_model(tmp_path, monkeypatch, capsys):
    patch_transport(monkeypatch, openai_ok_handler)
    source = tmp_path / "keys.txt"
    source.write_text(f"provider=openai key={CANARY}\n", encoding="utf-8")
    source.chmod(0o600)

    exit_code = main(["verify", "--source", str(source), "--generate"])
    output = capsys.readouterr()
    assert exit_code == 0
    assert "gpt-4o-mini" in output.out
    assert "filtered position 0" in output.out


def test_verify_bad_key_exit_one_without_leak(tmp_path, monkeypatch, capsys):
    def bad_key_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"bad key: {CANARY}"}}
        )

    patch_transport(monkeypatch, bad_key_handler)
    source = tmp_path / "keys.txt"
    source.write_text(f"provider=openai key={CANARY}\n", encoding="utf-8")
    source.chmod(0o600)

    exit_code = main(["verify", "--source", str(source)])
    output = capsys.readouterr()
    assert exit_code == 1
    assert "invalid_api_key" in output.out
    assert CANARY not in output.out
    assert CANARY not in output.err


def test_verify_missing_source_file_exit_two(capsys):
    exit_code = main(["verify", "--source", "/nonexistent/keys.txt"])
    assert exit_code == 2


def test_verify_strict_credentials_promotes_warning(tmp_path, monkeypatch, capsys):
    patch_transport(monkeypatch, openai_ok_handler)
    source = tmp_path / "keys.txt"
    source.write_text(f"provider=openai key={CANARY}\n", encoding="utf-8")
    source.chmod(0o644)  # broadly readable → warning → strict error

    exit_code = main(["verify", "--source", str(source), "--strict-credentials"])
    assert exit_code == 2


def test_run_verify_records_raw_and_filtered_positions():
    from keycall import KeyCall
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            # An embedding model precedes the text model, so raw and
            # filtered positions differ.
            return httpx.Response(
                200,
                json={"data": [{"id": "text-embedding-3-small"}, {"id": "gpt-4o-mini"}]},
            )
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            },
        )

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = run_verify(
        Target(provider="openai", key=CANARY), generate=True, client=client
    )
    client.close()
    assert result.generate_ok
    attempt = result.attempts[0]
    assert attempt.position == 0
    assert attempt.raw_position == 1


def test_run_verify_records_digest_rule_version_and_evidence():
    from keycall import KeyCall
    from keycall._sources import Target
    from keycall._verify_core import SELECTION_RULE_VERSION, run_verify

    def make_handler(model_ids):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": m} for m in model_ids]})
            return httpx.Response(
                200,
                json={
                    "model": model_ids[-1],
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                },
            )

        return handler

    def verify_with(model_ids):
        client = KeyCall(
            provider="openai",
            api_key=CANARY,
            httpx_transport=httpx.MockTransport(make_handler(model_ids)),
        )
        result = run_verify(Target(provider="openai", key=CANARY), generate=True, client=client)
        client.close()
        return result

    first = verify_with(["text-embedding-3-small", "gpt-4o-mini"])
    assert first.selection_rule_version == SELECTION_RULE_VERSION
    assert first.model_list_digest is not None
    assert len(first.model_list_digest) == 16
    assert first.attempts[0].classification_source == "keycall_rule"

    # Same advertised surface, same digest; a changed surface changes it.
    assert verify_with(["text-embedding-3-small", "gpt-4o-mini"]).model_list_digest == (
        first.model_list_digest
    )
    assert verify_with(["gpt-4o-mini"]).model_list_digest != first.model_list_digest


def test_run_verify_list_failure_has_no_digest():
    from keycall import KeyCall
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = run_verify(Target(provider="openai", key=CANARY), generate=True, client=client)
    client.close()
    assert not result.listed_ok
    assert result.model_list_digest is None


def test_verify_walk_tries_maintained_aliases_first():
    """Gemini advertised six withdrawn models ahead of every working one
    on 2026-08-09; walking in list order spends the budget on models the
    provider has already shut down."""
    from keycall import KeyCall
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    retired = "This model is no longer available to new users."
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
                        for m in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")
                    ]
                },
            )
        model = request.url.path.split("/models/")[1].split(":")[0]
        calls.append(model)
        if model != "gemini-flash-latest":
            return httpx.Response(404, json={"error": {"status": "NOT_FOUND", "message": retired}})
        return httpx.Response(
            200,
            json={
                "modelVersion": model,
                "candidates": [
                    {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1},
            },
        )

    client = KeyCall(
        provider="gemini", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = run_verify(Target(provider="gemini", key=CANARY), generate=True, client=client)
    client.close()

    assert calls[0] == "gemini-flash-latest", "the maintained alias must be tried first"
    assert result.generate_ok
    assert len(result.attempts) == 1
    # Promotion must not lose where the model sat in the raw list.
    assert result.attempts[0].raw_position == 2


def test_verify_walk_prefers_the_newest_model_a_provider_dates():
    """OpenAI dates every model it lists, and on 2026-08-10 all four of its
    `-chat-latest` aliases were dead (two unknown, two newly deprecated)
    while the numbered models worked. Alias-first burned half the budget
    before reaching anything that could answer, so a provider that reports
    its own dates is walked newest-first instead."""
    from keycall import KeyCall
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    calls: list[str] = []
    # Deliberately listed oldest-first with the dead aliases last, so
    # neither list order nor alias-first would reach the working model.
    listing = [
        {"id": "gpt-3.5-turbo", "created": 1_600_000_000},
        {"id": "gpt-4", "created": 1_650_000_000},
        {"id": "gpt-5.9", "created": 1_780_000_000},
        {"id": "gpt-5-chat-latest", "created": 1_700_000_000},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": listing})
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(
            200,
            json={
                "model": model,
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
                ],
                "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
            },
        )

    client = KeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = run_verify(Target(provider="openai", key=CANARY), generate=True, client=client)
    client.close()

    assert calls[0] == "gpt-5.9", "the newest dated model must be tried first"
    assert result.generate_ok
    # The alias is newer than two of the models, so recency must not be
    # quietly re-sorting on the name.
    assert calls == ["gpt-5.9"]
    assert result.attempts[0].raw_position == 2


def test_candidate_order_falls_back_when_only_some_models_are_dated():
    """A half-dated list says nothing about where the undated models
    belong, so mixing the two rules would order on invented evidence."""
    from datetime import datetime, timezone

    from keycall._types import Model
    from keycall._verify_core import order_candidates

    def model(name, when=None):
        return Model(
            id=name,
            provider="x",
            categories=frozenset(),
            released_at=when,
        )

    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    partly = [model("old-one"), model("new-one", when), model("thing-latest")]
    assert [m.id for m in order_candidates(partly)] == [
        "thing-latest",
        "old-one",
        "new-one",
    ], "a partly dated list must fall back to alias-first"

    fully = [model("older", when), model("newer", datetime(2026, 6, 1, tzinfo=timezone.utc))]
    assert [m.id for m in order_candidates(fully)] == ["newer", "older"]

    # Undated everywhere and no alias: the provider's own order stands.
    plain = [model("a"), model("b"), model("c")]
    assert [m.id for m in order_candidates(plain)] == ["a", "b", "c"]


def test_unresolvable_target_is_reported_not_raised():
    """One bad entry in a key file must not abort verification of the rest:
    run_verify promises it never raises for a target it can't use."""
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    result = run_verify(
        Target(provider="not-a-provider", key=CANARY, name="typo"), generate=True
    )
    assert result.outcome == "unresolvable_target"
    assert not result.listed_ok
    assert result.list_error_code == "unsupported_provider"
    assert "base_url" in (result.list_error_message or ""), "say how to fix it"


# --- CLI presentation: welcome, humanized errors, color discipline ----------


def test_bare_keycall_is_a_welcome_not_a_help_dump(capsys):
    from keycall import __version__

    exit_code = main([])
    output = capsys.readouterr()
    assert exit_code == 0
    assert __version__ in output.out
    assert "keycall verify" in output.out
    assert "keycall view" in output.out
    assert "usage:" not in output.out
    assert output.err == ""


def test_version_flag(capsys):
    import pytest as _pytest

    from keycall import __version__

    with _pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_command_gets_one_confident_suggestion(capsys):
    exit_code = main(["verfy"])
    output = capsys.readouterr()
    assert exit_code == 2
    assert output.out == ""
    assert "isn't a keycall command" in output.err
    assert "keycall verify" in output.err
    assert "usage:" not in output.err


def test_hopeless_command_gets_no_guess(capsys):
    exit_code = main(["zzqqxx"])
    output = capsys.readouterr()
    assert exit_code == 2
    assert "Perhaps" not in output.err
    assert "isn't a keycall command" in output.err


def test_pasted_key_as_command_is_hidden_with_guidance(capsys):
    pasted = "AQ.Ab8Pk29xLmNvbTradWQ4d2"
    exit_code = main([pasted])
    output = capsys.readouterr()
    assert exit_code == 2
    assert pasted not in output.err
    assert pasted not in output.out
    assert "hidden" in output.err
    assert "shell history" in output.err
    assert "env:MY_KEY" in output.err


def test_pasted_key_as_extra_argument_is_hidden(capsys):
    pasted = "sk-proj-Ab8Pk29xLmNvbTradWQ4d2"
    exit_code = main(["verify", pasted])
    output = capsys.readouterr()
    assert exit_code == 2
    assert pasted not in output.err
    assert pasted not in output.out
    assert "hidden" in output.err


def test_mistyped_flag_suggests_the_intended_one(capsys):
    exit_code = main(["verify", "--sourc", "keys.toml"])
    output = capsys.readouterr()
    assert exit_code == 2
    assert "--source" in output.err
    assert "usage:" not in output.err


def test_flag_abbreviation_is_rejected_not_expanded(tmp_path, capsys):
    # With abbreviation on, --sour would silently mean --source; the day a
    # new flag shares the prefix, every script using it breaks. Reject now.
    exit_code = main(["verify", "--sour", str(tmp_path / "keys.txt")])
    assert exit_code == 2


def test_bad_int_value_reads_as_a_sentence(capsys):
    exit_code = main(["verify", "--attempts", "abc"])
    output = capsys.readouterr()
    assert exit_code == 2
    assert "whole number" in output.err
    assert "invalid int value" not in output.err


def test_flag_missing_its_value_reads_as_a_sentence(capsys):
    exit_code = main(["verify", "--source"])
    output = capsys.readouterr()
    assert exit_code == 2
    assert "needs a value" in output.err
    assert "expected one argument" not in output.err


def test_no_escape_codes_off_tty(capsys):
    main(["verfy"])
    output = capsys.readouterr()
    assert "\x1b[" not in output.err
    assert "\x1b[" not in output.out


def test_color_respects_no_color_even_on_a_tty(monkeypatch):
    from keycall._cli import _paint

    class FakeTTY:
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert _paint("x", "31", FakeTTY()) == "x"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.delenv("TERM", raising=False)
    assert _paint("x", "31", FakeTTY()) == "\x1b[31mx\x1b[0m"


def test_interactive_prompt_names_every_provider_and_ignores_case(monkeypatch, capsys):
    from keycall._registry import supported_providers
    from keycall._sources import load_targets

    prompts = {}

    def fake_input(prompt):
        prompts["provider"] = prompt
        return "OpenAI"  # any casing works

    import getpass

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(getpass, "getpass", lambda prompt: "sk-canary-interactive-key")

    targets, warnings = load_targets("-")
    for name in supported_providers():
        assert name in prompts["provider"]
    assert targets[0].provider == "openai"
    assert warnings == []
