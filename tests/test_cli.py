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
    # Promotion must not lose where the model really sat in the raw list.
    assert result.attempts[0].raw_position == 2
