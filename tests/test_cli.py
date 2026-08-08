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
