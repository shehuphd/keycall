"""code_interpreter flag: wire shapes per provider, and its gate.

Fixture responses mirror the live rounds captured 2026-08-22
(project/provider-tool-types.md): OpenAI, Gemini, xAI, and Anthropic each
run code server-side and hand back a code/output pair, but the four wire
shapes diverge more than function calling ever did.
"""

import json

import httpx
import pytest

from keycall import ErrorCode, KeyCall, KeyCallError, Message, TextInput

CANARY = "sk-canary-code-interp-key"


def make_client(provider, handler, **kwargs):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler), **kwargs
    )


def simple_messages():
    return [Message(role="user", content=[TextInput(text="Compute 17 * 23")])]


def capture(response_json):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        captured["count"] = captured.get("count", 0) + 1
        return httpx.Response(200, json=response_json)

    return handler, captured


# --- request-side gating ----------------------------------------------------


def test_code_interpreter_rejected_for_providers_without_it():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must fail before any network call")

    client = make_client("deepseek", handler)
    with pytest.raises(KeyCallError) as excinfo:
        client.generate_text(
            model="deepseek-chat", messages=simple_messages(), code_interpreter=True
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "code interpreter" in excinfo.value.message


# --- OpenAI -------------------------------------------------------------


OPENAI_CODE_INTERPRETER_RESPONSE = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        {
            "id": "ci_1",
            "type": "code_interpreter_call",
            "status": "completed",
            "code": "17 * 23",
            "container_id": "cntr_1",
            "outputs": None,
        },
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [],
                    "text": "The result of 17 * 23 is **391**.",
                }
            ],
        },
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}


def test_openai_code_interpreter_tool_appended():
    handler, captured = capture(OPENAI_CODE_INTERPRETER_RESPONSE)
    make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=simple_messages(), code_interpreter=True
    )
    assert {"type": "code_interpreter", "container": {"type": "auto"}} in captured["body"][
        "tools"
    ]


def test_openai_code_interpreter_call_parsed():
    handler, _ = capture(OPENAI_CODE_INTERPRETER_RESPONSE)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=simple_messages(), code_interpreter=True
    )
    (execution,) = result.code_executions
    assert execution.code == "17 * 23"
    # outputs is null even on a real computation (live-verified 2026-08-22);
    # the human-readable answer arrives as the following text part instead.
    assert execution.output == ""
    assert execution.language == "python"
    assert result.text == "The result of 17 * 23 is **391**."


def test_openai_code_interpreter_outputs_logs_are_joined_when_present():
    response = {
        "model": "gpt-5.1",
        "status": "completed",
        "output": [
            {
                "id": "ci_2",
                "type": "code_interpreter_call",
                "status": "completed",
                "code": "print(391)",
                "container_id": "cntr_2",
                "outputs": [{"type": "logs", "logs": "391\n"}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }
    handler, _ = capture(response)
    result = make_client("openai", handler).generate_text(
        model="gpt-5.1", messages=simple_messages(), code_interpreter=True
    )
    (execution,) = result.code_executions
    assert execution.output == "391\n"


# --- Gemini ---------------------------------------------------------------


GEMINI_CODE_EXECUTION_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "executableCode": {
                            "language": "PYTHON",
                            "code": "result = 17 * 23\nprint(result)",
                            "id": "call_1",
                        }
                    },
                    {
                        "codeExecutionResult": {
                            "outcome": "OUTCOME_OK",
                            "output": "391\n",
                            "id": "call_1",
                        }
                    },
                    {"text": "The result of 17 x 23 is **391**."},
                ],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    "modelVersion": "gemini-3.7-flash",
}


def test_gemini_code_execution_tool_appended():
    handler, captured = capture(GEMINI_CODE_EXECUTION_RESPONSE)
    make_client("gemini", handler).generate_text(
        model="gemini-3.7-flash", messages=simple_messages(), code_interpreter=True
    )
    assert {"codeExecution": {}} in captured["body"]["tools"]


def test_gemini_paired_code_execution_parsed():
    handler, _ = capture(GEMINI_CODE_EXECUTION_RESPONSE)
    result = make_client("gemini", handler).generate_text(
        model="gemini-3.7-flash", messages=simple_messages(), code_interpreter=True
    )
    (execution,) = result.code_executions
    assert execution.code == "result = 17 * 23\nprint(result)"
    assert execution.output == "391\n"
    assert execution.language == "python"
    assert result.text == "The result of 17 x 23 is **391**."


def test_gemini_inline_image_becomes_image_output():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }
    handler, _ = capture(response)
    result = make_client("gemini", handler).generate_text(
        model="gemini-3.7-flash", messages=simple_messages(), code_interpreter=True
    )
    (image,) = [p for p in result.parts if p.kind == "image"]
    assert image.base64_data == "AAAA"
    assert image.media_type == "image/png"


def test_gemini_unmatched_code_execution_result_becomes_unknown_output():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "391\n",
                                "id": "orphan",
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }
    handler, _ = capture(response)
    result = make_client("gemini", handler).generate_text(
        model="gemini-3.7-flash", messages=simple_messages(), code_interpreter=True
    )
    assert not result.code_executions
    (unknown,) = [p for p in result.parts if p.kind == "unknown"]
    assert unknown.provider_kind == "codeExecutionResult"


# --- xAI: delegates to the OpenAI Responses adapter -------------------------


def test_xai_code_interpreter_reroutes_to_the_responses_surface():
    """Rides POST /v1/responses like web_search and reasoning_effort, and
    reuses OpenAIAdapter's parser — verified live 2026-08-22. xAI's shape
    differs from OpenAI's own (no container_id, outputs is a non-null
    logs list), so this also checks the shared parser tolerates that."""
    response = {
        "id": "resp-1",
        "model": "grok-4.3",
        "status": "completed",
        "output": [
            {
                "id": "ci_x1",
                "type": "code_interpreter_call",
                "status": "completed",
                "code": "print(17 * 23)",
                "outputs": [{"type": "logs", "logs": ""}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "391", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
    }
    handler, captured = capture(response)
    client = make_client("xai", handler)
    result = client.generate_text(
        model="grok-4.3", messages=simple_messages(), code_interpreter=True
    )
    client.close()

    assert captured["path"] == "/v1/responses"
    assert {"type": "code_interpreter", "container": {"type": "auto"}} in captured["body"][
        "tools"
    ]
    (execution,) = result.code_executions
    assert execution.code == "print(17 * 23)"
    assert execution.output == ""  # xAI's logs entry was empty in this round
    assert result.text == "391"


# --- Anthropic --------------------------------------------------------------


ANTHROPIC_CODE_EXECUTION_RESPONSE = {
    "model": "claude-opus-5",
    "content": [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "bash_code_execution",
            "input": {"command": "echo $((17 * 23))"},
        },
        {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": {
                "type": "bash_code_execution_result",
                "stdout": "391\n",
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        },
        {"type": "text", "text": "17 * 23 = **391**"},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def test_anthropic_code_execution_sends_beta_header_and_tool():
    handler, captured = capture(ANTHROPIC_CODE_EXECUTION_RESPONSE)
    make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=simple_messages(), code_interpreter=True
    )
    assert captured["headers"]["anthropic-beta"] == "code-execution-2025-08-25"
    assert {"type": "code_execution_20250825", "name": "code_execution"} in captured["body"][
        "tools"
    ]


def test_anthropic_code_execution_omits_beta_header_when_unset():
    handler, captured = capture({"model": "claude-opus-5", "content": [], "usage": {}})
    make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=simple_messages()
    )
    assert "anthropic-beta" not in captured["headers"]


def test_anthropic_bash_code_execution_pair_parsed():
    handler, _ = capture(ANTHROPIC_CODE_EXECUTION_RESPONSE)
    result = make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=simple_messages(), code_interpreter=True
    )
    (execution,) = result.code_executions
    assert execution.code == "echo $((17 * 23))"
    assert execution.output == "391\n"
    assert execution.language == "bash"
    assert result.text == "17 * 23 = **391**"


def test_anthropic_text_editor_code_execution_falls_through_to_unknown():
    response = {
        "model": "claude-opus-5",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_2",
                "name": "text_editor_code_execution",
                "input": {"command": "create", "path": "solve.py"},
            },
            {"type": "text", "text": "Wrote the script."},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    handler, _ = capture(response)
    result = make_client("anthropic", handler).generate_text(
        model="claude-opus-5", messages=simple_messages(), code_interpreter=True
    )
    assert not result.code_executions
    (unknown,) = [p for p in result.parts if p.kind == "unknown"]
    assert unknown.provider_kind == "server_tool_use"
