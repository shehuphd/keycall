"""Batch generation: five wire dialects, one job surface.

Verified live 2026-09-02 against all five supporting providers. OpenAI
and Moonshot speak the Files-API dialect (upload a JSONL, create a batch
against it, download output and error files — Moonshot reports per-line
status_code 0 on success where OpenAI reports 200, and serves downloads
as text/plain where OpenAI serves application/octet-stream). Anthropic
takes the requests inline on the create call, mixes models freely, and
streams results as application/x-jsonl from results_url, out of
submission order. Gemini binds the model into the URL, takes requests
inline, and answers results inline on the finished operation object.
xAI creates a named container, adds requests atomically, exposes no
status string (the state is derived from counters), and paginates
results. DeepSeek and Perplexity publish no batch API.
"""

import json
import pickle

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    BatchJob,
    BatchJobTimeout,
    BatchRequest,
    ErrorCode,
    KeyCall,
    KeyCallError,
    Message,
    TextInput,
)

CANARY = "sk-canary-batch-key"


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def two_requests(model="m-1", second_model=None):
    return [
        BatchRequest(model=model, messages=[Message(role="user", content=[TextInput(text="one")])]),
        BatchRequest(
            model=second_model or model, messages=[Message(role="user", content=[TextInput(text="two")])]
        ),
    ]


def openai_response_body(text):
    return {
        "id": "resp_1",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def chat_completion_body(text):
    return {
        "id": "chatcmpl-1",
        "model": "m-1",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


def anthropic_message_body(text):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


# --- gates and validation, all before the network ---


def test_batchless_provider_refuses_and_names_the_supporting_ones():
    client = make_client("deepseek", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.start_batch(two_requests())
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    for name in ("openai", "anthropic", "gemini", "moonshot", "xai"):
        assert name in info.value.message


def test_embedding_batch_gate_excludes_the_generation_only_providers():
    client = make_client("anthropic", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.start_embedding_batch(model="some-embedder", inputs=["a"])
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION


def test_empty_and_mistyped_submissions_are_refused():
    client = make_client("openai", refuse_network)
    with pytest.raises(ValueError):
        client.start_batch([])
    with pytest.raises(TypeError):
        client.start_batch(["not a request"])
    with pytest.raises(ValueError):
        client.start_embedding_batch(model="m", inputs=[])


def test_mixed_models_are_refused_outside_anthropic():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.start_batch(two_requests(model="m-1", second_model="m-2"))
    assert info.value.code is ErrorCode.MODEL_NOT_SUITABLE
    assert "m-1" in info.value.message and "m-2" in info.value.message


def test_a_job_never_polls_another_providers_batch():
    client = make_client("gemini", refuse_network)
    foreign = BatchJob(
        provider="openai",
        job_id="batch_1",
        operation="batch_generation",
        request_keys=("kc-0",),
    )
    with pytest.raises(KeyCallError) as info:
        client.check_batch(foreign)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    with pytest.raises(TypeError):
        client.check_batch("not a job")


def test_ended_jobs_are_returned_as_is_without_a_network_call():
    client = make_client("openai", refuse_network)
    done = BatchJob(
        provider="openai",
        job_id="batch_1",
        operation="batch_generation",
        status="finished",
        request_keys=("kc-0",),
    )
    assert client.check_batch(done) is done
    assert client.cancel_batch(done) is done


def test_fetching_results_early_or_from_a_failed_batch_is_refused():
    client = make_client("openai", refuse_network)
    running = BatchJob(
        provider="openai",
        job_id="batch_1",
        operation="batch_generation",
        request_keys=("kc-0",),
    )
    with pytest.raises(ValueError):
        client.fetch_batch_results(running)
    failed = BatchJob(
        provider="openai",
        job_id="batch_1",
        operation="batch_generation",
        status="failed",
        provider_status="failed",
        error_message="input file was malformed",
        request_keys=("kc-0",),
    )
    with pytest.raises(KeyCallError) as info:
        client.fetch_batch_results(failed)
    assert info.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "input file was malformed" in info.value.message


# --- OpenAI: the Files-API dialect ---


def openai_batch_handler(captured, *, status="validating", output_lines=None,
                         error_lines=None, output_file="file-out", error_file=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            captured["upload"] = request.content
            captured["upload_content_type"] = request.headers.get("content-type", "")
            return httpx.Response(200, json={"id": "file-in", "purpose": "batch"})
        if path == "/v1/batches" and request.method == "POST":
            captured["create"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "batch_1", "status": status})
        if path == "/v1/batches/batch_1":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "completed",
                    "request_counts": {"total": 2, "completed": 1, "failed": 1},
                    "output_file_id": output_file,
                    "error_file_id": error_file,
                },
            )
        if path == "/v1/files/file-out/content":
            body = "\n".join(json.dumps(line) for line in output_lines or [])
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "application/octet-stream"},
            )
        if path == "/v1/files/file-err/content":
            body = "\n".join(json.dumps(line) for line in error_lines or [])
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "application/octet-stream"},
            )
        if path == "/v1/batches/batch_1/cancel":
            return httpx.Response(200, json={"id": "batch_1", "status": "cancelling"})
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}})

    return handler


def test_openai_submits_a_jsonl_upload_then_creates_the_batch_against_it():
    captured = {}
    client = make_client("openai", openai_batch_handler(captured))
    job = client.start_batch(two_requests())
    client.close()

    assert captured["upload_content_type"].startswith("multipart/form-data")
    assert b'"custom_id": "kc-0"' in captured["upload"]
    assert b'"url": "/v1/responses"' in captured["upload"]
    assert b"batch" in captured["upload"]  # the purpose form field
    assert captured["create"] == {
        "input_file_id": "file-in",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    assert job.provider == "openai"
    assert job.job_id == "batch_1"
    assert job.status == "running"
    assert job.provider_status == "validating"
    assert job.request_keys == ("kc-0", "kc-1")
    assert job.model == "m-1"


def test_openai_results_come_back_in_submission_order_with_per_request_errors():
    captured = {}
    output = [
        # Out of order on purpose: the provider promises no ordering.
        {
            "custom_id": "kc-1",
            "response": {"status_code": 200, "body": openai_response_body("two")},
        },
    ]
    errors = [
        {
            "custom_id": "kc-0",
            "error": {"code": "rate_limit_exceeded", "message": "too fast"},
        },
    ]
    client = make_client(
        "openai",
        openai_batch_handler(
            captured, output_lines=output, error_lines=errors, error_file="file-err"
        ),
    )
    job = client.start_batch(two_requests())
    job = client.check_batch(job)
    assert job.status == "finished"
    assert job.counts.total == 2
    assert job.counts.succeeded == 1
    assert job.counts.errored == 1
    assert job.counts.pending == 0

    results = client.fetch_batch_results(job)
    client.close()
    assert [r.key for r in results] == ["kc-0", "kc-1"]
    assert [r.index for r in results] == [0, 1]
    assert not results[0].succeeded
    assert results[0].error_code == "rate_limit_exceeded"
    assert results[0].error_message == "too fast"
    assert results[1].succeeded
    assert results[1].result.text == "two"
    assert results[1].result.usage.input_tokens == 3


def test_openai_batch_with_no_output_file_still_reports_each_error():
    # A batch whose every request failed completes with only an error
    # file; fetching results must report them, never raise.
    captured = {}
    errors = [
        {"custom_id": "kc-0", "error": {"code": "server_error", "message": "boom"}},
        {"custom_id": "kc-1", "error": {"code": "server_error", "message": "boom"}},
    ]
    client = make_client(
        "openai",
        openai_batch_handler(
            captured, error_lines=errors, output_file=None, error_file="file-err"
        ),
    )
    job = client.check_batch(client.start_batch(two_requests()))
    results = client.fetch_batch_results(job)
    client.close()
    assert len(results) == 2
    assert all(not r.succeeded for r in results)
    assert all(r.error_code == "server_error" for r in results)


def test_a_missing_result_reads_as_an_error_not_a_crash():
    captured = {}
    output = [
        {
            "custom_id": "kc-0",
            "response": {"status_code": 200, "body": openai_response_body("one")},
        },
        # kc-1 never appears anywhere.
    ]
    client = make_client("openai", openai_batch_handler(captured, output_lines=output))
    job = client.check_batch(client.start_batch(two_requests()))
    results = client.fetch_batch_results(job)
    client.close()
    assert results[0].succeeded
    assert not results[1].succeeded
    assert "no result" in results[1].error_message


def test_duplicate_keys_keep_the_first_outcome():
    captured = {}
    output = [
        {
            "custom_id": "kc-0",
            "response": {"status_code": 200, "body": openai_response_body("first")},
        },
        {
            "custom_id": "kc-0",
            "response": {"status_code": 200, "body": openai_response_body("second")},
        },
        {
            "custom_id": "kc-1",
            "response": {"status_code": 200, "body": openai_response_body("two")},
        },
    ]
    client = make_client("openai", openai_batch_handler(captured, output_lines=output))
    job = client.check_batch(client.start_batch(two_requests()))
    results = client.fetch_batch_results(job)
    client.close()
    assert results[0].result.text == "first"


def test_openai_embedding_batch_targets_the_embeddings_endpoint():
    captured = {}
    output = [
        {
            "custom_id": "kc-0",
            "response": {
                "status_code": 200,
                "body": {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
            },
        },
    ]
    client = make_client("openai", openai_batch_handler(captured, output_lines=output))
    job = client.start_embedding_batch(model="text-embedding-3-small", inputs=["hello"])
    assert b'"url": "/v1/embeddings"' in captured["upload"]
    assert captured["create"]["endpoint"] == "/v1/embeddings"
    assert job.operation == "batch_embedding"
    results = client.fetch_batch_results(client.check_batch(job))
    client.close()
    assert results[0].succeeded
    assert results[0].result.parts[0].values == (0.1, 0.2)


def test_openai_cancel_reports_the_provider_state():
    captured = {}
    client = make_client("openai", openai_batch_handler(captured))
    job = client.start_batch(two_requests())
    cancelled = client.cancel_batch(job)
    client.close()
    # cancelling is still running: requests in flight may yet finish.
    assert cancelled.status == "running"
    assert cancelled.provider_status == "cancelling"


def test_batch_jobs_survive_pickling_for_cross_process_polling():
    captured = {}
    client = make_client("openai", openai_batch_handler(captured))
    job = client.start_batch(two_requests())
    restored = pickle.loads(pickle.dumps(job))
    assert restored == job
    assert client.check_batch(restored).status == "finished"
    client.close()


def test_generate_batch_polls_to_completion():
    captured = {}
    output = [
        {
            "custom_id": "kc-0",
            "response": {"status_code": 200, "body": openai_response_body("one")},
        },
        {
            "custom_id": "kc-1",
            "response": {"status_code": 200, "body": openai_response_body("two")},
        },
    ]
    client = make_client("openai", openai_batch_handler(captured, output_lines=output))
    results = client.generate_batch(two_requests(), timeout=5.0, poll_interval=0.01)
    client.close()
    assert [r.result.text for r in results] == ["one", "two"]


def test_generate_batch_timeout_hands_back_the_still_valid_job():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "file-in"})
        if path == "/v1/batches" and request.method == "POST":
            return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
        if path == "/v1/batches/batch_1":
            return httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})
        return httpx.Response(404, json={"error": {"message": "no such path"}})

    client = make_client("openai", handler)
    with pytest.raises(BatchJobTimeout) as info:
        client.generate_batch(two_requests(), timeout=0.05, poll_interval=0.01)
    client.close()
    assert info.value.code is ErrorCode.TIMEOUT
    assert info.value.retryable
    assert info.value.job.job_id == "batch_1"
    assert info.value.job.status == "running"


# --- Moonshot: the same dialect, text/plain downloads, status_code 0 ---


def moonshot_handler(captured, *, output_lines):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            captured["upload"] = request.content
            return httpx.Response(200, json={"id": "file-in"})
        if path == "/v1/batches" and request.method == "POST":
            return httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})
        if path == "/v1/batches/batch_1":
            return httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "completed",
                    "request_counts": {
                        "total": len(output_lines),
                        "completed": len(output_lines),
                        "failed": 0,
                    },
                    "output_file_id": "file-out",
                },
            )
        if path == "/v1/files/file-out/content":
            body = "\n".join(json.dumps(line) for line in output_lines)
            # Moonshot serves JSONL downloads as text/plain (observed
            # live 2026-09-02).
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}})

    return handler


def test_moonshot_reads_text_plain_jsonl_and_status_code_zero():
    captured = {}
    output = [
        {
            "custom_id": "kc-1",
            "response": {"status_code": 0, "body": chat_completion_body("two")},
        },
        {
            "custom_id": "kc-0",
            "response": {"status_code": 0, "body": chat_completion_body("one")},
        },
    ]
    client = make_client("moonshot", moonshot_handler(captured, output_lines=output))
    job = client.check_batch(client.start_batch(two_requests()))
    results = client.fetch_batch_results(job)
    client.close()
    assert b'"url": "/v1/chat/completions"' in captured["upload"]
    assert [r.result.text for r in results] == ["one", "two"]


def test_a_single_record_download_parses_even_when_it_arrives_as_json():
    # One JSONL line is itself valid JSON; under text/plain the transport
    # parses it to a dict before the line parser sees it.
    captured = {}
    output = [
        {
            "custom_id": "kc-0",
            "response": {"status_code": 0, "body": chat_completion_body("only")},
        },
    ]
    client = make_client("moonshot", moonshot_handler(captured, output_lines=output))
    job = client.check_batch(
        client.start_batch(
            [BatchRequest(model="m-1", messages=[Message(role="user", content=[TextInput(text="hi")])])]
        )
    )
    results = client.fetch_batch_results(job)
    client.close()
    assert len(results) == 1
    assert results[0].succeeded
    assert results[0].result.text == "only"


# --- Anthropic: inline requests, mixed models, results_url JSONL ---


def anthropic_handler(captured, *, results_lines=(), results_host_path=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/messages/batches" and request.method == "POST":
            captured["create"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 2,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                        "expired": 0,
                    },
                },
            )
        if path == "/v1/messages/batches/msgbatch_1":
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "processing_status": "ended",
                    "request_counts": {
                        "processing": 0,
                        "succeeded": 1,
                        "errored": 1,
                        "canceled": 0,
                        "expired": 0,
                    },
                    "results_url": results_host_path
                    or "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results",
                },
            )
        if path == "/v1/messages/batches/msgbatch_1/results":
            body = "\n".join(json.dumps(line) for line in results_lines)
            # Anthropic serves results as application/x-jsonl (observed
            # live 2026-09-02).
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "application/x-jsonl"},
            )
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}})

    return handler


def test_anthropic_takes_mixed_models_inline_on_the_create_call():
    captured = {}
    client = make_client("anthropic", anthropic_handler(captured))
    job = client.start_batch(two_requests(model="claude-a", second_model="claude-b"))
    client.close()
    body = captured["create"]
    assert [entry["custom_id"] for entry in body["requests"]] == ["kc-0", "kc-1"]
    assert body["requests"][0]["params"]["model"] == "claude-a"
    assert body["requests"][1]["params"]["model"] == "claude-b"
    assert job.model is None  # mixed batch binds no single model
    assert job.status == "running"
    assert job.counts.pending == 2


def test_anthropic_results_read_the_nested_error_envelope_in_any_order():
    results = [
        {
            "custom_id": "kc-1",
            "result": {
                "type": "errored",
                # The per-request error nests one envelope deeper than an
                # HTTP error body (observed live 2026-09-02).
                "error": {
                    "type": "error",
                    "error": {"type": "billing_error", "message": "no credits"},
                },
            },
        },
        {
            "custom_id": "kc-0",
            "result": {"type": "succeeded", "message": anthropic_message_body("one")},
        },
    ]
    captured = {}
    client = make_client("anthropic", anthropic_handler(captured, results_lines=results))
    job = client.check_batch(client.start_batch(two_requests()))
    assert job.status == "finished"
    assert job.counts.succeeded == 1
    assert job.counts.errored == 1
    fetched = client.fetch_batch_results(job)
    client.close()
    assert fetched[0].succeeded
    assert fetched[0].result.text == "one"
    assert not fetched[1].succeeded
    assert fetched[1].error_code == "permission_denied"
    assert fetched[1].error_message == "no credits"


def test_anthropic_refuses_a_results_url_off_its_own_host():
    captured = {}
    client = make_client(
        "anthropic",
        anthropic_handler(
            captured, results_host_path="https://evil.example.com/steal-the-key"
        ),
    )
    job = client.start_batch(two_requests())
    with pytest.raises(KeyCallError) as info:
        client.check_batch(job)
    client.close()
    assert info.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE
    assert "results_url" in info.value.message


# --- Gemini: model in the URL, inline requests, inline results ---


def gemini_generation_operation(*, state, inlined=None, stats=None):
    payload = {
        "name": "batches/b1",
        "metadata": {"state": state, "batchStats": stats or {}},
    }
    if inlined is not None:
        payload["response"] = {"inlinedResponses": {"inlinedResponses": inlined}}
    return payload


def test_gemini_binds_the_model_into_the_url_and_echoes_metadata_keys():
    captured = {}
    inlined = [
        {
            "metadata": {"key": "kc-1"},
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": "two"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
            },
        },
        {
            "metadata": {"key": "kc-0"},
            "error": {"status": "INVALID_ARGUMENT", "message": "bad request"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("models/m-1:batchGenerateContent"):
            captured["create"] = json.loads(request.content)
            return httpx.Response(
                200,
                json=gemini_generation_operation(
                    state="BATCH_STATE_PENDING",
                    stats={"requestCount": "2", "pendingRequestCount": "2"},
                ),
            )
        if path.endswith("/batches/b1"):
            return httpx.Response(
                200,
                json=gemini_generation_operation(
                    state="BATCH_STATE_SUCCEEDED",
                    inlined=inlined,
                    stats={
                        "requestCount": "2",
                        "successfulRequestCount": "1",
                        "failedRequestCount": "1",
                    },
                ),
            )
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}})

    client = make_client("gemini", handler)
    job = client.start_batch(two_requests())
    assert job.job_id == "batches/b1"
    assert job.status == "running"
    # Gemini reports counters as strings; they normalize to ints.
    assert job.counts.total == 2
    requests_sent = captured["create"]["batch"]["input_config"]["requests"]["requests"]
    assert [entry["metadata"]["key"] for entry in requests_sent] == ["kc-0", "kc-1"]

    job = client.check_batch(job)
    assert job.status == "finished"
    results = client.fetch_batch_results(job)
    client.close()
    assert not results[0].succeeded
    assert results[0].error_code == "INVALID_ARGUMENT"
    assert results[1].succeeded
    assert results[1].result.text == "two"


def test_gemini_embedding_batch_uses_the_async_embed_verb_and_inline_vectors():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("models/embed-1:asyncBatchEmbedContent"):
            captured["create"] = json.loads(request.content)
            return httpx.Response(
                200,
                json=gemini_generation_operation(state="BATCH_STATE_PENDING"),
            )
        if path.endswith("/batches/b1"):
            return httpx.Response(
                200,
                json=gemini_generation_operation(
                    state="BATCH_STATE_SUCCEEDED",
                    inlined=[
                        {
                            "metadata": {"key": "kc-0"},
                            "response": {"embedding": {"values": [0.5, 0.6]}},
                        }
                    ],
                ),
            )
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}})

    client = make_client("gemini", handler)
    job = client.start_embedding_batch(model="embed-1", inputs=["hello"])
    item = captured["create"]["batch"]["input_config"]["requests"]["requests"][0]
    assert item["request"]["model"] == "models/embed-1"
    results = client.fetch_batch_results(client.check_batch(job))
    client.close()
    assert results[0].succeeded
    assert results[0].result.parts[0].values == (0.5, 0.6)


def test_gemini_results_fall_back_to_position_when_a_key_is_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("models/m-1:batchGenerateContent"):
            return httpx.Response(
                200, json=gemini_generation_operation(state="BATCH_STATE_PENDING")
            )
        return httpx.Response(
            200,
            json=gemini_generation_operation(
                state="BATCH_STATE_SUCCEEDED",
                inlined=[
                    {
                        "response": {
                            "candidates": [
                                {
                                    "content": {"parts": [{"text": "one"}]},
                                    "finishReason": "STOP",
                                }
                            ]
                        }
                    }
                ],
            ),
        )

    client = make_client("gemini", handler)
    job = client.check_batch(
        client.start_batch(
            [BatchRequest(model="m-1", messages=[Message(role="user", content=[TextInput(text="hi")])])]
        )
    )
    results = client.fetch_batch_results(job)
    client.close()
    assert results[0].succeeded
    assert results[0].result.text == "one"


def test_gemini_cancel_reports_only_that_cancellation_was_asked_for():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("models/m-1:batchGenerateContent"):
            return httpx.Response(
                200, json=gemini_generation_operation(state="BATCH_STATE_PENDING")
            )
        if request.url.path.endswith("/batches/b1:cancel"):
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": {"message": "no such path"}})

    client = make_client("gemini", handler)
    job = client.start_batch(two_requests())
    cancelled = client.cancel_batch(job)
    client.close()
    assert cancelled.provider_status == "cancelling"
    assert cancelled.status == "running"


# --- xAI: named container, atomic add, derived status, pagination ---


def xai_state(*, requests=2, pending=0, success=0, error=0, cancelled=0, cancel_time=None):
    payload = {
        "batch_id": "xb-1",
        "state": {
            "num_requests": requests,
            "num_pending": pending,
            "num_success": success,
            "num_error": error,
            "num_cancelled": cancelled,
        },
    }
    if cancel_time:
        payload["cancel_time"] = cancel_time
    return payload


def test_xai_creates_a_container_then_adds_all_requests_atomically():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/batches" and request.method == "POST":
            captured["create"] = json.loads(request.content)
            return httpx.Response(200, json={"batch_id": "xb-1"})
        if path == "/v1/batches/xb-1/requests":
            captured["add"] = json.loads(request.content)
            return httpx.Response(200, json=None)
        return httpx.Response(404, json={"error": f"no such path {path}"})

    client = make_client("xai", handler)
    job = client.start_batch(two_requests())
    client.close()
    assert captured["create"] == {"name": "keycall batch"}
    added = captured["add"]["batch_requests"]
    assert [entry["batch_request_id"] for entry in added] == ["kc-0", "kc-1"]
    assert added[0]["batch_request"]["chat_get_completion"]["model"] == "m-1"
    assert job.job_id == "xb-1"
    assert job.status == "running"
    assert job.counts.total == 2
    assert job.counts.pending == 2


def test_xai_status_is_derived_from_the_counters():
    responses = iter(
        [
            xai_state(pending=1, success=1),
            xai_state(success=1, error=1),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/batches/xb-1":
            return httpx.Response(200, json=next(responses))
        return httpx.Response(404, json={"error": "no such path"})

    client = make_client("xai", handler)
    job = BatchJob(
        provider="xai",
        job_id="xb-1",
        operation="batch_generation",
        model="m-1",
        request_keys=("kc-0", "kc-1"),
    )
    job = client.check_batch(job)
    assert job.status == "running"
    job = client.check_batch(job)
    client.close()
    assert job.status == "finished"
    assert job.counts.succeeded == 1
    assert job.counts.errored == 1


def test_xai_a_cancel_time_reads_as_cancelled():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=xai_state(pending=1, cancel_time="2026-09-02T12:00:00Z")
        )

    client = make_client("xai", handler)
    job = BatchJob(
        provider="xai",
        job_id="xb-1",
        operation="batch_generation",
        model="m-1",
        request_keys=("kc-0",),
    )
    job = client.check_batch(job)
    client.close()
    assert job.status == "cancelled"


def test_xai_pages_through_results_with_the_pagination_token():
    captured = {"tokens": []}

    def result_entry(key, text):
        return {
            "batch_request_id": key,
            "batch_result": {
                "response": {"chat_get_completion": chat_completion_body(text)}
            },
        }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/batches/xb-1/results":
            token = request.url.params.get("pagination_token")
            captured["tokens"].append(token)
            if token is None:
                return httpx.Response(
                    200,
                    json={
                        "results": [result_entry("kc-1", "two")],
                        "pagination_token": "page-2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "results": [
                        result_entry("kc-0", "one"),
                        {"batch_request_id": "kc-2", "error_message": "model refused"},
                    ]
                },
            )
        return httpx.Response(404, json={"error": "no such path"})

    client = make_client("xai", handler)
    job = BatchJob(
        provider="xai",
        job_id="xb-1",
        operation="batch_generation",
        model="m-1",
        status="finished",
        request_keys=("kc-0", "kc-1", "kc-2"),
    )
    results = client.fetch_batch_results(job)
    client.close()
    assert captured["tokens"] == [None, "page-2"]
    assert [r.key for r in results] == ["kc-0", "kc-1", "kc-2"]
    assert results[0].result.text == "one"
    assert results[1].result.text == "two"
    assert not results[2].succeeded
    assert results[2].error_message == "model refused"


# --- async parity ---


@pytest.mark.anyio
async def test_async_batch_loop_matches_the_sync_one():
    captured = {}
    output = [
        {
            "custom_id": "kc-1",
            "response": {"status_code": 200, "body": openai_response_body("two")},
        },
        {
            "custom_id": "kc-0",
            "response": {"status_code": 200, "body": openai_response_body("one")},
        },
    ]
    client = AsyncKeyCall(
        provider="openai",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(
            openai_batch_handler(captured, output_lines=output)
        ),
    )
    job = await client.start_batch(two_requests())
    assert job.status == "running"
    job = await client.check_batch(job)
    assert job.status == "finished"
    results = await client.fetch_batch_results(job)
    await client.close()
    assert [r.result.text for r in results] == ["one", "two"]


@pytest.mark.anyio
async def test_async_generate_batch_times_out_with_the_job_attached():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "file-in"})
        if path == "/v1/batches" and request.method == "POST":
            return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
        if path == "/v1/batches/batch_1":
            return httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})
        return httpx.Response(404, json={"error": {"message": "no such path"}})

    client = AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BatchJobTimeout) as info:
        await client.generate_batch(two_requests(), timeout=0.05, poll_interval=0.01)
    await client.close()
    assert info.value.job.job_id == "batch_1"
