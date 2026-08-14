"""Video generation: the job lifecycle, the gates, and the download rules.

Verified live 2026-08-13 against both supporting providers. Gemini's Veo:
:predictLongRunning answers only an operation name; a running operation
has no `done` field at all; a failed render is an HTTP 200 whose
operation carries an `error` object (high-demand code 14 was refused
eight times in a row across every Veo tier before one job passed);
the finished file lives on the API host behind a same-origin 302 that
needs the API key header on both hops. xAI's Grok Imagine:
POST /v1/videos/generations answers a request_id; status is pending /
done / expired / failed; the finished file is an unsigned public URL on
vidgen.x.ai where sending the credential would be wrong. The other
providers generate no video.
"""

import base64
import json

import httpx
import pytest

from keycall import (
    ErrorCode,
    KeyCall,
    KeyCallError,
    VideoGenerationRequest,
    VideoJob,
    VideoJobTimeout,
)

CANARY = "sk-canary-video-key"

GEMINI_OP = "models/veo-3.1-lite-generate-preview/operations/y5lxdapaztmq"
GEMINI_VIDEO_URI = (
    "https://generativelanguage.googleapis.com/v1beta/files/jn989ri0g72v:download?alt=media"
)


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def gemini_success_operation():
    # Shape captured live 2026-08-13; the full payload is archived in
    # project/veo_operation_final.json.
    return {
        "name": GEMINI_OP,
        "done": True,
        "response": {
            "@type": (
                "type.googleapis.com/google.ai.generativelanguage.v1beta."
                "PredictLongRunningResponse"
            ),
            "generateVideoResponse": {
                "generatedSamples": [{"video": {"uri": GEMINI_VIDEO_URI}}]
            },
        },
    }


def test_prompt_is_validated_before_anything_is_sent():
    with pytest.raises(ValueError):
        VideoGenerationRequest(model="m", prompt="")
    with pytest.raises(ValueError):
        VideoGenerationRequest(model="m", prompt="   ")


def test_gemini_start_names_the_long_running_verb_and_omits_unset_parameters():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"name": GEMINI_OP})

    client = make_client("gemini", handler)
    job = client.start_video(model="veo-3.1-lite-generate-preview", prompt="A boat.")
    client.close()

    assert captured["path"].endswith(":predictLongRunning")
    assert captured["body"] == {"instances": [{"prompt": "A boat."}]}
    assert "parameters" not in captured["body"], "unset parameters must not be sent"
    assert job.job_id == GEMINI_OP
    assert job.status == "running"


def test_gemini_start_sends_parameters_only_when_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"name": GEMINI_OP})

    client = make_client("gemini", handler)
    client.start_video(
        model="veo-3.1-lite-generate-preview",
        prompt="A boat.",
        duration_seconds=4,
        aspect_ratio="16:9",
    )
    client.close()
    assert captured["body"]["parameters"] == {"durationSeconds": 4, "aspectRatio": "16:9"}


def test_gemini_running_operation_has_no_done_field_at_all():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": GEMINI_OP})

    client = make_client("gemini", handler)
    job = VideoJob(provider="gemini", model="veo", job_id=GEMINI_OP)
    checked = client.check_video(job)
    client.close()
    assert checked.status == "running"


def test_gemini_job_failure_is_an_http_200_and_the_message_survives():
    """Captured live 2026-08-13: eight straight high-demand refusals, all
    HTTP 200 with `done: true` and an error object inside the operation."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": GEMINI_OP,
                "done": True,
                "error": {
                    "code": 14,
                    "message": (
                        "This model is currently experiencing high demand. "
                        "Spikes in demand are usually temporary. Please try again later."
                    ),
                },
            },
        )

    client = make_client("gemini", handler)
    job = client.check_video(VideoJob(provider="gemini", model="veo", job_id=GEMINI_OP))
    assert job.status == "failed"
    assert job.provider_status == "error code 14"
    assert "high demand" in (job.error_message or "")
    with pytest.raises(KeyCallError) as excinfo:
        client.fetch_video(job)
    client.close()
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "high demand" in str(excinfo.value)


def test_gemini_full_lifecycle_with_the_same_origin_redirect():
    """The download is a 302 to another path on the provider's own host,
    and both hops need the credential header (verified live 2026-08-13)."""
    video_bytes = b"\x00\x00\x00 ftypisommp4-ish"
    seen_paths = []
    auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path.endswith(":predictLongRunning"):
            return httpx.Response(200, json={"name": GEMINI_OP})
        if "/operations/" in request.url.path:
            return httpx.Response(200, json=gemini_success_operation())
        if ":download" in request.url.path:
            auth_headers.append(request.headers.get("x-goog-api-key"))
            return httpx.Response(
                302,
                headers={
                    "location": "https://generativelanguage.googleapis.com/v1beta/files/hop2"
                },
            )
        assert request.url.path.endswith("/files/hop2")
        auth_headers.append(request.headers.get("x-goog-api-key"))
        return httpx.Response(
            200, content=video_bytes, headers={"content-type": "video/mp4"}
        )

    client = make_client("gemini", handler)
    result = client.generate_video(
        model="veo-3.1-lite-generate-preview",
        prompt="A boat.",
        timeout=5.0,
        poll_interval=0.01,
    )
    client.close()

    clip = result.parts[0]
    assert clip.kind == "video"
    assert clip.media_type == "video/mp4"
    assert base64.b64decode(clip.base64_data) == video_bytes
    assert clip.url == GEMINI_VIDEO_URI
    assert auth_headers == [CANARY, CANARY], "both download hops carry the credential"


def test_gemini_cross_origin_redirect_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        if ":download" in request.url.path:
            return httpx.Response(
                302, headers={"location": "https://evil.example.com/clip.mp4"}
            )
        return httpx.Response(500)

    client = make_client("gemini", handler)
    job = VideoJob(
        provider="gemini",
        model="veo",
        job_id=GEMINI_OP,
        status="succeeded",
        video_url=GEMINI_VIDEO_URI,
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.fetch_video(job)
    client.close()
    assert "cross-origin" in str(excinfo.value)


def test_gemini_download_host_not_its_own_is_refused_before_any_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(200, content=b"x", headers={"content-type": "video/mp4"})

    client = make_client("gemini", handler)
    job = VideoJob(
        provider="gemini",
        model="veo",
        job_id=GEMINI_OP,
        status="succeeded",
        video_url="https://attacker.example.com/steal?creds=please",
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.fetch_video(job)
    client.close()
    assert not calls, "no request may leave for an unexpected host"
    assert "unexpected host" in str(excinfo.value)


def test_xai_start_and_status_map_the_provider_states():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/videos/generations":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"request_id": "ffdac69a-6f5a"})
        assert request.url.path == "/v1/videos/ffdac69a-6f5a"
        return httpx.Response(
            200,
            json={
                "status": "done",
                "video": {
                    "url": "https://vidgen.x.ai/xai-vidgen-bucket/clip.mp4",
                    "duration": 6,
                },
                "model": "grok-imagine-video-1.5",
            },
        )

    client = make_client("xai", handler)
    job = client.start_video(
        model="grok-imagine-video-1.5", prompt="A boat.", duration_seconds=6
    )
    assert captured["body"] == {
        "model": "grok-imagine-video-1.5",
        "prompt": "A boat.",
        "duration": 6,
    }
    assert job.job_id == "ffdac69a-6f5a"
    checked = client.check_video(job)
    client.close()
    assert checked.status == "succeeded"
    assert checked.provider_status == "done"
    assert checked.video_url == "https://vidgen.x.ai/xai-vidgen-bucket/clip.mp4"


def test_xai_expired_maps_to_failed_with_the_provider_word_kept():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "expired"})

    client = make_client("xai", handler)
    job = client.check_video(
        VideoJob(provider="xai", model="grok-imagine-video-1.5", job_id="abc")
    )
    client.close()
    assert job.status == "failed"
    assert job.provider_status == "expired"


def test_xai_download_carries_no_credential_and_is_pinned_to_its_host():
    """The finished file is an unsigned public URL on vidgen.x.ai; the
    auth header must never be sent there (verified live 2026-08-13), and a
    response naming any other host is refused before a request leaves."""
    video_bytes = b"grok-bytes"
    auth_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vidgen.x.ai":
            auth_seen.append(request.headers.get("authorization"))
            return httpx.Response(
                200, content=video_bytes, headers={"content-type": "video/mp4"}
            )
        return httpx.Response(500)

    client = make_client("xai", handler)
    good = VideoJob(
        provider="xai",
        model="grok-imagine-video-1.5",
        job_id="abc",
        status="succeeded",
        video_url="https://vidgen.x.ai/xai-vidgen-bucket/clip.mp4",
    )
    result = client.fetch_video(good)
    assert base64.b64decode(result.parts[0].base64_data) == video_bytes
    assert auth_seen == [None], "the credential must not be sent to the download host"

    elsewhere = VideoJob(
        provider="xai",
        model="grok-imagine-video-1.5",
        job_id="abc",
        status="succeeded",
        video_url="https://vidgen.evil.example.com/clip.mp4",
    )
    with pytest.raises(KeyCallError) as excinfo:
        client.fetch_video(elsewhere)
    client.close()
    assert "unexpected host" in str(excinfo.value)


def test_fetching_an_unfinished_job_is_caller_misuse_not_a_network_call():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    client = make_client("xai", handler)
    running = VideoJob(provider="xai", model="grok-imagine-video-1.5", job_id="abc")
    with pytest.raises(ValueError):
        client.fetch_video(running)
    client.close()
    assert not calls


def test_a_job_from_another_provider_is_refused():
    client = make_client("xai", lambda request: httpx.Response(500))
    foreign = VideoJob(provider="gemini", model="veo", job_id=GEMINI_OP)
    with pytest.raises(KeyCallError) as excinfo:
        client.check_video(foreign)
    client.close()
    assert "belongs to provider" in str(excinfo.value)


def test_generate_video_timeout_carries_the_still_valid_job():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/videos/generations":
            return httpx.Response(200, json={"request_id": "slow-render"})
        return httpx.Response(200, json={"status": "pending"})

    client = make_client("xai", handler)
    with pytest.raises(VideoJobTimeout) as excinfo:
        client.generate_video(
            model="grok-imagine-video-1.5",
            prompt="A boat.",
            timeout=0.05,
            poll_interval=0.01,
        )
    client.close()
    error = excinfo.value
    assert error.code is ErrorCode.TIMEOUT
    assert error.job.job_id == "slow-render"
    assert error.job.status == "running"


def test_the_gate_message_is_built_from_the_video_capability_key(monkeypatch):
    """`.operation` on the raised error is set independently in the same
    statement, so only watching what's asked for can prove the gate reads
    the right capability."""
    import keycall.adapters._base as base_module

    seen = []
    original = base_module.providers_with

    def spy(capability):
        seen.append(capability)
        return original(capability)

    monkeypatch.setattr(base_module, "providers_with", spy)

    client = make_client("anthropic", lambda request: httpx.Response(500))
    with pytest.raises(KeyCallError):
        client.start_video(model="whatever", prompt="Hi.")
    client.close()
    assert seen == ["video_generation"]


def test_providers_without_video_generation_refuse_before_the_network():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    for provider in ("openai", "anthropic", "deepseek", "perplexity", "moonshot"):
        client = make_client(provider, handler)
        with pytest.raises(KeyCallError) as excinfo:
            client.start_video(model="whatever", prompt="Hi.")
        client.close()
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_OPERATION
        assert excinfo.value.operation == "video_generation"
        message = str(excinfo.value)
        assert "cannot generate video" in message
        assert "gemini" in message and "xai" in message

    assert not calls, "the gate must fire before any request goes out"


@pytest.mark.anyio
async def test_async_video_lifecycle_matches_the_sync_client():
    from keycall import AsyncKeyCall

    video_bytes = b"async-video-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/videos/generations":
            return httpx.Response(200, json={"request_id": "async-job"})
        if request.url.path == "/v1/videos/async-job":
            return httpx.Response(
                200,
                json={
                    "status": "done",
                    "video": {"url": "https://vidgen.x.ai/bucket/clip.mp4"},
                },
            )
        assert request.url.host == "vidgen.x.ai"
        return httpx.Response(
            200, content=video_bytes, headers={"content-type": "video/mp4"}
        )

    async with AsyncKeyCall(
        provider="xai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    ) as client:
        result = await client.generate_video(
            model="grok-imagine-video-1.5",
            prompt="A boat.",
            timeout=5.0,
            poll_interval=0.01,
        )

    assert base64.b64decode(result.parts[0].base64_data) == video_bytes
