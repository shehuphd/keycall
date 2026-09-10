"""Prerecorded transcription: four providers, two wire forms, one result.

Verified live 2026-09-02. OpenAI and ElevenLabs take a multipart upload,
Deepgram a raw binary body (or a JSON url), and AssemblyAI runs a job:
raw-binary upload, submit with the speech_models array, poll to
completed. Word timings arrive in seconds everywhere but AssemblyAI
(milliseconds already); every parser converges on TranscriptWord's
millisecond timings. Gemini has no transcription endpoint and refuses
toward generate_text + AudioInput.
"""

import json
import pickle

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    ErrorCode,
    KeyCall,
    KeyCallError,
    TranscriptionJob,
    TranscriptionJobTimeout,
    TranscriptionRequest,
)

CANARY = "sk-canary-transcribe-key"

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


# --- gates and validation, all before the network ---


def test_transcriptionless_provider_refuses_and_names_the_supporting_ones():
    client = make_client("deepseek", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="m", audio=WAV)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    for name in ("openai", "elevenlabs", "deepgram", "assemblyai"):
        assert name in info.value.message


def test_gemini_refusal_points_at_the_audio_input_door():
    client = make_client("gemini", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="gemini-flash-latest", audio=WAV)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "AudioInput" in info.value.message
    assert "generate_text" in info.value.message


def test_request_validation_rejects_malformed_asks():
    with pytest.raises(ValueError):
        TranscriptionRequest(model="m")  # neither audio nor url
    with pytest.raises(ValueError):
        TranscriptionRequest(model="m", data=WAV, url="https://a.example/x.wav")
    with pytest.raises(ValueError):
        TranscriptionRequest(model="m", data=b"")
    with pytest.raises(ValueError):
        TranscriptionRequest(model="  ", data=WAV)


def test_url_audio_refused_where_the_provider_takes_bytes_only():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="whisper-1", url="https://a.example/x.wav")
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    for name in ("assemblyai", "deepgram", "elevenlabs"):
        assert name in info.value.message


def test_diarize_refused_where_the_provider_cannot():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="whisper-1", audio=WAV, diarize=True)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    for name in ("assemblyai", "deepgram", "elevenlabs"):
        assert name in info.value.message


def test_timeout_is_refused_on_one_round_trip_providers():
    client = make_client("openai", refuse_network)
    with pytest.raises(ValueError, match="read_timeout"):
        client.transcribe(model="whisper-1", audio=WAV, timeout=60.0)


def test_timeout_is_required_on_job_shaped_providers():
    client = make_client("assemblyai", refuse_network)
    with pytest.raises(ValueError, match="timeout"):
        client.transcribe(model="universal-2", audio=WAV)


def test_job_handles_refused_where_no_job_exists():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.start_transcription(model="whisper-1", audio=WAV)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "transcribe()" in info.value.message


def test_a_job_never_polls_another_providers_transcription():
    client = make_client("deepgram", refuse_network)
    foreign = TranscriptionJob(provider="assemblyai", model="universal-2", job_id="t-1")
    with pytest.raises(KeyCallError) as info:
        client.check_transcription(foreign)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    with pytest.raises(TypeError):
        client.check_transcription("not a job")


def test_fetching_early_or_from_a_failed_job_is_refused():
    client = make_client("assemblyai", refuse_network)
    running = TranscriptionJob(provider="assemblyai", model="universal-2", job_id="t-1")
    with pytest.raises(ValueError):
        client.fetch_transcription(running)
    failed = TranscriptionJob(
        provider="assemblyai",
        model="universal-2",
        job_id="t-1",
        status="failed",
        provider_status="error",
        error_message="audio was silent",
    )
    with pytest.raises(KeyCallError) as info:
        client.fetch_transcription(failed)
    assert info.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "audio was silent" in info.value.message


def test_unrecognized_audio_asks_for_a_media_type():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="whisper-1", audio=b"\x00\x01\x02\x03garbage")
    assert "media_type" in info.value.message


# --- OpenAI: multipart, whisper-1 words vs gpt-4o text-only ---


def test_openai_whisper_asks_for_word_timings_and_parses_them():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "task": "transcribe",
                "language": "english",
                "duration": 2.64,
                "text": "The quick brown fox.",
                "usage": {"type": "duration", "seconds": 3},
                "words": [
                    {"word": "The", "start": 0.0, "end": 0.16},
                    {"word": "quick", "start": 0.16, "end": 0.4},
                ],
            },
        )

    client = make_client("openai", handler)
    result = client.transcribe(model="whisper-1", audio=WAV, language="en")
    client.close()

    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b"verbose_json" in captured["body"]
    assert b"timestamp_granularities" in captured["body"]
    assert b'name="language"' in captured["body"]
    assert result.text == "The quick brown fox."
    assert result.language == "english"
    assert result.audio_duration_seconds == 2.64
    assert result.words[0].text == "The"
    assert result.words[1].start_ms == 160.0  # seconds converted to ms
    assert result.words[1].end_ms == 400.0
    assert result.words[0].confidence is None  # whisper reports none
    assert result.usage is None


def test_openai_gpt4o_family_stays_on_plain_json_with_token_usage():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "text": "The quick brown fox.",
                "usage": {
                    "type": "tokens",
                    "total_tokens": 148,
                    "input_tokens": 26,
                    "output_tokens": 122,
                },
            },
        )

    client = make_client("openai", handler)
    result = client.transcribe(model="gpt-4o-mini-transcribe", audio=WAV)
    client.close()

    # verbose_json is refused by this family (400 naming json/text,
    # observed live 2026-09-02), so it must not be asked for.
    assert b"verbose_json" not in captured["body"]
    assert result.words == ()
    assert result.usage.input_tokens == 26
    assert result.usage.output_tokens == 122
    assert result.audio_duration_seconds is None


# --- ElevenLabs: multipart, spacing entries filtered, speakers ---


def test_elevenlabs_filters_spacing_entries_and_carries_speakers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "language_code": "eng",
                "language_probability": 0.72,
                "text": "The quick",
                "transcription_id": "tr_1",
                "audio_duration_secs": 2.7,
                "words": [
                    {
                        "text": "The", "type": "word", "start": 0.039, "end": 0.159,
                        "speaker_id": "speaker_0", "logprob": -0.001,
                    },
                    {"text": " ", "type": "spacing", "start": 0.159, "end": 0.2},
                    {
                        "text": "quick", "type": "word", "start": 0.2, "end": 0.44,
                        "speaker_id": "speaker_0", "logprob": -0.002,
                    },
                ],
            },
        )

    client = make_client("elevenlabs", handler)
    result = client.transcribe(model="scribe_v2", audio=WAV, diarize=True)
    client.close()

    assert captured["path"] == "/v1/speech-to-text"
    assert b'name="model_id"' in captured["body"]
    assert b"scribe_v2" in captured["body"]
    assert b'name="diarize"' in captured["body"]
    assert b'name="timestamps_granularity"' in captured["body"]
    assert [w.text for w in result.words] == ["The", "quick"]  # spacing dropped
    assert result.words[0].speaker == "speaker_0"
    assert result.words[0].start_ms == 39.0
    assert result.words[0].confidence is None  # logprob is not a 0-1 score
    assert result.language == "eng"
    assert result.audio_duration_seconds == 2.7


def test_elevenlabs_url_audio_rides_a_multipart_form_without_a_file():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(
            200, json={"language_code": "eng", "text": "hi", "words": []}
        )

    client = make_client("elevenlabs", handler)
    result = client.transcribe(model="scribe_v2", url="https://a.example/x.wav")
    client.close()
    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="source_url"' in captured["body"]
    assert b"https://a.example/x.wav" in captured["body"]
    assert result.text == "hi"


# --- Deepgram: raw binary body or a JSON url, query params ---


def deepgram_payload():
    return {
        "metadata": {"request_id": "req_dg", "duration": 2.645},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "The quick brown fox.",
                            "confidence": 0.999,
                            "words": [
                                {
                                    "word": "the", "punctuated_word": "The",
                                    "start": 0.08, "end": 0.32,
                                    "confidence": 0.76, "speaker": 0,
                                },
                            ],
                        }
                    ]
                }
            ]
        },
    }


def test_deepgram_sends_the_audio_as_a_raw_binary_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json=deepgram_payload())

    client = make_client("deepgram", handler)
    result = client.transcribe(model="nova-3", audio=WAV, diarize=True, language="en")
    client.close()

    assert captured["path"] == "/v1/listen"
    assert captured["params"] == {
        "model": "nova-3", "punctuate": "true", "language": "en", "diarize": "true",
    }
    assert captured["content_type"] == "audio/wav"
    assert captured["body"] == WAV
    assert result.text == "The quick brown fox."
    assert result.confidence == 0.999
    assert result.words[0].text == "The"  # punctuated_word preferred
    assert result.words[0].start_ms == 80.0
    assert result.words[0].speaker == "0"  # int normalized to str
    assert result.audio_duration_seconds == 2.645
    assert result.provider_request_id == "req_dg"


def test_deepgram_url_audio_rides_a_json_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=deepgram_payload())

    client = make_client("deepgram", handler)
    client.transcribe(model="nova-3", url="https://a.example/x.wav")
    client.close()
    assert captured["body"] == {"url": "https://a.example/x.wav"}


# --- AssemblyAI: the job shape ---


def assemblyai_handler(captured, *, statuses=("processing", "completed")):
    remaining = list(statuses)

    def completed_payload(status):
        payload = {"id": "t-1", "status": status}
        if status == "completed":
            payload.update(
                {
                    "text": "The quick brown fox.",
                    "confidence": 0.996,
                    "audio_duration": 3,
                    "language_code": "en",
                    "words": [
                        {
                            "text": "The", "start": 80, "end": 160,
                            "confidence": 0.99, "speaker": "A",
                        },
                    ],
                }
            )
        if status == "error":
            payload["error"] = "download failed"
        return payload

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/upload":
            captured["upload_content_type"] = request.headers.get("content-type", "")
            captured["upload_body"] = request.content
            return httpx.Response(200, json={"upload_url": "https://cdn.aai/private/x"})
        if path == "/v2/transcript" and request.method == "POST":
            captured["submit"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "t-1", "status": "queued"})
        if path == "/v2/transcript/t-1":
            status = remaining.pop(0) if len(remaining) > 1 else statuses[-1]
            return httpx.Response(200, json=completed_payload(status))
        return httpx.Response(404, json={"error": f"no such path {path}"})

    return handler


def test_assemblyai_uploads_submits_and_polls_to_a_result():
    captured = {}
    client = make_client("assemblyai", assemblyai_handler(captured))
    result = client.transcribe(
        model="universal-2", audio=WAV, diarize=True, language="en",
        timeout=30.0, poll_interval=0.01,
    )
    client.close()

    assert captured["upload_content_type"] == "application/octet-stream"
    assert captured["upload_body"] == WAV
    assert captured["submit"] == {
        "audio_url": "https://cdn.aai/private/x",
        "speech_models": ["universal-2"],
        "language_code": "en",
        "speaker_labels": True,
    }
    assert result.text == "The quick brown fox."
    assert result.words[0].start_ms == 80.0  # already milliseconds
    assert result.words[0].speaker == "A"
    assert result.confidence == 0.996
    assert result.audio_duration_seconds == 3.0
    assert result.language == "en"


def test_assemblyai_url_audio_skips_the_upload():
    captured = {}
    client = make_client("assemblyai", assemblyai_handler(captured))
    job = client.start_transcription(model="universal-2", url="https://a.example/x.wav")
    client.close()
    assert "upload_body" not in captured
    assert captured["submit"]["audio_url"] == "https://a.example/x.wav"
    assert job.job_id == "t-1"
    assert job.status == "running"
    assert job.provider_status == "queued"


def test_assemblyai_job_handles_drive_the_loop_and_pickle():
    captured = {}
    client = make_client("assemblyai", assemblyai_handler(captured))
    job = client.start_transcription(model="universal-2", audio=WAV)
    restored = pickle.loads(pickle.dumps(job))
    assert restored == job
    job = client.check_transcription(restored)
    assert job.status in ("running", "finished")
    while job.status == "running":
        job = client.check_transcription(job)
    result = client.fetch_transcription(job)
    assert result.text == "The quick brown fox."
    assert client.check_transcription(job) is job  # ended: no network call
    client.close()


def test_assemblyai_error_status_reports_the_providers_reason():
    captured = {}
    client = make_client(
        "assemblyai", assemblyai_handler(captured, statuses=("error",))
    )
    with pytest.raises(KeyCallError) as info:
        client.transcribe(model="universal-2", audio=WAV, timeout=30.0, poll_interval=0.01)
    client.close()
    assert info.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert "download failed" in info.value.message


def test_assemblyai_timeout_hands_back_the_still_valid_job():
    captured = {}
    client = make_client(
        "assemblyai", assemblyai_handler(captured, statuses=("processing", "processing"))
    )
    with pytest.raises(TranscriptionJobTimeout) as info:
        client.transcribe(model="universal-2", audio=WAV, timeout=0.05, poll_interval=0.01)
    client.close()
    assert info.value.code is ErrorCode.TIMEOUT
    assert info.value.retryable
    assert info.value.job.job_id == "t-1"
    assert info.value.job.status == "running"


# --- async parity ---


@pytest.mark.anyio
async def test_async_sync_provider_transcription():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": "hello", "language": "english", "duration": 1.0,
                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            },
        )

    client = AsyncKeyCall(
        provider="openai", api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )
    result = await client.transcribe(model="whisper-1", audio=WAV)
    await client.close()
    assert result.text == "hello"
    assert result.words[0].end_ms == 500.0


@pytest.mark.anyio
async def test_async_job_provider_transcription():
    captured = {}
    client = AsyncKeyCall(
        provider="assemblyai",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(assemblyai_handler(captured)),
    )
    result = await client.transcribe(
        model="universal-2", audio=WAV, timeout=30.0, poll_interval=0.01
    )
    await client.close()
    assert result.text == "The quick brown fox."


def test_realtime_only_transcription_model_refused_before_the_network():
    """OpenAI lists models only its realtime socket serves (gpt-live-transcribe,
    gpt-realtime-whisper) beside the ones the stored-file endpoint takes, with
    nothing in the listing to tell them apart, and answers the first kind with
    a bare 404. KeyCall splits them on the id and refuses here, naming the
    surface that does serve them."""
    client = make_client("openai", refuse_network)
    for model in ("gpt-live-transcribe", "gpt-realtime-whisper"):
        with pytest.raises(KeyCallError) as excinfo:
            client.transcribe(model=model, audio=WAV)
        assert excinfo.value.code is ErrorCode.MODEL_NOT_SUITABLE
        assert "transcribe_stream()" in excinfo.value.message
    client.close()


def test_a_stored_file_transcription_model_is_not_caught_by_the_family_rule():
    """The families are substrings, so the rule has to leave every model the
    stored-file endpoint does serve alone — including the dated snapshots."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"text": "ok"})

    client = make_client("openai", handler)
    for model in ("whisper-1", "gpt-transcribe", "gpt-4o-mini-transcribe-2025-12-15"):
        assert client.transcribe(model=model, audio=WAV).text == "ok"
    assert len(seen) == 3
    client.close()
