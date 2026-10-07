"""Ideogram: pictures and video on a wire where the model is a path
segment and results come back as expiring links.

Live-probed 2026-10-02 against api.ideogram.ai. Every route a model
serves is a catalog row; the adapter reads the row to build the request,
gates the inputs that route honours, sends async=true where the route
takes it, polls the job, and downloads the links from the pinned host
without the key. These tests drive that whole path through a mock
transport, so each assertion is about what went on the wire.
"""

import base64
import email.parser
import email.policy
import json
import struct
import zlib

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    ErrorCode,
    ImageInput,
    KeyCall,
    KeyCallError,
    Message,
    ModelCategory,
    TextBlockOutput,
    TextInput,
    ToolJob,
)
from keycall._mask import read_mask

CANARY = "sk-canary-ideogram-key"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
SIGNED = "https://ideogram.ai/api/images/ephemeral/abc.png?exp=1&signature=x"


def tiny_png(width, height, value):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    rows = b"".join(b"\x00" + bytes(value(x, y) for x in range(width)) for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# KeyCall's convention: white over the area to change.
WHITE_EDIT_MASK = tiny_png(8, 8, lambda x, y: 255 if 2 <= x < 6 and 2 <= y < 6 else 0)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    import keycall._client as client_module

    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)


class Recorder:
    """A mock provider that records every request and answers from a
    script keyed by (method, path)."""

    def __init__(self, routes):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if request.url.host == "ideogram.ai":
            return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/png"})
        answer = self.routes.get(key)
        if answer is None:
            raise AssertionError(f"unexpected request {key}")
        if callable(answer):
            answer = answer(request)
        status, body = answer
        return httpx.Response(status, json=body, headers={"x-request-id": "req-1"})

    def api_requests(self):
        return [r for r in self.requests if r.url.host == "api.ideogram.ai"]


def client(recorder):
    return KeyCall(
        provider="ideogram", api_key=CANARY, httpx_transport=httpx.MockTransport(recorder)
    )


def refuse_network(request):
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


def form(request):
    """A multipart request's parts as {name: [(filename, bytes)]}."""
    raw = (
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    parts: dict[str, list[tuple[str | None, bytes]]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        parts.setdefault(name, []).append((part.get_filename(), part.get_payload(decode=True)))
    return parts


def text(parts, name):
    return parts[name][0][1].decode()


def job(generation_id="gen-1"):
    return (200, {"generation_id": generation_id, "seed": 7})


def completed(*entries):
    return (200, {"generation_id": "gen-1", "status": "completed", "created": "2026-10-02T00:00:00Z", "data": list(entries)})


def picture(url=SIGNED, **extra):
    return {"url": url, "resolution": "1024x1024", "is_image_safe": True, "seed": 7, "prompt": "p", **extra}


POLL = ("GET", "/v2/generations/gen-1")


# --- discovery ----------------------------------------------------------


def test_listing_proves_the_key_with_a_free_dry_run_and_returns_catalog_models():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4-5"): (200, {"object": "price_quote", "usd_micros": 100000})})
    with client(recorder) as kc:
        discovery = kc.list_models(categories=set(ModelCategory), refresh=True)
    sent = recorder.requests[0]
    assert sent.url.params["dry_run"] == "true"
    assert sent.headers["Api-Key"] == CANARY
    ids = {model.id for model in discovery.models}
    assert {"ideogram-4-5", "topaz-bloom-2", "kling-3-standard", "ideogram-1"} <= ids
    upscaler = next(m for m in discovery.models if m.id == "topaz-bloom-2")
    # An upscaler draws nothing from a prompt, so it stays out of a
    # picker of picture generators.
    assert upscaler.categories == frozenset({ModelCategory.IMAGE_EDITING})
    video = next(m for m in discovery.models if m.id == "kling-3-standard")
    assert video.categories == frozenset({ModelCategory.VIDEO_GENERATION})


def test_a_rejected_key_is_typed():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4-5"): (401, {"message": "Access denied. Please verify your API Token is valid."})})
    with client(recorder) as kc, pytest.raises(KeyCallError) as caught:
        kc.list_models(refresh=True)
    assert caught.value.code is ErrorCode.INVALID_API_KEY
    assert "Access denied" in caught.value.message
    # A key paused at a zero balance answers with the same 401, so the
    # message names both causes rather than calling a funded-later key
    # wrong.
    assert "balance reached zero" in caught.value.message
    assert ".." not in caught.value.message


def test_text_generation_is_refused_toward_the_picture_methods():
    with client(refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_text(
            model="ideogram-4-5",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "generate_image()" in caught.value.message


# --- generation ---------------------------------------------------------


def test_generate_sends_async_polls_and_downloads_without_the_key():
    recorder = Recorder(
        {
            ("POST", "/v2/image/generate/ideogram-4-5"): job(),
            POLL: iter_answers([(200, {"generation_id": "gen-1", "status": "pending", "created": "x"}), completed(picture())]),
        }
    )
    with client(recorder) as kc:
        result = kc.generate_image(model="ideogram-4-5", prompt="an apple", quality="low", seed=7, size="1024x1024")
    start = recorder.requests[0]
    body = json.loads(start.content)
    assert body == {"prompt": "an apple", "quality": "low", "seed": 7, "size": "1024x1024", "async": True}
    polls = [r for r in recorder.requests if r.url.path == "/v2/generations/gen-1"]
    assert len(polls) == 2
    download = recorder.requests[-1]
    assert download.url.host == "ideogram.ai"
    assert "Api-Key" not in download.headers
    assert result.operation.value == "image_generation"
    assert result.parts[0].media_type == "image/png"
    assert base64.b64decode(result.parts[0].base64_data) == PNG_BYTES
    assert result.parts[0].url == SIGNED
    assert result.provider_request_id == "req-1"


def iter_answers(answers):
    remaining = list(answers)

    def answer(request):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return answer


@pytest.mark.parametrize(
    ("model", "size", "field", "value"),
    [
        ("ideogram-3", "16:9", "aspect_ratio", "16x9"),
        ("ideogram-3", "1280x768", "resolution", "1280x768"),
        ("gpt-image-2-5-flare", "16:9", "aspect_ratio", "16:9"),
        ("nano-banana-2", "9:16", "aspect_ratio", "9:16"),
        ("z-image", "1024x1024", "resolution", "1024x1024"),
    ],
)
def test_size_is_spelled_the_way_each_route_takes_it(model, size, field, value):
    recorder = Recorder({("POST", f"/v2/image/generate/{model}"): job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.generate_image(model=model, prompt="an apple", size=size)
    start = recorder.requests[0]
    sent = json.loads(start.content) if start.headers["content-type"].startswith("application/json") else {k: text(form(start), k) for k in form(start)}
    assert sent[field] == value


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"model": "ideogram-4-5", "size": "16:9"}, 'pixel size like "1280x768"'),
        ({"model": "nano-banana-2", "size": "1024x1024"}, 'aspect ratio like "16:9"'),
        ({"model": "ideogram-4-5", "quality": "very_low"}, "low, medium, high"),
        ({"model": "ideogram-4", "quality": "low"}, "does not take quality"),
        ({"model": "no-such-model"}, "has no model 'no-such-model'"),
        ({"model": "topaz-bloom-2"}, "has no image generation endpoint"),
    ],
)
def test_generate_refusals_happen_before_any_request(kwargs, fragment):
    with client(refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_image(prompt="an apple", **kwargs)
    assert fragment in caught.value.message


def test_provider_options_are_sent_and_cannot_shadow_a_keycall_parameter():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.generate_image(model="ideogram-4", prompt="an apple", provider_options={"rendering_speed": "turbo"})
    assert json.loads(recorder.requests[0].content)["rendering_speed"] == "turbo"
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="set it in one place"):
        kc.generate_image(model="ideogram-4", prompt="an apple", seed=3, provider_options={"seed": 4})


# --- edits --------------------------------------------------------------


def test_precise_edit_redraws_the_mask_black_for_the_edit_area():
    route = ("POST", "/v2/image/precise-edit/ideogram-4-5")
    recorder = Recorder({route: job(), POLL: completed(picture())})
    with client(recorder) as kc:
        result = kc.edit_image(
            model="ideogram-4-5",
            prompt="make it green",
            image=ImageInput(data=JPEG_BYTES),
            reference_images=[ImageInput(data=PNG_BYTES)],
            mask=ImageInput(data=WHITE_EDIT_MASK),
            quality="very_low",
            seed=7,
        )
    parts = form(recorder.requests[0])
    assert text(parts, "prompt") == "make it green"
    assert text(parts, "quality") == "very_low"
    assert text(parts, "async") == "true"
    assert parts["image"][0] == ("image.jpg", JPEG_BYTES)
    assert parts["reference_images"][0][1] == PNG_BYTES
    _, _, sent_flags = read_mask(parts["mask"][0][1], provider="t", operation="o")
    _, _, original = read_mask(WHITE_EDIT_MASK, provider="t", operation="o")
    assert sent_flags == bytes(1 - flag for flag in original)
    assert result.operation.value == "image_edit"


def test_a_mask_takes_one_reference_slot_on_precise_edit():
    refs = [ImageInput(data=PNG_BYTES)] * 4
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="at most 3 reference image"):
        kc.edit_image(model="ideogram-4-5", prompt="p", image=ImageInput(data=PNG_BYTES), reference_images=refs, mask=ImageInput(data=WHITE_EDIT_MASK))


def test_ideogram_3_masked_edit_goes_to_inpaint_and_unmasked_to_remix():
    recorder = Recorder(
        {
            ("POST", "/v2/image/inpaint/ideogram-3"): job(),
            ("POST", "/v2/image/remix/ideogram-3"): job(),
            POLL: completed(picture()),
        }
    )
    with client(recorder) as kc:
        kc.edit_image(model="ideogram-3", prompt="p", image=ImageInput(data=PNG_BYTES), mask=ImageInput(data=WHITE_EDIT_MASK))
        kc.edit_image(model="ideogram-3", prompt="p", image=ImageInput(data=PNG_BYTES), size="16:9")
    paths = [r.url.path for r in recorder.api_requests() if r.method == "POST"]
    assert paths == ["/v2/image/inpaint/ideogram-3", "/v2/image/remix/ideogram-3"]
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="no size with a mask"):
        kc.edit_image(model="ideogram-3", prompt="p", image=ImageInput(data=PNG_BYTES), mask=ImageInput(data=WHITE_EDIT_MASK), size="16:9")


def test_hosted_models_edit_through_their_generate_route_with_images_files():
    route = ("POST", "/v2/image/generate/nano-banana-2")
    recorder = Recorder({route: job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.edit_image(model="nano-banana-2", prompt="p", image=ImageInput(data=PNG_BYTES), reference_images=[ImageInput(data=JPEG_BYTES)])
    parts = form(recorder.requests[0])
    assert [payload for _, payload in parts["images"]] == [PNG_BYTES, JPEG_BYTES]


def test_an_input_the_route_ignores_is_refused_not_dropped():
    # Ideogram accepts and ignores a seed on remix/ideogram-4 (live
    # 2026-10-02); sending it would hand back a picture that silently
    # disregarded it.
    with client(refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.edit_image(model="ideogram-4", prompt="p", image=ImageInput(data=PNG_BYTES), seed=3)
    assert caught.value.code is ErrorCode.MODEL_NOT_SUITABLE
    assert "ideogram-4-5" in caught.value.message


def test_a_picture_by_url_is_refused():
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="as bytes"):
        kc.edit_image(model="ideogram-4-5", prompt="p", image=ImageInput(url="https://example.com/a.png"))


# --- the other picture operations ---------------------------------------


def test_upscale_spells_the_factor_and_refuses_one_the_model_lacks():
    recorder = Recorder({("POST", "/v2/image/upscale/topaz-standard-2"): job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.upscale_image(model="topaz-standard-2", image=ImageInput(data=PNG_BYTES), factor=4)
    assert text(form(recorder.requests[0]), "upscale_factor") == "x4"
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="upscales by 2, 4, 8, not 3"):
        kc.upscale_image(model="topaz-standard-2", image=ImageInput(data=PNG_BYTES), factor=3)
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="does not take prompt"):
        kc.upscale_image(model="topaz-standard-2", image=ImageInput(data=PNG_BYTES), prompt="crisper")


def test_expand_takes_each_route_s_size_form():
    recorder = Recorder({("POST", "/v2/image/reframe/ideogram-3"): job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.expand_image(model="ideogram-3", image=ImageInput(data=PNG_BYTES), size="1536x640")
    assert text(form(recorder.requests[0]), "resolution") == "1536x640"
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="pixel size"):
        kc.expand_image(model="ideogram-3", image=ImageInput(data=PNG_BYTES), size="16:9")


def test_erase_sends_keycall_s_white_mask_unchanged():
    recorder = Recorder({("POST", "/v2/image/remove-object/ideogram-1"): job(), POLL: completed(picture())})
    with client(recorder) as kc:
        kc.erase_object(model="ideogram-1", image=ImageInput(data=PNG_BYTES), mask=ImageInput(data=WHITE_EDIT_MASK))
    assert form(recorder.requests[0])["mask"][0][1] == WHITE_EDIT_MASK


def test_describe_returns_text_and_sends_no_async_flag():
    recorder = Recorder({("POST", "/v2/image/describe/ideogram-4"): (200, {"description_id": "d", "created": "x", "json_prompt": {"high_level_description": "A red apple.", "compositional_deconstruction": {"background": "white", "elements": []}}})})
    with client(recorder) as kc:
        result = kc.describe_image(model="ideogram-4", image=ImageInput(data=PNG_BYTES))
    assert result.text == "A red apple."
    assert "async" not in form(recorder.requests[0])


def test_layerize_returns_both_pictures_and_the_text_blocks():
    entry = picture(base_image_url="https://ideogram.ai/api/images/ephemeral/base.png?s=1", text_blocks=[{"x": 10, "y": 20, "width": 300, "height": 40, "text": "SALE", "alignment": "center", "formatting": ["bold"], "font_name": "Inter", "font_size": 32, "angle": 0}])
    recorder = Recorder({("POST", "/v2/design/layerize/ideogram-3"): job(), POLL: completed(entry)})
    with client(recorder) as kc:
        result = kc.layerize_image(model="ideogram-3", image=ImageInput(data=PNG_BYTES))
    labels = [getattr(part, "label", None) for part in result.parts if part.kind == "image"]
    assert labels == ["design", "base"]
    block = next(part for part in result.parts if isinstance(part, TextBlockOutput))
    assert (block.text, block.x, block.width, block.font_name, block.formatting) == ("SALE", 10, 300, "Inter", ("bold",))


# --- answers that aren't a clean picture --------------------------------


def test_a_link_on_another_host_is_refused_before_it_is_fetched():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: completed(picture(url="https://evil.example/x.png"))})
    with client(recorder) as kc, pytest.raises(KeyCallError):
        kc.generate_image(model="ideogram-4", prompt="p")
    assert not any(r.url.host == "evil.example" for r in recorder.requests)


def test_every_picture_withheld_by_safety_review_is_an_error_saying_so():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: completed(picture(url=None, is_image_safe=False))})
    with client(recorder) as kc, pytest.raises(KeyCallError, match="safety review withheld every picture"):
        kc.generate_image(model="ideogram-4", prompt="p")


def test_a_partly_withheld_answer_returns_the_rest_with_a_warning():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: completed(picture(), picture(url=None, is_image_safe=False))})
    with client(recorder) as kc:
        result = kc.generate_image(model="ideogram-4", prompt="p")
    assert len(result.parts) == 1
    assert any("withheld 1" in warning for warning in result.warnings)


def test_a_failed_job_carries_the_provider_s_reason():
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: (200, {"generation_id": "gen-1", "status": "failed", "created": "x", "failure_reason": "content_policy_violation"})})
    with client(recorder) as kc, pytest.raises(KeyCallError, match="content_policy_violation"):
        kc.generate_image(model="ideogram-4", prompt="p")


def test_a_job_that_never_finishes_times_out(monkeypatch):
    import keycall._client as client_module

    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(client_module.time, "monotonic", lambda: float(next(clock)))
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): job(), POLL: (200, {"generation_id": "gen-1", "status": "pending", "created": "x"})})
    with client(recorder) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_image(model="ideogram-4", prompt="p")
    assert caught.value.code is ErrorCode.TIMEOUT


@pytest.mark.parametrize(
    ("status", "body", "code", "fragment"),
    [
        (400, {"type": "about:blank", "title": "Bad Request", "detail": "'x' is not one of ['x2', 'x4', 'x8'] - 'upscale_factor'", "status": 400}, ErrorCode.UNSUPPORTED_OPERATION, "is not one of"),
        (400, {"error": "Could not read a source image."}, ErrorCode.UNSUPPORTED_OPERATION, "Could not read"),
        (402, {"error": "Add a payment method and credits to use this API key.", "reject_reason": "insufficient_funds"}, ErrorCode.PERMISSION_DENIED, "insufficient_funds"),
        (429, {"error": "Too many requests in flight.", "reject_reason": "inflight_limit"}, ErrorCode.RATE_LIMITED, "inflight_limit"),
        (422, {}, ErrorCode.UNSUPPORTED_OPERATION, "safety review"),
        (403, {"message": "This account is not authorized for this operation."}, ErrorCode.PERMISSION_DENIED, "not authorized"),
    ],
)
def test_each_error_body_reaches_the_caller(status, body, code, fragment):
    recorder = Recorder({("POST", "/v2/image/generate/ideogram-4"): (status, body)})
    with client(recorder) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_image(model="ideogram-4", prompt="p")
    assert caught.value.code is code
    assert fragment in caught.value.message


# --- tools --------------------------------------------------------------


def test_tools_are_listed_from_the_catalog():
    with client(refuse_network) as kc:
        tools = {tool.name: tool for tool in kc.list_tools()}
    assert "ad-resizer" in tools
    assert tools["colorways"].files["masks"] == 4
    assert "colors" in tools["colorways"].fields


def test_a_tool_run_starts_polls_and_downloads():
    recorder = Recorder({("POST", "/v2/tool/ad-resizer"): job(), POLL: completed(picture(), picture())})
    with client(recorder) as kc:
        started = kc.start_tool(tool="ad-resizer", inputs={"image": ImageInput(data=PNG_BYTES), "resolution": "1080x1920", "num_images": 2})
        assert isinstance(started, ToolJob) and started.status == "running"
        finished = kc.check_tool(started)
        assert finished.status == "succeeded" and len(finished.output_urls) == 2
        result = kc.fetch_tool(finished)
    parts = form(recorder.requests[0])
    assert text(parts, "resolution") == "1080x1920"
    assert parts["image"][0][1] == PNG_BYTES
    assert len(result.parts) == 2


@pytest.mark.parametrize(
    ("tool", "inputs", "fragment"),
    [
        ("ad-resizer", {"image": None}, "needs image, resolution"),
        ("ad-resizer", {"image": "x", "resolution": "1080x1920", "colour": "red"}, "takes no input 'colour'"),
        ("colorways", {"image": "x", "masks": ["m"] * 5, "colors": ["#fff"] * 5}, "at most 4 picture"),
        ("ghost-mannequin", {}, "has no tool 'ghost-mannequin'"),
    ],
)
def test_tool_inputs_are_checked_before_any_request(tool, inputs, fragment):
    pictures = {key: ([ImageInput(data=PNG_BYTES)] * len(value) if isinstance(value, list) and key != "colors" else (ImageInput(data=PNG_BYTES) if value == "x" or value == "m" else value)) for key, value in inputs.items()}
    with client(refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.start_tool(tool=tool, inputs=pictures)
    assert fragment in caught.value.message


def test_a_job_from_another_provider_is_refused():
    with client(refuse_network) as kc, pytest.raises(KeyCallError, match="belongs to provider"):
        kc.check_tool(ToolJob(provider="openai", tool="ad-resizer", job_id="x"))


# --- video --------------------------------------------------------------


def test_video_routes_by_its_inputs_and_spells_the_ratio_16x9():
    recorder = Recorder(
        {
            ("POST", "/v2/video/generate/kling-3-standard-text-to-video"): job(),
            ("POST", "/v2/video/generate/kling-3-standard-image-to-video"): job(),
            ("POST", "/v2/video/generate/seedance-2-reference-to-video"): job(),
        }
    )
    with client(recorder) as kc:
        kc.start_video(model="kling-3-standard", prompt="a rolling apple", duration_seconds=5, aspect_ratio="16:9")
        kc.start_video(model="kling-3-standard", prompt="it rolls", image=ImageInput(data=PNG_BYTES), last_frame=ImageInput(data=PNG_BYTES))
        kc.start_video(model="seedance-2", prompt="@Image1 rolls", reference_images=[ImageInput(data=PNG_BYTES)])
    text_start, image_start, reference_start = recorder.requests
    assert json.loads(text_start.content) == {"prompt": "a rolling apple", "duration": 5, "aspect_ratio": "16x9"}
    parts = form(image_start)
    assert parts["image"][0][1] == PNG_BYTES and parts["end_image"][0][1] == PNG_BYTES
    assert form(reference_start)["reference_images"][0][1] == PNG_BYTES


def test_video_refusals():
    with client(refuse_network) as kc:
        with pytest.raises(KeyCallError, match="keeps the picture's own shape"):
            kc.start_video(model="kling-3-standard", prompt="p", image=ImageInput(data=PNG_BYTES), aspect_ratio="16:9")
        with pytest.raises(KeyCallError, match="does not make video from reference_images"):
            kc.start_video(model="kling-3-standard", prompt="p", reference_images=[ImageInput(data=PNG_BYTES)])


def test_video_job_completes_to_a_pinned_download():
    recorder = Recorder(
        {
            ("POST", "/v2/video/generate/kling-3-standard-text-to-video"): job(),
            POLL: (200, {"generation_id": "gen-1", "status": "completed", "created": "x", "data": [{"object_type": "video.generation", "url": "https://ideogram.ai/api/videos/v.mp4?s=1", "prompt": "p", "duration": 5, "aspect_ratio": "16:9", "resolution": "720p"}]}),
        }
    )
    with client(recorder) as kc:
        started = kc.start_video(model="kling-3-standard", prompt="p")
        finished = kc.check_video(started)
    assert finished.status == "succeeded"
    assert finished.video_url.startswith("https://ideogram.ai/")


# --- async twin ---------------------------------------------------------


@pytest.mark.anyio
async def test_async_client_drives_the_same_job(monkeypatch):
    import anyio

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    recorder = Recorder({("POST", "/v2/image/upscale/auto"): job(), POLL: completed(picture())})
    async with AsyncKeyCall(provider="ideogram", api_key=CANARY, httpx_transport=httpx.MockTransport(recorder)) as kc:
        result = await kc.upscale_image(model="auto", image=ImageInput(data=PNG_BYTES), factor=2)
    assert text(form(recorder.requests[0]), "upscale_factor") == "x2"
    assert base64.b64decode(result.parts[0].base64_data) == PNG_BYTES
