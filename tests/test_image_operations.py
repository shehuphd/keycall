"""Picture inputs and settings on OpenAI, Gemini, and xAI: size, quality,
seed, edits with references and masks, provider_options, and the gates
that refuse an input a provider would ignore.

Live 2026-10-02: OpenAI's gpt-image-1 family takes three fixed sizes and
the gpt-image-2 family any size with sides divisible by 16 (1536x864
accepted); /images/edits takes repeated image[] files and an alpha mask.
Gemini edits through generateContent with inline pictures and takes a
ratio as imageConfig.aspectRatio. xAI edits through /v1/images/edits as
JSON and ignores fields it doesn't know, so seed and mask are refused.
"""

import base64
import email.parser
import email.policy
import json
import struct
import zlib

import httpx
import pytest

from keycall import ErrorCode, ImageInput, ImageOperationRequest, KeyCall, KeyCallError, Operation
from keycall._mask import read_mask

CANARY = "sk-canary-picture-key"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32


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


WHITE_EDIT_MASK = tiny_png(8, 8, lambda x, y: 255 if x < 4 else 0)
OPENAI_IMAGE = {"created": 1, "output_format": "png", "data": [{"b64_json": "QUJD"}], "usage": {"input_tokens": 5, "output_tokens": 9, "total_tokens": 14}}
GEMINI_IMAGE = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": "QUJD"}}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 9, "totalTokenCount": 12}}
XAI_IMAGE = {"data": [{"b64_json": "QUJD", "mime_type": "image/jpeg"}], "usage": {"cost_in_usd_ticks": 400000000}}


class Capture:
    def __init__(self, body):
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(200, json=self.body)

    @property
    def json(self):
        return json.loads(self.requests[-1].content)


def client(provider, handler):
    return KeyCall(provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler))


def refuse_network(request):
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


def form(request):
    raw = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    parts: dict[str, list[tuple[str | None, bytes]]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        parts.setdefault(name, []).append((part.get_filename(), part.get_payload(decode=True)))
    return parts


# --- OpenAI -------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "size", "sent"),
    [
        ("gpt-image-2", "16:9", "1536x864"),
        ("gpt-image-2", "9:16", "864x1536"),
        ("gpt-image-2", "1:1", "1024x1024"),
        ("gpt-image-2", "4:3", "1536x1152"),
        ("gpt-image-2", "2048x1152", "2048x1152"),
        ("gpt-image-1-mini", "3:2", "1536x1024"),
        ("gpt-image-1", "1024x1536", "1024x1536"),
    ],
)
def test_openai_size_converts_a_ratio_by_model_family(model, size, sent):
    capture = Capture(OPENAI_IMAGE)
    with client("openai", capture) as kc:
        kc.generate_image(model=model, prompt="an apple", size=size, quality="low")
    assert capture.json == {"model": model, "prompt": "an apple", "size": sent, "quality": "low"}


@pytest.mark.parametrize(
    ("model", "size", "fragment"),
    [
        ("gpt-image-1-mini", "16:9", "ratios 1:1, 3:2, 2:3"),
        ("gpt-image-1-mini", "1536x864", "one of 1024x1024"),
        ("gpt-image-2", "4:1", "no wider than 3:1"),
        ("gpt-image-2", "1000x1000", "multiples of 16"),
        ("gpt-image-2", "4096x2048", "at most 3840"),
    ],
)
def test_openai_refuses_a_size_the_family_cannot_draw(model, size, fragment):
    with client("openai", refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_image(model=model, prompt="an apple", size=size)
    assert caught.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert fragment in caught.value.message


def test_openai_refuses_seed_naming_where_it_works():
    with client("openai", refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.generate_image(model="gpt-image-1", prompt="an apple", seed=3)
    assert "seed is honoured on: gemini, ideogram" in caught.value.message


def test_openai_edit_sends_files_and_redraws_the_mask_transparent():
    capture = Capture(OPENAI_IMAGE)
    with client("openai", capture) as kc:
        result = kc.edit_image(
            model="gpt-image-1-mini",
            prompt="make it green",
            image=ImageInput(data=JPEG_BYTES),
            reference_images=[ImageInput(data=PNG_BYTES)],
            mask=ImageInput(data=WHITE_EDIT_MASK),
            quality="low",
            size="1:1",
        )
    request = capture.requests[0]
    assert request.url.path.endswith("/images/edits")
    parts = form(request)
    assert [payload for _, payload in parts["image[]"]] == [JPEG_BYTES, PNG_BYTES]
    assert parts["quality"][0][1] == b"low"
    assert parts["size"][0][1] == b"1024x1024"
    mask = parts["mask"][0][1]
    width, height, _depth, colour = struct.unpack(">IIBB", mask[16:26])
    assert (width, height, colour) == (8, 8, 6)
    assert result.operation is Operation.IMAGE_EDIT


def test_openai_provider_options_merge_and_refuse_a_collision():
    capture = Capture(OPENAI_IMAGE)
    with client("openai", capture) as kc:
        kc.generate_image(model="gpt-image-1", prompt="p", provider_options={"background": "transparent"})
    assert capture.json["background"] == "transparent"
    with client("openai", refuse_network) as kc, pytest.raises(KeyCallError, match="set it in one place"):
        kc.generate_image(model="gpt-image-1", prompt="p", provider_options={"model": "other"})


def test_openai_offers_no_upscale_and_names_who_does():
    with client("openai", refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.upscale_image(model="gpt-image-1", image=ImageInput(data=PNG_BYTES))
    assert "upscale operation" in caught.value.message
    assert "ideogram" in caught.value.message


# --- Gemini -------------------------------------------------------------


def test_gemini_ratio_and_seed_ride_generation_config():
    capture = Capture(GEMINI_IMAGE)
    with client("gemini", capture) as kc:
        kc.generate_image(model="gemini-3.1-flash-image", prompt="an apple", size="16:9", seed=5)
    assert capture.json["generationConfig"] == {"imageConfig": {"aspectRatio": "16:9"}, "seed": 5}


def test_gemini_edit_sends_pictures_before_the_instruction():
    capture = Capture(GEMINI_IMAGE)
    with client("gemini", capture) as kc:
        result = kc.edit_image(
            model="gemini-3.1-flash-image",
            prompt="blend them",
            image=ImageInput(data=PNG_BYTES),
            reference_images=[ImageInput(data=JPEG_BYTES)],
        )
    parts = capture.json["contents"][0]["parts"]
    assert [part.get("inlineData", {}).get("mimeType") for part in parts[:2]] == ["image/png", "image/jpeg"]
    assert base64.b64decode(parts[0]["inlineData"]["data"]) == PNG_BYTES
    assert parts[2] == {"text": "blend them"}
    assert result.operation is Operation.IMAGE_EDIT


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"mask": ImageInput(data=WHITE_EDIT_MASK)}, "does not take mask"),
        ({"quality": "low"}, "does not take quality"),
        ({"size": "1024x1024"}, 'aspect ratio like "16:9"'),
        ({"reference_images": [ImageInput(data=PNG_BYTES)] * 14}, "at most 13 reference"),
        ({"image": ImageInput(url="https://example.com/a.png")}, "as bytes"),
    ],
)
def test_gemini_edit_refusals(kwargs, fragment):
    arguments = {"model": "gemini-3.1-flash-image", "prompt": "p", "image": ImageInput(data=PNG_BYTES), **kwargs}
    with client("gemini", refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.edit_image(**arguments)
    assert fragment in caught.value.message


# --- xAI ----------------------------------------------------------------


def test_xai_ratio_and_quality_and_bytes_kept():
    capture = Capture(XAI_IMAGE)
    with client("xai", capture) as kc:
        kc.generate_image(model="grok-imagine-image-2.0", prompt="an apple", size="16:9", quality="low")
    assert capture.json == {
        "prompt": "an apple",
        "model": "grok-imagine-image-2.0",
        "response_format": "b64_json",
        "aspect_ratio": "16:9",
        "quality": "low",
    }


def test_xai_edit_sends_one_picture_as_image_and_several_as_images():
    capture = Capture(XAI_IMAGE)
    with client("xai", capture) as kc:
        kc.edit_image(model="grok-imagine-image-2.0", prompt="p", image=ImageInput(url="https://example.com/a.png"))
        single = capture.json
        kc.edit_image(
            model="grok-imagine-image-2.0",
            prompt="p",
            image=ImageInput(data=PNG_BYTES),
            reference_images=[ImageInput(data=JPEG_BYTES)],
        )
        several = capture.json
    assert capture.requests[0].url.path == "/v1/images/edits"
    assert single["image"] == {"url": "https://example.com/a.png"}
    assert [entry["url"][:22] for entry in several["images"]] == ["data:image/png;base64,", "data:image/jpeg;base64"]


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"seed": 1}, "does not take seed"),
        ({"mask": ImageInput(data=WHITE_EDIT_MASK)}, "does not take mask"),
        ({"reference_images": [ImageInput(data=PNG_BYTES)] * 5}, "at most 4 reference"),
    ],
)
def test_xai_edit_refusals(kwargs, fragment):
    arguments = {"model": "grok-imagine-image-2.0", "prompt": "p", "image": ImageInput(data=PNG_BYTES), **kwargs}
    with client("xai", refuse_network) as kc, pytest.raises(KeyCallError) as caught:
        kc.edit_image(**arguments)
    assert fragment in caught.value.message


# --- requests -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"operation": Operation.IMAGE_EDIT}, "needs prompt"),
        ({"operation": Operation.BACKGROUND_REMOVAL, "seed": 1}, "takes no seed"),
        ({"operation": Operation.IMAGE_EXPAND}, "needs size"),
        ({"operation": Operation.IMAGE_UPSCALE, "factor": 1}, "2 or more"),
        ({"operation": Operation.IMAGE_EDIT, "prompt": "p", "size": "wide"}, "aspect ratio"),
        ({"operation": Operation.IMAGE_EDIT, "prompt": "p", "seed": -1}, "non-negative"),
        ({"operation": Operation.TEXT_GENERATION}, "not a picture operation"),
    ],
)
def test_requests_are_validated_before_any_provider_sees_them(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        ImageOperationRequest(model="m", image=ImageInput(data=PNG_BYTES), **kwargs)


def test_mask_conversion_matches_what_openai_was_sent():
    capture = Capture(OPENAI_IMAGE)
    with client("openai", capture) as kc:
        kc.edit_image(model="gpt-image-1", prompt="p", image=ImageInput(data=PNG_BYTES), mask=ImageInput(data=WHITE_EDIT_MASK))
    mask = form(capture.requests[0])["mask"][0][1]
    raw = zlib.decompress(mask[mask.index(b"IDAT") + 4 : mask.index(b"IEND") - 8])
    alphas = b"".join(raw[row * 33 + 1 :][:32][3::4] for row in range(8))
    _, _, flags = read_mask(WHITE_EDIT_MASK, provider="t", operation="o")
    assert alphas == bytes(0 if flag else 255 for flag in flags)
