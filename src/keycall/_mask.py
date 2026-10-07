"""Edit-mask conversion between KeyCall's one convention and each
provider's own.

KeyCall takes a mask as a PNG the same size as the picture being edited,
where white marks the area to change and black the area to keep. That is
the convention most image providers use; the ones that differ get the
mask redrawn here: Ideogram wants black for the area to change, OpenAI
wants it fully transparent. Reading and writing the PNG is done with the
standard library alone (zlib and struct), so a mask adds no dependency.
"""

from __future__ import annotations

import struct
import zlib

from ._errors import ErrorCode, KeyCallError

__all__ = ["mask_to_alpha_edit", "mask_to_black_edit", "read_mask"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Bytes per pixel sample set, by PNG colour type at 8 bits per channel.
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
# A pixel at or above this luminance marks the area to change. Masks are
# meant to be pure black and white; the midpoint split settles anything
# an encoder or a resize left in between, the same rounding Ideogram
# documents for its own masks.
_THRESHOLD = 128
# A mask is an uncompressed byte per pixel once read. 64 megapixels
# covers every provider's largest accepted picture with room to spare
# and bounds what a hostile PNG header can make this module allocate.
_MAX_PIXELS = 64_000_000


def _refuse(message: str, *, provider: str, operation: str) -> KeyCallError:
    return KeyCallError(
        message,
        code=ErrorCode.UNSUPPORTED_OPERATION,
        provider=provider,
        operation=operation,
    )


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    chunks = []
    offset = len(_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if len(body) != length:
            raise ValueError("truncated chunk")
        chunks.append((kind, body))
        offset += 12 + length
        if kind == b"IEND":
            break
    return chunks


def _unfilter(raw: bytes, *, width: int, height: int, bpp: int) -> list[bytes]:
    """Undo PNG's per-row filters. Returns one bytes object per row."""
    stride = width * bpp
    if len(raw) < (stride + 1) * height:
        raise ValueError("pixel data shorter than the header says")
    rows: list[bytes] = []
    previous = bytes(stride)
    position = 0
    for _ in range(height):
        kind = raw[position]
        line = bytearray(raw[position + 1 : position + 1 + stride])
        position += stride + 1
        if kind in (2, 4) and not any(line):
            # An all-zero Up or Paeth row repeats the row above it, which
            # is most rows of a mask: large flat areas. Copying it skips
            # the per-byte loop where a mask spends nearly all its time.
            rows.append(previous)
            continue
        if kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif kind == 2:
            line = bytearray((a + b) & 255 for a, b in zip(line, previous))
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 255
        elif kind == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up = previous[i]
                upper_left = previous[i - bpp] if i >= bpp else 0
                estimate = left + up - upper_left
                near_left = abs(estimate - left)
                near_up = abs(estimate - up)
                near_corner = abs(estimate - upper_left)
                if near_left <= near_up and near_left <= near_corner:
                    predictor = left
                elif near_up <= near_corner:
                    predictor = up
                else:
                    predictor = upper_left
                line[i] = (line[i] + predictor) & 255
        elif kind != 0:
            raise ValueError(f"unknown row filter {kind}")
        previous = bytes(line)
        rows.append(previous)
    return rows


def read_mask(data: bytes, *, provider: str, operation: str) -> tuple[int, int, bytes]:
    """Read a mask PNG into (width, height, flags), one byte per pixel:
    1 where the picture may change, 0 where it must stay. Anything that
    isn't a PNG this module can read is refused with the reason, before
    any request is sent."""
    if not data.startswith(_SIGNATURE):
        raise _refuse(
            "the mask must be a PNG; convert it and send white for the area "
            "to change and black for the area to keep",
            provider=provider,
            operation=operation,
        )
    try:
        chunks = _chunks(data)
        header = next(body for kind, body in chunks if kind == b"IHDR")
        width, height, depth, colour, _compression, _filter, interlace = struct.unpack(
            ">IIBBBBB", header
        )
    except (StopIteration, ValueError, struct.error):
        raise _refuse(
            "the mask PNG could not be read; it is truncated or malformed",
            provider=provider,
            operation=operation,
        ) from None
    if colour not in _CHANNELS or depth != 8:
        raise _refuse(
            f"the mask PNG uses {depth} bits per channel (colour type {colour}); "
            "save it as an 8-bit greyscale, RGB, or RGBA PNG",
            provider=provider,
            operation=operation,
        )
    if interlace:
        raise _refuse(
            "the mask PNG is interlaced; save it without interlacing",
            provider=provider,
            operation=operation,
        )
    if width == 0 or height == 0 or width * height > _MAX_PIXELS:
        raise _refuse(
            f"the mask is {width}x{height}; KeyCall reads masks up to "
            f"{_MAX_PIXELS // 1_000_000} megapixels",
            provider=provider,
            operation=operation,
        )
    bpp = _CHANNELS[colour]
    try:
        # The declared size bounds the inflate: a small file can't expand
        # past the pixels its own header claims.
        inflater = zlib.decompressobj()
        raw = inflater.decompress(
            b"".join(body for kind, body in chunks if kind == b"IDAT"),
            (width * bpp + 1) * height,
        )
        rows = _unfilter(raw, width=width, height=height, bpp=bpp)
    except (ValueError, zlib.error):
        raise _refuse(
            "the mask PNG could not be read; its pixel data is malformed",
            provider=provider,
            operation=operation,
        ) from None

    palette = next((body for kind, body in chunks if kind == b"PLTE"), b"")
    flags = bytearray(width * height)
    position = 0
    for row in rows:
        if colour == 0:
            values = row
        elif colour == 4:
            values = row[0::2]
        elif colour == 3:
            # Palette entries are RGB triples; an index past the table is
            # treated as black rather than read out of bounds.
            luma = [
                (palette[i] * 299 + palette[i + 1] * 587 + palette[i + 2] * 114) // 1000
                for i in range(0, len(palette) - 2, 3)
            ]
            values = bytes(luma[index] if index < len(luma) else 0 for index in row)
        else:
            red, green, blue = row[0::bpp], row[1::bpp], row[2::bpp]
            values = bytes(
                (r * 299 + g * 587 + b * 114) // 1000 for r, g, b in zip(red, green, blue)
            )
        flags[position : position + width] = bytes(
            1 if value >= _THRESHOLD else 0 for value in values
        )
        position += width
    return width, height, bytes(flags)


def _png(width: int, height: int, colour: int, rows: list[bytes]) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    raw = b"".join(b"\x00" + row for row in rows)
    return (
        _SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _require_both(flags: bytes, *, provider: str, operation: str) -> None:
    if 1 not in flags or 0 not in flags:
        marked = "the whole picture" if 1 in flags else "nothing"
        raise _refuse(
            f"the mask marks {marked} for change; a mask needs a white area "
            "to change and a black area to keep. To change the whole picture, "
            "send no mask",
            provider=provider,
            operation=operation,
        )


def mask_to_black_edit(data: bytes, *, provider: str, operation: str) -> bytes:
    """KeyCall's mask redrawn as a greyscale PNG where black marks the
    area to change and white the area to keep (Ideogram's convention)."""
    width, height, flags = read_mask(data, provider=provider, operation=operation)
    _require_both(flags, provider=provider, operation=operation)
    table = bytes([255, 0]) + bytes(254)
    rows = [
        flags[start : start + width].translate(table)
        for start in range(0, width * height, width)
    ]
    return _png(width, height, 0, rows)


def mask_to_alpha_edit(data: bytes, *, provider: str, operation: str) -> bytes:
    """KeyCall's mask redrawn as an RGBA PNG that is fully transparent
    over the area to change and opaque black elsewhere (OpenAI's
    convention)."""
    width, height, flags = read_mask(data, provider=provider, operation=operation)
    _require_both(flags, provider=provider, operation=operation)
    keep, change = b"\x00\x00\x00\xff", b"\x00\x00\x00\x00"
    rows = [
        b"".join(change if flag else keep for flag in flags[start : start + width])
        for start in range(0, width * height, width)
    ]
    return _png(width, height, 6, rows)
