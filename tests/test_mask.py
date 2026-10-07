"""Edit masks: KeyCall's one convention (white marks the area to change)
read from any 8-bit PNG and redrawn for the providers whose own
convention differs. Malformed or ambiguous masks are refused before any
request, with the reason."""

import struct
import zlib

import pytest

from keycall import ErrorCode, KeyCallError
from keycall._mask import mask_to_alpha_edit, mask_to_black_edit, read_mask

W, H = 48, 32
# The edit area: a rectangle away from every edge.
INSIDE = (12, 8, 36, 24)


def inside(x, y):
    left, top, right, bottom = INSIDE
    return left <= x < right and top <= y < bottom


def png(width, height, colour, rows, *, row_filter=0, depth=8, interlace=0, plte=b""):
    """A PNG built by hand so every colour type and row filter is
    exercised, not only what one encoder happens to emit."""
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
    encoded = []
    previous = bytes(width * bpp)
    for row in rows:
        if row_filter == 0:
            filtered = row
        elif row_filter == 1:
            filtered = bytes(
                (row[i] - (row[i - bpp] if i >= bpp else 0)) & 255 for i in range(len(row))
            )
        elif row_filter == 2:
            filtered = bytes((a - b) & 255 for a, b in zip(row, previous))
        elif row_filter == 3:
            filtered = bytes(
                (row[i] - (((row[i - bpp] if i >= bpp else 0) + previous[i]) >> 1)) & 255
                for i in range(len(row))
            )
        else:

            def paeth(a, b, c):
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                return a if pa <= pb and pa <= pc else (b if pb <= pc else c)

            filtered = bytes(
                (
                    row[i]
                    - paeth(
                        row[i - bpp] if i >= bpp else 0,
                        previous[i],
                        previous[i - bpp] if i >= bpp else 0,
                    )
                )
                & 255
                for i in range(len(row))
            )
        encoded.append(bytes([row_filter]) + filtered)
        previous = row

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, interlace)
    body = chunk(b"IHDR", header)
    if plte:
        body += chunk(b"PLTE", plte)
    return (
        b"\x89PNG\r\n\x1a\n"
        + body
        + chunk(b"IDAT", zlib.compress(b"".join(encoded)))
        + chunk(b"IEND", b"")
    )


def grey_rows(on=255, off=0):
    return [bytes(on if inside(x, y) else off for x in range(W)) for y in range(H)]


EXPECTED = bytes(1 if inside(x, y) else 0 for y in range(H) for x in range(W))


def read(data):
    return read_mask(data, provider="test", operation="image_edit")


@pytest.mark.parametrize("row_filter", [0, 1, 2, 3, 4])
def test_every_row_filter_reads_the_same_mask(row_filter):
    assert read(png(W, H, 0, grey_rows(), row_filter=row_filter)) == (W, H, EXPECTED)


@pytest.mark.parametrize(
    ("colour", "pixel_on", "pixel_off"),
    [
        (2, (255, 255, 255), (0, 0, 0)),
        (4, (255, 255), (0, 255)),
        (6, (250, 250, 250, 255), (3, 3, 3, 255)),
    ],
)
def test_colour_types_read_by_luminance(colour, pixel_on, pixel_off):
    rows = [
        bytes(v for x in range(W) for v in (pixel_on if inside(x, y) else pixel_off))
        for y in range(H)
    ]
    for row_filter in (0, 4):
        assert read(png(W, H, colour, rows, row_filter=row_filter))[2] == EXPECTED


def test_palette_mask_reads_through_its_palette():
    rows = [bytes(1 if inside(x, y) else 0 for x in range(W)) for y in range(H)]
    palette = bytes((0, 0, 0, 255, 255, 255))
    assert read(png(W, H, 3, rows, plte=palette))[2] == EXPECTED


def test_grey_values_split_at_the_midpoint():
    # Anti-aliased edges and resize residue fall on one side or the other,
    # the rounding Ideogram documents for its own masks.
    rows = [bytes((128 if inside(x, y) else 127) for x in range(W)) for y in range(H)]
    assert read(png(W, H, 0, rows))[2] == EXPECTED


def test_black_edit_conversion_inverts_and_keeps_size():
    converted = mask_to_black_edit(png(W, H, 0, grey_rows()), provider="t", operation="o")
    width, height, flags = read(converted)
    assert (width, height) == (W, H)
    assert flags == bytes(1 - flag for flag in EXPECTED)


def test_alpha_edit_conversion_is_transparent_over_the_edit_area():
    converted = mask_to_alpha_edit(png(W, H, 0, grey_rows()), provider="t", operation="o")
    assert converted.startswith(b"\x89PNG")
    header = converted[16:29]
    width, height, depth, colour = struct.unpack(">IIBB", header[:10])
    assert (width, height, depth, colour) == (W, H, 8, 6)
    raw = zlib.decompress(converted[converted.index(b"IDAT") + 4 : converted.index(b"IEND") - 8])
    alphas = b"".join(raw[r * (W * 4 + 1) + 1 :][: W * 4][3::4] for r in range(H))
    assert alphas == bytes(0 if flag else 255 for flag in EXPECTED)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (b"\xff\xd8\xff\xe0 a jpeg", "must be a PNG"),
        (b"\x89PNG\r\n\x1a\n", "could not be read"),
        (png(W, H, 0, grey_rows(), depth=16), "16 bits"),
        (png(W, H, 0, grey_rows(), interlace=1), "interlaced"),
    ],
)
def test_unreadable_masks_are_refused_with_the_reason(data, fragment):
    with pytest.raises(KeyCallError) as caught:
        read(data)
    assert caught.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert fragment in caught.value.message


def test_short_pixel_data_is_refused():
    # A header claiming more rows than the pixel data holds.
    rows = grey_rows()[: H // 2]
    short = bytearray(png(W, H // 2, 0, rows))
    short[16:24] = struct.pack(">II", W, H)
    # The header's checksum no longer matches; the reader doesn't check it,
    # so the mismatch in row count is what's left to catch.
    with pytest.raises(KeyCallError, match="malformed"):
        read(bytes(short))


def test_a_header_claiming_too_many_pixels_is_refused_before_inflating():
    header_only = png(1, 1, 0, [b"\x00"])
    huge = bytearray(header_only)
    huge[16:24] = struct.pack(">II", 100_000, 100_000)
    with pytest.raises(KeyCallError, match="megapixels"):
        read(bytes(huge))


@pytest.mark.parametrize(("on", "off", "marked"), [(255, 255, "the whole picture"), (0, 0, "nothing")])
def test_a_mask_with_one_colour_is_refused(on, off, marked):
    for convert in (mask_to_black_edit, mask_to_alpha_edit):
        with pytest.raises(KeyCallError) as caught:
            convert(png(W, H, 0, grey_rows(on=on, off=off)), provider="t", operation="o")
        assert marked in caught.value.message
        assert "send no mask" in caught.value.message
