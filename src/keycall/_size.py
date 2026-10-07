"""Size arithmetic shared by the picture adapters.

A KeyCall size is an aspect ratio ("16:9") or a pixel size ("1280x768").
Providers that take only pixels get a ratio converted here, by a rule the
catalog records per model family: the sides are multiples of a step, the
long side is a fixed length (the provider's own default long side),
and the ratio must sit within the family's bounds.
"""

from __future__ import annotations

__all__ = ["is_ratio", "parse_pixels", "parse_ratio", "pixels_for_ratio"]


def is_ratio(size: str) -> bool:
    return ":" in size


def parse_ratio(size: str) -> float:
    width, height = size.split(":")
    return float(width) / float(height)


def parse_pixels(size: str) -> tuple[int, int]:
    width, height = size.split("x")
    return int(width), int(height)


def pixels_for_ratio(ratio: float, *, long_side: int, square_side: int, step: int) -> tuple[int, int]:
    """The pixel size for a ratio: the long side fixed, the short side the
    nearest multiple of ``step``. A square ratio takes ``square_side``."""
    if abs(ratio - 1.0) < 1e-9:
        return square_side, square_side
    short = max(step, round(long_side / max(ratio, 1 / ratio) / step) * step)
    return (long_side, short) if ratio > 1 else (short, long_side)
