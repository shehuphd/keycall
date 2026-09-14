"""Live-suite spend meter.

A ``-m live`` run records the usage it already gets back (tokens, billed
voice seconds, transcribed audio seconds, images, video seconds) into a
process-local ledger, and ``conftest.py`` prints a per-operation tally plus
an approximate USD total at the end of the run.

The token/second/count totals are exact (they come straight off the provider
responses). The USD figures are an ESTIMATE on top of them: ``PRICING`` is
hand-maintained, coarse (per-provider, not per-model), and dated, since exact
prices move and vary by tier. Read the usage totals as the measured line and
the USD as a budgeting approximation; refine ``PRICING`` against each
provider's own pricing page when a number needs to be exact.

Only the material cost drivers are wired to ``record()`` (the per-provider
smoke generate, gpt-live voice, image, video, and transcription); the long
tail of small text tests is fractions of a cent each and is left out rather
than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- pricing (approximate, dated 2026-09-14) --------------------------------
# Text is USD per 1M tokens, a single blended rate per provider (the smoke
# walk uses whichever model verify picks, so a per-model table would be false
# precision here). Media rates are per unit as noted.

TEXT_PER_1M: dict[str, float] = {
    "openai": 5.0,
    "anthropic": 9.0,
    "gemini": 2.0,
    "deepseek": 1.0,
    "perplexity": 3.0,
    "moonshot": 2.0,
    "xai": 5.0,
}
TEXT_DEFAULT_PER_1M = 4.0

VOICE_PER_SEC = 0.03          # gpt-live / realtime billed voice second
TRANSCRIBE_PER_SEC = 0.0001   # streaming/prerecorded STT per audio second
IMAGE_PER_IMAGE = 0.06        # one generated image, lightest tier
VIDEO_PER_SEC = 0.20          # generated video second, lightest tier


@dataclass
class _Entry:
    provider: str
    operation: str
    tokens: int = 0
    voice_seconds: float = 0.0
    audio_seconds: float = 0.0
    images: int = 0
    video_seconds: float = 0.0


LEDGER: list[_Entry] = []


def reset() -> None:
    LEDGER.clear()


def record(
    provider: str,
    operation: str,
    *,
    tokens: int = 0,
    voice_seconds: float = 0.0,
    audio_seconds: float = 0.0,
    images: int = 0,
    video_seconds: float = 0.0,
) -> None:
    """Add one billable observation to the run ledger. Every field is a
    measured quantity from a provider response; absent quantities stay zero."""
    LEDGER.append(
        _Entry(
            provider=provider,
            operation=operation,
            tokens=int(tokens or 0),
            voice_seconds=float(voice_seconds or 0.0),
            audio_seconds=float(audio_seconds or 0.0),
            images=int(images or 0),
            video_seconds=float(video_seconds or 0.0),
        )
    )


def _entry_usd(e: _Entry) -> float:
    rate = TEXT_PER_1M.get(e.provider, TEXT_DEFAULT_PER_1M)
    return (
        e.tokens / 1_000_000 * rate
        + e.voice_seconds * VOICE_PER_SEC
        + e.audio_seconds * TRANSCRIBE_PER_SEC
        + e.images * IMAGE_PER_IMAGE
        + e.video_seconds * VIDEO_PER_SEC
    )


def summary_lines() -> list[str]:
    """A human-readable per-operation tally with exact usage and an
    approximate USD total. Empty when nothing was recorded."""
    if not LEDGER:
        return []
    by_op: dict[str, _Entry] = {}
    for e in LEDGER:
        agg = by_op.setdefault(e.operation, _Entry(provider="*", operation=e.operation))
        agg.tokens += e.tokens
        agg.voice_seconds += e.voice_seconds
        agg.audio_seconds += e.audio_seconds
        agg.images += e.images
        agg.video_seconds += e.video_seconds
    lines = ["live spend (usage exact; USD approximate, pricing dated 2026-09-14):"]
    total = 0.0
    for op, agg in sorted(by_op.items()):
        usd = sum(_entry_usd(e) for e in LEDGER if e.operation == op)
        total += usd
        bits: list[str] = []
        if agg.tokens:
            bits.append(f"{agg.tokens} tok")
        if agg.voice_seconds:
            bits.append(f"{agg.voice_seconds:.1f} voice-s")
        if agg.audio_seconds:
            bits.append(f"{agg.audio_seconds:.1f} audio-s")
        if agg.images:
            bits.append(f"{agg.images} img")
        if agg.video_seconds:
            bits.append(f"{agg.video_seconds:.1f} video-s")
        lines.append(f"  {op:28s} {', '.join(bits):32s} ~${usd:.3f}")
    lines.append(f"  {'TOTAL':28s} {'':32s} ~${total:.3f}")
    return lines
