"""Conservative model classification.

Precedence: explicit provider metadata first (the Gemini adapter passes
supported generation methods through), then maintained identifier rules.
Conflicts and unknowns resolve to UNKNOWN — an unknown model must never
silently enter the default text picker.
"""

from __future__ import annotations

from ._enums import ModelCategory
from ._registry import resolve_provider
from ._types import AliasFact

__all__ = ["alias_fact", "classify_model_id"]


def alias_fact(provider: str, model_id: str) -> AliasFact | None:
    """Whether ``model_id`` is a rolling alias under ``provider``'s
    recorded naming convention.

    Keyless: reads only the bundled catalog's convention evidence, so a
    build pipeline can classify ids without holding any credential.
    Returns an :class:`AliasFact` only when the id matches a recorded
    convention; ``None`` otherwise, covering both a dated/pinned id and a
    provider with no recorded convention — absent, never a guess. An
    unknown provider name raises ``UNSUPPORTED_PROVIDER``, the same as
    client construction would.
    """
    resolved = resolve_provider(provider)
    lowered = model_id.lower()
    for convention in resolved.alias_conventions:
        if lowered.endswith(convention.suffix.lower()):
            return AliasFact(
                provider=resolved.provider,
                model_id=model_id,
                convention=f"{convention.suffix} suffix",
                maintained=convention.maintained,
                verified=convention.verified,
                note=convention.note,
            )
    return None

# Ordered: first match wins. More specific non-text signals come before the
# broad text-family patterns so "gpt-image-1" never classifies as text.
_RULES: tuple[tuple[tuple[str, ...], ModelCategory], ...] = (
    (("embed",), ModelCategory.EMBEDDING),
    (("whisper", "transcribe"), ModelCategory.TRANSCRIPTION),
    (("tts", "speech"), ModelCategory.SPEECH_GENERATION),
    # xAI's realtime model is "grok-voice-latest", with no "realtime" substring
    # and no provider metadata to fall back on (its /v1/models lists bare
    # ids), so the identifier is the only signal available for it.
    (("realtime", "voice"), ModelCategory.REALTIME),
    (("dall-e", "image", "imagen", "flux"), ModelCategory.IMAGE_GENERATION),
    (("sora", "veo", "imagine-video"), ModelCategory.VIDEO_GENERATION),
    # Ambiguous or out-of-taxonomy families stay unknown rather than
    # guessing: moderation/reranking/guard models, audio-hybrid previews.
    (("moderation", "rerank", "guard", "audio"), ModelCategory.UNKNOWN),
)

_TEXT_MARKERS: tuple[str, ...] = (
    "gpt",
    "chatgpt",
    "claude",
    "gemini",
    "gemma",
    "deepseek",
    "sonar",
    "moonshot",
    "kimi",
    "mistral",
    "llama",
    "qwen",
    "glm",
    "grok",
)


def classify_model_id(model_id: str) -> ModelCategory:
    """Classify by identifier rules alone. Adapters with provider metadata
    should prefer that evidence and only fall back here."""
    lowered = model_id.lower()
    for markers, category in _RULES:
        if any(marker in lowered for marker in markers):
            return category
    if any(marker in lowered for marker in _TEXT_MARKERS):
        return ModelCategory.TEXT_GENERATION
    # OpenAI reasoning families: o1, o3, o4-mini, ...
    if len(lowered) >= 2 and lowered[0] == "o" and lowered[1].isdigit():
        return ModelCategory.TEXT_GENERATION
    return ModelCategory.UNKNOWN
