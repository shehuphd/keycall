"""Conservative model classification.

Precedence: explicit provider metadata first (the Gemini adapter passes
supported generation methods through), then maintained identifier rules.
Conflicts and unknowns resolve to UNKNOWN — an unknown model must never
silently enter the default text picker.
"""

from __future__ import annotations

from ._enums import ModelCategory

__all__ = ["classify_model_id"]

# Ordered: first match wins. More specific non-text signals come before the
# broad text-family patterns so "gpt-image-1" never classifies as text.
_RULES: tuple[tuple[tuple[str, ...], ModelCategory], ...] = (
    (("embed",), ModelCategory.EMBEDDING),
    (("whisper", "transcribe"), ModelCategory.TRANSCRIPTION),
    (("tts", "speech"), ModelCategory.SPEECH_GENERATION),
    (("realtime",), ModelCategory.REALTIME),
    (("dall-e", "image", "imagen", "flux"), ModelCategory.IMAGE_GENERATION),
    (("sora", "veo"), ModelCategory.VIDEO_GENERATION),
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
