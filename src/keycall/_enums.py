"""Public closed enums: model categories, wire protocols, operations."""

from __future__ import annotations

from enum import Enum

__all__ = ["ModelCategory", "Operation", "ProviderProtocol"]


class ModelCategory(str, Enum):
    """What a model is suited for. Grows as the classifier learns new kinds."""

    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    EMBEDDING = "embedding"
    TRANSCRIPTION = "transcription"
    SPEECH_GENERATION = "speech_generation"
    VIDEO_GENERATION = "video_generation"
    REALTIME = "realtime"
    UNKNOWN = "unknown"


class ProviderProtocol(str, Enum):
    """Wire protocol an adapter speaks. Distinct from provider identity."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai-compatible"


class Operation(str, Enum):
    """What KeyCall was asked to do. Seeded with v1's only operation;
    members ship with the adapter code that implements them."""

    TEXT_GENERATION = "text_generation"
