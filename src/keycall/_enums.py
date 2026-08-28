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
    # Speech-to-text WebSocket providers (AssemblyAI, Deepgram). Each named
    # provider has its own frame dialect and adapter; there is no generic
    # "stt-compatible" wire the way there is an OpenAI-compatible one, so
    # custom targets cannot claim this protocol.
    STT = "stt"


class Operation(str, Enum):
    """What KeyCall was asked to do. Members ship with the adapter code
    that implements them, never ahead of it."""

    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    SPEECH_GENERATION = "speech_generation"
    VIDEO_GENERATION = "video_generation"
    STREAMING_TRANSCRIPTION = "streaming_transcription"
