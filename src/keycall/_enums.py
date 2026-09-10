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
    # ElevenLabs speaks its own wire on every operation (xi-api-key REST
    # for models/voices/TTS, a JSON-message WebSocket for realtime STT);
    # single-vendor, so custom targets cannot claim this protocol either.
    ELEVENLABS = "elevenlabs"
    # Service providers (kind "service" in the catalog): no models, no
    # generation; validated by a live probe per service category. Each is
    # single-vendor with its own wire, so custom targets cannot claim
    # these protocols either.
    GOOGLE_MAPS = "google_maps"
    LIVEKIT = "livekit"


class Operation(str, Enum):
    """What KeyCall was asked to do. Members ship with the adapter code
    that implements them, never ahead of it."""

    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    BATCH_GENERATION = "batch_generation"
    BATCH_EMBEDDING = "batch_embedding"
    IMAGE_GENERATION = "image_generation"
    SPEECH_GENERATION = "speech_generation"
    VIDEO_GENERATION = "video_generation"
    TRANSCRIPTION = "transcription"
    STREAMING_TRANSCRIPTION = "streaming_transcription"
    SERVICE_PROBE = "service_probe"
