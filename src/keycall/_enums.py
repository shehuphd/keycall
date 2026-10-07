"""Public closed enums: model categories, wire protocols, operations."""

from __future__ import annotations

from enum import Enum

__all__ = ["ModelCategory", "Operation", "ProviderProtocol"]


class ModelCategory(str, Enum):
    """What a model is suited for. Grows as the classifier learns new kinds."""

    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    # Works on a picture it is given (edit, upscale, expand, background
    # and object removal, description). A model can carry this beside
    # IMAGE_GENERATION; an upscaler carries it alone, so it never enters a
    # picker of models that draw from a prompt.
    IMAGE_EDITING = "image_editing"
    EMBEDDING = "embedding"
    TRANSCRIPTION = "transcription"
    SPEECH_GENERATION = "speech_generation"
    VIDEO_GENERATION = "video_generation"
    REALTIME = "realtime"
    # Full-duplex voice on a dedicated live-sessions endpoint that delegates
    # reasoning to a separate backend model (OpenAI's gpt-live on
    # /v1/live/sessions), distinct from the Realtime API that REALTIME covers.
    LIVE = "live"
    # Typed judgments with calibrated probabilities (TypeSafe's jev family):
    # the model answers fixed-shape questions about a state and never
    # generates text, so it must not enter the default text picker.
    DECISION = "decision"
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
    # TypeSafe's System One judgment wire (POST /v1/systemone): typed
    # questions in, typed answers with probabilities out. Single-vendor,
    # so custom targets cannot claim this protocol either.
    TYPESAFE = "typesafe"
    # Ideogram's image and video wire: the model is a path segment
    # (POST /v2/{content}/{action}/{model}), results come back as
    # expiring links, and some endpoints answer with a job to poll.
    # Single-vendor, so custom targets cannot claim this protocol either.
    IDEOGRAM = "ideogram"


class Operation(str, Enum):
    """What KeyCall was asked to do. Members ship with the adapter code
    that implements them, never ahead of it."""

    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    BATCH_GENERATION = "batch_generation"
    BATCH_EMBEDDING = "batch_embedding"
    IMAGE_GENERATION = "image_generation"
    # Picture-in, picture-out operations. Each is its own member so a
    # provider's support is declared per operation, not as one flag.
    IMAGE_EDIT = "image_edit"
    IMAGE_UPSCALE = "image_upscale"
    IMAGE_EXPAND = "image_expand"
    BACKGROUND_REMOVAL = "background_removal"
    BACKGROUND_REPLACEMENT = "background_replacement"
    OBJECT_ERASE = "object_erase"
    IMAGE_LAYERIZE = "image_layerize"
    # Picture in, text out.
    IMAGE_DESCRIPTION = "image_description"
    # A provider's own named tool, run as a job (Ideogram's ad tools).
    PROVIDER_TOOL = "provider_tool"
    SPEECH_GENERATION = "speech_generation"
    VIDEO_GENERATION = "video_generation"
    TRANSCRIPTION = "transcription"
    STREAMING_TRANSCRIPTION = "streaming_transcription"
    DICTATION = "dictation"
    JUDGMENT = "judgment"
    SERVICE_PROBE = "service_probe"
