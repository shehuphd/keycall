"""KeyCall: one consistent interface for validating AI-provider API keys,
listing and filtering their models, and making normalized calls."""

from ._client import AsyncKeyCall, KeyCall
from ._enums import ModelCategory, Operation, ProviderProtocol
from ._errors import ErrorCode, KeyCallError
from ._types import (
    AudioInput,
    AudioOutput,
    EmbeddingOutput,
    FileInput,
    FileOutput,
    ImageInput,
    ImageOutput,
    InputPart,
    InvocationResult,
    Message,
    MessageRole,
    Model,
    ModelDiscovery,
    OutputPart,
    TextGenerationRequest,
    TextInput,
    TextOutput,
    TranscriptOutput,
    UnknownOutput,
    Usage,
    VideoOutput,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncKeyCall",
    "AudioInput",
    "AudioOutput",
    "EmbeddingOutput",
    "ErrorCode",
    "FileInput",
    "FileOutput",
    "ImageInput",
    "ImageOutput",
    "InputPart",
    "InvocationResult",
    "KeyCall",
    "KeyCallError",
    "Message",
    "MessageRole",
    "Model",
    "ModelCategory",
    "ModelDiscovery",
    "Operation",
    "OutputPart",
    "ProviderProtocol",
    "TextGenerationRequest",
    "TextInput",
    "TextOutput",
    "TranscriptOutput",
    "UnknownOutput",
    "Usage",
    "VideoOutput",
    "__version__",
]
