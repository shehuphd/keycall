"""Normalized error codes and the public exception types.

One exception class with a typed ``ErrorCode`` discriminator, plus one
subclass, ``VideoJobTimeout``, which exists because it carries something a
flat error can't: the still-valid job handle, so a caller whose waiting
budget ran out never loses a render they already paid to start.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import VideoJob

__all__ = ["ErrorCode", "KeyCallError", "VideoJobTimeout"]


class ErrorCode(str, Enum):
    # Setup / configuration failures — raised before any provider call.
    UNSUPPORTED_PROVIDER = "unsupported_provider"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    CATALOG_UPDATE_REQUIRED = "catalog_update_required"

    # Provider-call failures.
    INVALID_API_KEY = "invalid_api_key"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    MODEL_NOT_AVAILABLE = "model_not_available"
    MODEL_NOT_SUITABLE = "model_not_suitable"


class KeyCallError(Exception):
    """Any KeyCall failure. Inspect ``code`` to branch on the cause.

    Messages must already be sanitized before construction: no credentials,
    no authorization headers, no raw request bodies, no unsanitized provider
    error text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        provider: str | None = None,
        operation: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
        provider_request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.provider = provider
        self.operation = operation
        self.retryable = retryable
        self.status_code = status_code
        self.provider_request_id = provider_request_id
        self.retry_after = retry_after

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code.value!r}, message={self.message!r}, "
            f"provider={self.provider!r}, retryable={self.retryable!r})"
        )


class VideoJobTimeout(KeyCallError):
    """generate_video's waiting budget ran out while the render was still
    going. The job is not dead: ``job`` is the still-valid handle, and
    ``check_video(error.job)`` picks up polling where the wait left off.
    Raised with ``code=ErrorCode.TIMEOUT``."""

    def __init__(self, message: str, *, provider: str, job: VideoJob) -> None:
        super().__init__(
            message,
            code=ErrorCode.TIMEOUT,
            provider=provider,
            operation="video_generation",
            retryable=True,
        )
        self.job = job
