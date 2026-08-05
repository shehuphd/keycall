"""Adapter interface: pure request-building and response-parsing.

Adapters never perform I/O and never see the credential. The client drives
the page loop and hands specs to the transport layer, which is the single
place credentials are revealed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .._enums import Operation
from .._errors import ErrorCode
from .._registry import ResolvedProvider
from .._transport import RequestSpec
from .._types import InvocationResult, Model, TextGenerationRequest


class ProviderAdapter(ABC):
    """One instance per resolved provider profile. Stateless and pure."""

    def __init__(self, resolved: ResolvedProvider) -> None:
        self.resolved = resolved

    # --- model listing (client drives the page loop) ---

    @abstractmethod
    def initial_list_request(self) -> RequestSpec: ...

    @abstractmethod
    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        """Return this page's normalized models and the next page's spec,
        or None when there are no more pages. Must tolerate unknown fields
        and classify conservatively."""

    # --- text generation ---

    @abstractmethod
    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec: ...

    @abstractmethod
    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        """Decode into the common envelope. Raw provider objects must not
        leak through (PRD section 7)."""

    # --- error translation (transport calls this, then scrubs) ---

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        """Map a provider error response to (code, retryable, message).
        The returned message is scrubbed by the transport before use.
        Default covers the common OpenAI-shaped error body."""
        message = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", ""))
            elif isinstance(error, str):
                message = error
        if status_code == 401:
            return ErrorCode.INVALID_API_KEY, False, message or "invalid API key"
        if status_code == 403:
            return ErrorCode.PERMISSION_DENIED, False, message or "permission denied"
        if status_code == 404:
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "not found"
        if status_code == 429:
            return ErrorCode.RATE_LIMITED, True, message or "rate limited"
        if status_code >= 500:
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider server error"
        return (
            ErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
            message or f"unexpected status {status_code}",
        )

    # --- shared helpers ---

    def validate_generation_request(self, request: TextGenerationRequest) -> None:
        """Pre-flight checks that mirror what the provider would reject:
        text-only parts (v1), and sampling params against models with
        maintained evidence that they reject them."""
        from .._capabilities import rejects_sampling_params
        from .._errors import KeyCallError
        from .._types import TextInput

        for message in request.messages:
            for part in message.content:
                if not isinstance(part, TextInput):
                    raise KeyCallError(
                        f"v1 text generation supports text input parts only, "
                        f"got {type(part).__name__}",
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        operation=Operation.TEXT_GENERATION.value,
                    )
        if (request.temperature is not None or request.top_p is not None) and (
            rejects_sampling_params(request.model)
        ):
            raise KeyCallError(
                f"model {request.model!r} rejects temperature/top_p; remove the "
                "sampling parameters for this model",
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )

    @staticmethod
    def sampling_fields(request: TextGenerationRequest) -> dict[str, float]:
        """temperature/top_p body fields, omitted when unset (the OpenAI-shaped
        field names, which Anthropic shares)."""
        fields: dict[str, float] = {}
        if request.temperature is not None:
            fields["temperature"] = request.temperature
        if request.top_p is not None:
            fields["top_p"] = request.top_p
        return fields
