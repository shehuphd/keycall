"""Anthropic adapter: GET /v1/models (cursor pagination), POST /v1/messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._classify import classify_model_id
from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import (
    InvocationResult,
    Model,
    OutputPart,
    TextGenerationRequest,
    TextOutput,
    UnknownOutput,
    Usage,
)
from ._base import ProviderAdapter

# Anthropic requires max_tokens on every messages call; used when the
# caller didn't specify one.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096
_PAGE_LIMIT = "1000"


class AnthropicAdapter(ProviderAdapter):
    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"], params={"limit": _PAGE_LIMIT})

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise KeyCallError(
                "model list response missing 'data' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = []
        for entry in payload["data"]:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            model_id = str(entry["id"])
            models.append(
                Model(
                    id=model_id,
                    provider=self.resolved.provider,
                    categories=frozenset({classify_model_id(model_id)}),
                    display_name=entry.get("display_name"),
                    classification_source="keycall_rule",
                )
            )
        next_spec = None
        if payload.get("has_more") and payload.get("last_id"):
            op = self.resolved.operations["list_models"]
            next_spec = RequestSpec(
                method=op["method"],
                path=op["path"],
                params={"limit": _PAGE_LIMIT, "after_id": str(payload["last_id"])},
            )
        return models, next_spec

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        system_texts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            texts = [part.text for part in message.content]
            if message.role == "system":
                # Anthropic takes system content as a top-level parameter.
                system_texts.extend(texts)
            else:
                messages.append(
                    {
                        "role": message.role,
                        "content": [{"type": "text", "text": text} for text in texts],
                    }
                )
        if not messages:
            raise KeyCallError(
                "anthropic requires at least one non-system message",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if system_texts:
            body["system"] = "\n\n".join(system_texts)
        body.update(self.sampling_fields(request))
        return RequestSpec(method=op["method"], path=op["path"], json_body=body)

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        if not isinstance(payload, dict):
            raise KeyCallError(
                "generation response was not a JSON object",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        parts: list[OutputPart] = []
        warnings: list[str] = []
        for block in payload.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(TextOutput(text=str(block.get("text", ""))))
            elif isinstance(block, dict):
                parts.append(UnknownOutput(provider_kind=str(block.get("type", "?"))))

        usage_raw = payload.get("usage")
        if isinstance(usage_raw, dict):
            # Anthropic reports no total; None stays None — never fabricated.
            usage = Usage(
                input_tokens=usage_raw.get("input_tokens"),
                output_tokens=usage_raw.get("output_tokens"),
                cached_input_tokens=usage_raw.get("cache_read_input_tokens"),
            )
        else:
            usage = Usage()
            warnings.append("provider reported no usage information")

        return InvocationResult(
            provider=self.resolved.provider,
            model=str(payload.get("model", model)),
            operation=Operation.TEXT_GENERATION,
            parts=tuple(parts),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=headers.get("request-id"),
            finish_reason=payload.get("stop_reason"),
            warnings=tuple(warnings),
        )

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        message = ""
        error_type = ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", ""))
            error_type = str(payload["error"].get("type", ""))
        if error_type == "authentication_error" or status_code == 401:
            return ErrorCode.INVALID_API_KEY, False, message or "invalid API key"
        if error_type in ("permission_error", "billing_error") or status_code in (402, 403):
            return ErrorCode.PERMISSION_DENIED, False, message or "permission denied"
        if error_type == "rate_limit_error" or status_code == 429:
            return ErrorCode.RATE_LIMITED, True, message or "rate limited"
        if error_type == "overloaded_error":
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider overloaded"
        if status_code == 404:
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "not found"
        if status_code >= 500:
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider server error"
        return (
            ErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
            message or f"unexpected status {status_code}",
        )
