"""OpenAI adapter: GET /models, POST /responses (Responses API)."""

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


class OpenAIAdapter(ProviderAdapter):
    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"])

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
                    classification_source="keycall_rule",
                )
            )
        return models, None  # GET /models is unpaginated

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        input_items = []
        for message in request.messages:
            # Responses API: assistant history uses output_text parts.
            part_type = "output_text" if message.role == "assistant" else "input_text"
            input_items.append(
                {
                    "role": message.role,
                    "content": [{"type": part_type, "text": part.text} for part in message.content],
                }
            )
        body: dict[str, Any] = {"model": request.model, "input": input_items}
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
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
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", ""))
            if item_type == "message":
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        parts.append(TextOutput(text=str(content.get("text", ""))))
                    elif isinstance(content, dict):
                        parts.append(UnknownOutput(provider_kind=str(content.get("type", "?"))))
            elif item_type == "reasoning":
                continue  # reasoning traces are not output content
            elif item_type:
                parts.append(UnknownOutput(provider_kind=item_type))

        usage_raw = payload.get("usage")
        if isinstance(usage_raw, dict):
            input_details = usage_raw.get("input_tokens_details") or {}
            output_details = usage_raw.get("output_tokens_details") or {}
            usage = Usage(
                input_tokens=usage_raw.get("input_tokens"),
                output_tokens=usage_raw.get("output_tokens"),
                cached_input_tokens=input_details.get("cached_tokens"),
                reasoning_tokens=output_details.get("reasoning_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            )
        else:
            usage = Usage()
            warnings.append("provider reported no usage information")

        finish = payload.get("status")
        incomplete = payload.get("incomplete_details")
        if finish == "incomplete" and isinstance(incomplete, dict) and incomplete.get("reason"):
            finish = f"incomplete:{incomplete['reason']}"

        return InvocationResult(
            provider=self.resolved.provider,
            model=str(payload.get("model", model)),
            operation=Operation.TEXT_GENERATION,
            parts=tuple(parts),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=headers.get("x-request-id"),
            finish_reason=str(finish) if finish else None,
            warnings=tuple(warnings),
        )
