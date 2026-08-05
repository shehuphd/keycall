"""OpenAI-compatible adapter: GET /models, POST /chat/completions.

Serves DeepSeek, Perplexity, Moonshot, and explicit custom targets. Only
the conventional Chat Completions surface is assumed; provider-specific
extensions are not (registry research section 9).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._classify import classify_model_id
from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import (
    Citation,
    InvocationResult,
    Model,
    OutputPart,
    TextGenerationRequest,
    TextOutput,
    Usage,
)
from ._base import ProviderAdapter


class OpenAICompatibleAdapter(ProviderAdapter):
    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"])

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        # Tolerate both {"data": [...]} and a bare list — "compatible"
        # endpoints vary.
        entries: Any = None
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            entries = payload["data"]
        elif isinstance(payload, list):
            entries = payload
        if entries is None:
            raise KeyCallError(
                "model list response missing 'data' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = []
        for entry in entries:
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
        return models, None

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        messages = []
        for message in request.messages:
            text = "\n".join(part.text for part in message.content)
            messages.append({"role": message.role, "content": text})
        body: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
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
        citations: list[Citation] = []
        finish_reason = None
        by_url: dict[str, Citation] = {}
        for entry in payload.get("search_results") or []:
            if isinstance(entry, dict) and entry.get("url"):
                by_url[str(entry["url"])] = Citation(
                    url=str(entry["url"]),
                    title=entry.get("title"),
                    cited_text=entry.get("snippet"),
                )
        for url in payload.get("citations") or []:
            if isinstance(url, str) and url not in by_url:
                by_url[url] = Citation(url=url)
        citations.extend(by_url.values())
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content"):
                    parts.append(TextOutput(text=str(message["content"])))

        usage_raw = payload.get("usage")
        if isinstance(usage_raw, dict):
            details = usage_raw.get("prompt_tokens_details") or {}
            usage = Usage(
                input_tokens=usage_raw.get("prompt_tokens"),
                output_tokens=usage_raw.get("completion_tokens"),
                cached_input_tokens=details.get("cached_tokens")
                or usage_raw.get("prompt_cache_hit_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
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
            provider_request_id=None,  # not documented for compat targets
            finish_reason=str(finish_reason) if finish_reason else None,
            citations=tuple(citations),
            warnings=tuple(warnings),
        )
