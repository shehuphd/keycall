"""Google Gemini adapter: GET /models (pageToken pagination),
POST /models/{model}:generateContent.

Gemini's list response is the strongest provider-native classification
source in the v1 set: supportedGenerationMethods is explicit provider
metadata and takes precedence over identifier rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .._classify import classify_model_id
from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    Citation,
    InvocationResult,
    Model,
    OutputPart,
    TextGenerationRequest,
    TextOutput,
    UnknownOutput,
    Usage,
)
from ._base import ProviderAdapter

_PAGE_SIZE = "1000"

# Modalities specific enough that an identifier match outranks the generic
# "supports generateContent" signal.
_DISTINCTIVE_MODALITIES = frozenset(
    {
        ModelCategory.SPEECH_GENERATION,
        ModelCategory.TRANSCRIPTION,
        ModelCategory.IMAGE_GENERATION,
        ModelCategory.VIDEO_GENERATION,
        ModelCategory.EMBEDDING,
    }
)

_METHOD_CATEGORIES = {
    "generateContent": ModelCategory.TEXT_GENERATION,
    "embedContent": ModelCategory.EMBEDDING,
    "embedText": ModelCategory.EMBEDDING,
    "predict": ModelCategory.IMAGE_GENERATION,
    "predictLongRunning": ModelCategory.VIDEO_GENERATION,
}


def _strip_prefix(name: str) -> str:
    return name.removeprefix("models/")


class GeminiAdapter(ProviderAdapter):
    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"], params={"pageSize": _PAGE_SIZE})

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise KeyCallError(
                "model list response missing 'models' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = []
        for entry in payload["models"]:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            model_id = _strip_prefix(str(entry["name"]))
            methods = entry.get("supportedGenerationMethods")
            categories: set[ModelCategory] = set()
            source = "keycall_rule"
            if isinstance(methods, list):
                for method in methods:
                    category = _METHOD_CATEGORIES.get(str(method))
                    if category is not None:
                        categories.add(category)
                if categories:
                    source = "provider_metadata"
            # generateContent is the transport, not a modality claim: the TTS
            # and image variants use it to return audio/images and reject a
            # TEXT response. When the identifier names a distinctive
            # non-text modality, it wins over the generic mapping.
            rule_category = classify_model_id(model_id)
            if rule_category in _DISTINCTIVE_MODALITIES and (
                ModelCategory.TEXT_GENERATION in categories
            ):
                categories.discard(ModelCategory.TEXT_GENERATION)
                categories.add(rule_category)
                source = "provider_metadata+keycall_rule"
            if not categories:
                categories = {rule_category}
            models.append(
                Model(
                    id=model_id,
                    provider=self.resolved.provider,
                    categories=frozenset(categories),
                    display_name=entry.get("displayName"),
                    context_limit=entry.get("inputTokenLimit"),
                    classification_source=source,
                )
            )
        next_spec = None
        token = payload.get("nextPageToken")
        if token:
            op = self.resolved.operations["list_models"]
            next_spec = RequestSpec(
                method=op["method"],
                path=op["path"],
                params={"pageSize": _PAGE_SIZE, "pageToken": str(token)},
            )
        return models, next_spec

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            texts = [part.text for part in message.content]
            if message.role == "system":
                system_texts.extend(texts)
            else:
                contents.append(
                    {
                        # Gemini's assistant role is "model".
                        "role": "model" if message.role == "assistant" else "user",
                        "parts": [{"text": text} for text in texts],
                    }
                )
        if not contents:
            raise KeyCallError(
                "gemini requires at least one non-system message",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        body: dict[str, Any] = {"contents": contents}
        if system_texts:
            body["systemInstruction"] = {"parts": [{"text": text} for text in system_texts]}
        generation_config: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.response_schema is not None:
            # Live-verified 2026-08-06: standard lowercase JSON Schema type
            # names ("object", "string", ...) work directly, no dialect
            # conversion to Gemini's uppercase OpenAPI-subset spelling needed.
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = dict(request.response_schema)
        if generation_config:
            body["generationConfig"] = generation_config
        if request.web_search:
            body["tools"] = [{"google_search": {}}]
        path = op["path"].format(model=quote(_strip_prefix(request.model), safe=""))
        return RequestSpec(method=op["method"], path=path, json_body=body)

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
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                finish_reason = candidate.get("finishReason")
                content = candidate.get("content")
                if isinstance(content, dict):
                    for part in content.get("parts", []):
                        if isinstance(part, dict) and "text" in part:
                            parts.append(TextOutput(text=str(part["text"])))
                        elif isinstance(part, dict):
                            kind = next(iter(part.keys()), "?")
                            parts.append(UnknownOutput(provider_kind=str(kind)))
                grounding = candidate.get("groundingMetadata")
                if isinstance(grounding, dict):
                    # web.uri is a vertexaisearch.cloud.google.com redirect,
                    # by Google's design; title names the real source.
                    for chunk in grounding.get("groundingChunks") or []:
                        web = chunk.get("web") if isinstance(chunk, dict) else None
                        if isinstance(web, dict) and web.get("uri"):
                            citations.append(
                                Citation(url=str(web["uri"]), title=web.get("title"))
                            )

        usage_raw = payload.get("usageMetadata")
        if isinstance(usage_raw, dict):
            usage = Usage(
                input_tokens=usage_raw.get("promptTokenCount"),
                output_tokens=usage_raw.get("candidatesTokenCount"),
                cached_input_tokens=usage_raw.get("cachedContentTokenCount"),
                reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                total_tokens=usage_raw.get("totalTokenCount"),
            )
        else:
            usage = Usage()
            warnings.append("provider reported no usage information")

        return InvocationResult(
            provider=self.resolved.provider,
            model=_strip_prefix(str(payload.get("modelVersion", model))),
            operation=Operation.TEXT_GENERATION,
            parts=tuple(parts),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(payload.get("responseId")),
            finish_reason=str(finish_reason) if finish_reason else None,
            citations=tuple(citations),
            warnings=tuple(warnings),
        )

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        message = ""
        status_name = ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", ""))
            status_name = str(payload["error"].get("status", ""))
        # Gemini reports an invalid key as 400 INVALID_ARGUMENT with an
        # API_KEY_INVALID detail, not as a 401.
        if status_code in (401, 403) or "API key" in message or "API_KEY" in message:
            code = (
                ErrorCode.PERMISSION_DENIED
                if status_name == "PERMISSION_DENIED"
                else ErrorCode.INVALID_API_KEY
            )
            return code, False, message or "invalid API key"
        if status_code == 429 or status_name == "RESOURCE_EXHAUSTED":
            return ErrorCode.RATE_LIMITED, True, message or "rate or quota limited"
        if status_name == "NOT_FOUND" or status_code == 404:
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "model not found"
        if status_name in ("UNAVAILABLE", "DEADLINE_EXCEEDED") or status_code >= 500:
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider unavailable"
        return (
            ErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
            message or f"unexpected status {status_code}",
        )
