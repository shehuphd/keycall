"""Anthropic adapter: GET /v1/models (cursor pagination), POST /v1/messages."""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from .._classify import classify_model_id
from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    Citation,
    CitationFound,
    FileInput,
    ImageInput,
    InvocationResult,
    Model,
    OutputPart,
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextGenerationRequest,
    TextInput,
    TextOutput,
    ToolCall,
    ToolResult,
    UnknownOutput,
    UnknownStreamEvent,
    Usage,
)
from ._base import (
    InbandStreamError,
    ProviderAdapter,
    StreamAssembler,
    context_limit,
    dedupe_citations,
    image_media_type,
    media_type_for,
    released_at,
)

# Anthropic requires max_tokens on every messages call; used when the
# caller didn't specify one.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096
_PAGE_LIMIT = "1000"

# Structured output has no native response-format API on Anthropic: it's
# implemented by forcing a single synthetic tool whose input_schema is the
# caller's response_schema, then reading that tool call's input back as
# the answer (live-verified 2026-08-06). This name only needs to be
# distinguishable from a real tool the caller might add once general tool
# calling exists — it's never sent to or interpreted by the model as
# anything but an arbitrary tool name.
_STRUCTURED_OUTPUT_TOOL_NAME = "keycall_response"


class _AnthropicStreamAssembler(StreamAssembler):
    """Event names and shapes live-verified 2026-08-08: message_start,
    content_block_start/delta/stop, message_delta (usage + stop_reason),
    message_stop terminal, ping keep-alives, in-band error events."""

    def __init__(self, resolved, request, adapter: AnthropicAdapter) -> None:
        super().__init__(resolved, request)
        self._adapter = adapter
        # index -> content block type ("text", "tool_use:<name>", ...)
        self._blocks: dict[int, str] = {}

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        payload = self._parse_data(data)
        if not isinstance(payload, dict):
            return []
        kind = event_name or str(payload.get("type", ""))
        if kind == "ping":
            return []
        if kind == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                if message.get("model"):
                    self.model = str(message["model"])
                usage = message.get("usage")
                if isinstance(usage, dict):
                    self.usage = Usage(
                        input_tokens=usage.get("input_tokens"),
                        cached_input_tokens=usage.get("cache_read_input_tokens"),
                    )
            return [StreamStart(model=self.model)]
        if kind == "content_block_start":
            index = int(payload.get("index", 0))
            block = payload.get("content_block")
            block_type = str(block.get("type", "?")) if isinstance(block, dict) else "?"
            if block_type == "tool_use" and isinstance(block, dict):
                name = str(block.get("name", ""))
                block_type = f"tool_use:{name}"
                self._blocks[index] = block_type
                if name != _STRUCTURED_OUTPUT_TOOL_NAME:
                    return [
                        self.begin_tool_call(
                            index, call_id=str(block.get("id", "")), name=name
                        )
                    ]
                return []
            self._blocks[index] = block_type
            return []
        if kind == "content_block_delta":
            index = int(payload.get("index", 0))
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                return []
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = str(delta.get("text", ""))
                self.append_text(text)
                return [TextDelta(text=text)]
            if delta_type == "input_json_delta":
                fragment = str(delta.get("partial_json", ""))
                if self._blocks.get(index) == f"tool_use:{_STRUCTURED_OUTPUT_TOOL_NAME}":
                    # The forced structured-output tool: its input is the
                    # answer, streamed as JSON fragments, matching the
                    # non-streaming contract that result.text carries the
                    # JSON string.
                    self.append_text(fragment)
                    return [TextDelta(text=fragment)]
                return self.append_tool_arguments(index, fragment)
            if delta_type == "citations_delta":
                note = delta.get("citation")
                if isinstance(note, dict) and note.get("url"):
                    citation = Citation(
                        url=str(note["url"]),
                        title=note.get("title"),
                        cited_text=note.get("cited_text"),
                    )
                    self.citations.append(citation)
                    return [CitationFound(citation=citation)]
                return []
            return []
        if kind == "content_block_stop":
            return self.complete_tool_call(int(payload.get("index", 0)))
        if kind == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason"):
                self.finish_reason = str(delta["stop_reason"])
            usage = payload.get("usage")
            if isinstance(usage, dict) and usage.get("output_tokens") is not None:
                self.usage = dataclasses.replace(
                    self.usage, output_tokens=usage.get("output_tokens")
                )
                self.usage_reported = True
            return []
        if kind == "message_stop":
            self.saw_terminal = True
            events = self.flush_tool_calls()
            events.append(StreamFinish(finish_reason=self.finish_reason, usage=self.usage))
            return events
        if kind == "error":
            code, retryable, message = self._adapter.translate_error(500, payload)
            raise InbandStreamError(code, retryable, message)
        return [UnknownStreamEvent(provider_kind=kind or "?")]


class AnthropicAdapter(ProviderAdapter):
    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        spec = self.build_generation_spec(request)
        return RequestSpec(
            method=spec.method,
            path=spec.path,
            params=spec.params,
            json_body={**(spec.json_body or {}), "stream": True},
        )

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        return _AnthropicStreamAssembler(self.resolved, request, self)

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
                    released_at=released_at(entry),
                    context_limit=context_limit(entry),
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
            if message.role == "system":
                # Anthropic takes system content as a top-level parameter.
                system_texts.extend(
                    part.text for part in message.content if isinstance(part, TextInput)
                )
                continue
            blocks: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextInput):
                    blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageInput):
                    # Both source forms verified 2026-08-09.
                    source = (
                        {"type": "url", "url": part.url}
                        if part.url is not None
                        else {
                            "type": "base64",
                            "media_type": image_media_type(part, provider="anthropic"),
                            "data": base64.b64encode(part.data or b"").decode(),
                        }
                    )
                    blocks.append({"type": "image", "source": source})
                elif isinstance(part, FileInput):
                    # Anthropic calls a document its own block type rather
                    # than a variant of image (verified 2026-08-09).
                    blocks.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type_for(
                                    part, kind="file", provider="anthropic"
                                ),
                                "data": base64.b64encode(part.data or b"").decode(),
                            },
                        }
                    )
                elif isinstance(part, ToolCall):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.id,
                            "name": part.name,
                            "input": dict(part.arguments),
                        }
                    )
                elif isinstance(part, ToolResult):
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": self.tool_result_text(part.content),
                        }
                    )
            messages.append({"role": message.role, "content": blocks})
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
        tools: list[dict[str, Any]] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ]
        if request.web_search:
            tools.append({"type": "web_search_20250305", "name": "web_search"})
        if tools:
            body["tools"] = tools
        if request.tool_choice is not None:
            # Anthropic's spellings, live-verified 2026-08-08.
            body["tool_choice"] = (
                {"type": "any"} if request.tool_choice == "required"
                else {"type": request.tool_choice}
            )
        if request.response_schema is not None:
            # validate_generation_request already rejects this combined
            # with web_search or caller tools, so overwriting is safe.
            body["tools"] = [
                {
                    "name": _STRUCTURED_OUTPUT_TOOL_NAME,
                    "description": "Return the structured response.",
                    "input_schema": dict(request.response_schema),
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL_NAME}
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
        for block in payload.get("content", []):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(TextOutput(text=str(block.get("text", ""))))
                for note in block.get("citations") or []:
                    if isinstance(note, dict) and note.get("url"):
                        citations.append(
                            Citation(
                                url=str(note["url"]),
                                title=note.get("title"),
                                cited_text=note.get("cited_text"),
                            )
                        )
            elif block_type == "tool_use" and block.get("name") != _STRUCTURED_OUTPUT_TOOL_NAME:
                parts.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=self.parse_tool_arguments(block.get("input", {})),
                    )
                )
            elif block_type == "tool_use" and block.get("name") == _STRUCTURED_OUTPUT_TOOL_NAME:
                # The forced structured-output tool: its input *is* the
                # answer. Serialize back to a JSON string so result.text
                # carries JSON-as-a-string uniformly across every provider,
                # regardless of which mechanism produced it.
                parts.append(TextOutput(text=json.dumps(block.get("input", {}))))
            elif block_type in ("thinking", "server_tool_use", "web_search_tool_result"):
                continue  # traces of server-side work, not output content
            else:
                parts.append(UnknownOutput(provider_kind=str(block_type or "?")))

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
            provider_request_id=safe_request_id(headers.get("request-id")),
            finish_reason=payload.get("stop_reason"),
            citations=dedupe_citations(citations),
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
