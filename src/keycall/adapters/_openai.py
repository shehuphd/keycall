"""OpenAI adapter: GET /models, POST /responses (Responses API)."""

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
    dedupe_citations,
    image_media_type,
    media_type_for,
)

# Stream plumbing events that carry no content of their own; the terminal
# response.completed/incomplete event carries the whole final response.
# Live-verified 2026-08-08.
_STREAM_PLUMBING = frozenset(
    {
        "response.in_progress",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
    }
)


def _image_url(part: ImageInput, *, provider: str = "openai") -> str:
    """OpenAI takes both an https URL and an inline data URL in the same
    field (both verified 2026-08-09)."""
    if part.url is not None:
        return part.url
    encoded = base64.b64encode(part.data or b"").decode()
    return f"data:{image_media_type(part, provider=provider)};base64,{encoded}"


def _call_echo(item: dict[str, Any], reasoning: dict[str, Any] | None) -> str | None:
    """Provider echo data a function_call needs back verbatim on replay:
    its own item id, plus the reasoning item it belongs to when the model
    produced one. Reasoning items appear only when the model actually
    reasons, so a response can legitimately carry none."""
    echo: dict[str, Any] = {}
    if item.get("id"):
        echo["id"] = item["id"]
    if reasoning is not None:
        echo["reasoning"] = reasoning
    return json.dumps(echo) if echo else None


class _OpenAIStreamAssembler(StreamAssembler):
    def __init__(self, resolved, request, adapter: OpenAIAdapter) -> None:
        super().__init__(resolved, request)
        self._adapter = adapter
        self._final: InvocationResult | None = None
        # Reasoning items arrive as their own output item before the call
        # they belong to; the call cannot be replayed without one.
        self._reasoning: dict[str, Any] | None = None

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        payload = self._parse_data(data)
        if not isinstance(payload, dict):
            return []
        kind = str(payload.get("type", ""))
        if kind == "response.created":
            response = payload.get("response")
            if isinstance(response, dict) and response.get("model"):
                self.model = str(response["model"])
            return [StreamStart(model=self.model)]
        if kind == "response.output_text.delta":
            delta = str(payload.get("delta", ""))
            self.append_text(delta)
            return [TextDelta(text=delta)]
        if kind == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "reasoning":
                self._reasoning = item
            elif isinstance(item, dict) and item.get("type") == "function_call":
                # Keyed by item id, which is what the argument deltas
                # reference; the call_id is the id the caller replies with.
                return [
                    self.begin_tool_call(
                        str(item.get("id", "")),
                        call_id=str(item.get("call_id", "")),
                        name=str(item.get("name", "")),
                        opaque=_call_echo(item, self._reasoning),
                    )
                ]
            return []
        if kind == "response.function_call_arguments.delta":
            return self.append_tool_arguments(
                str(payload.get("item_id", "")), str(payload.get("delta", ""))
            )
        if kind == "response.function_call_arguments.done":
            return self.complete_tool_call(
                str(payload.get("item_id", "")), arguments=payload.get("arguments")
            )
        if kind in ("response.completed", "response.incomplete"):
            self.saw_terminal = True
            self._final = self._adapter.parse_generation_response(
                payload.get("response"),
                headers=self.response_headers,
                round_trip_duration_ms=0.0,
                model=self.model,
            )
            self.usage = self._final.usage
            self.usage_reported = self._final.usage.output_tokens is not None
            self.finish_reason = self._final.finish_reason
            self.model = self._final.model
            self.provider_request_id = self._final.provider_request_id
            self.citations = list(self._final.citations)
            self.warnings.extend(self._final.warnings)
            # Citations surface at completion in the Responses stream.
            events: list[StreamEvent] = [
                CitationFound(citation=c) for c in self._final.citations
            ]
            events.append(StreamFinish(finish_reason=self.finish_reason, usage=self.usage))
            return events
        if kind == "response.failed":
            response = payload.get("response")
            error = response.get("error") if isinstance(response, dict) else None
            message = str(error.get("message", "generation failed")) if isinstance(error, dict) else "generation failed"
            raise InbandStreamError(ErrorCode.PROVIDER_UNAVAILABLE, False, message)
        if kind in _STREAM_PLUMBING:
            return []
        return [UnknownStreamEvent(provider_kind=kind or "?")]

    def finalize(self, *, round_trip_duration_ms: float) -> InvocationResult:
        # The terminal event carried the complete response; reuse the full
        # non-streaming parse so both paths produce identical results.
        assert self._final is not None
        return dataclasses.replace(self._final, round_trip_duration_ms=round_trip_duration_ms)


class OpenAIAdapter(ProviderAdapter):
    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        spec = self.build_generation_spec(request)
        return RequestSpec(
            method=spec.method,
            path=spec.path,
            params=spec.params,
            json_body={**(spec.json_body or {}), "stream": True},
        )

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        return _OpenAIStreamAssembler(self.resolved, request, self)

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
        input_items: list[dict[str, Any]] = []
        replayed_reasoning: set[str] = set()
        for message in request.messages:
            # Responses API: assistant history uses output_text parts, and
            # tool calls/results are top-level input items, not message
            # content (live-verified 2026-08-08).
            part_type = "output_text" if message.role == "assistant" else "input_text"
            content: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextInput):
                    content.append({"type": part_type, "text": part.text})
                elif isinstance(part, ImageInput):
                    content.append({"type": "input_image", "image_url": _image_url(part)})
                elif isinstance(part, FileInput):
                    # Documents go as their own item type with a data URL,
                    # and the Responses API wants a filename alongside it
                    # (verified 2026-08-09 with a PDF).
                    encoded = base64.b64encode(part.data or b"").decode()
                    media = media_type_for(part, kind="file", provider="openai")
                    content.append(
                        {
                            "type": "input_file",
                            "filename": part.filename or "document.pdf",
                            "file_data": f"data:{media};base64,{encoded}",
                        }
                    )
            if content:
                input_items.append({"role": message.role, "content": content})
            for part in message.content:
                if isinstance(part, ToolCall):
                    echo = json.loads(part.opaque) if part.opaque else {}
                    # The reasoning item goes back as its own input item,
                    # ahead of the call it belongs to. Parallel calls share
                    # one, so it is emitted once.
                    reasoning = echo.pop("reasoning", None)
                    if isinstance(reasoning, dict):
                        reasoning_id = str(reasoning.get("id", ""))
                        if reasoning_id not in replayed_reasoning:
                            replayed_reasoning.add(reasoning_id)
                            input_items.append(reasoning)
                    item: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": part.id,
                        "name": part.name,
                        "arguments": json.dumps(dict(part.arguments)),
                    }
                    item.update(echo)
                    input_items.append(item)
                elif isinstance(part, ToolResult):
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": part.tool_call_id,
                            "output": self.tool_result_text(part.content),
                        }
                    )
        body: dict[str, Any] = {"model": request.model, "input": input_items}
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
        body.update(self.sampling_fields(request))
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            }
            for tool in request.tools
        ]
        if request.web_search:
            tools.append({"type": "web_search"})
        if tools:
            body["tools"] = tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.response_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "keycall_response",
                    "schema": dict(request.response_schema),
                    "strict": True,
                }
            }
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
        # The reasoning item that precedes a function_call has to travel
        # back with it: replaying the call without it is an HTTP 400 on
        # reasoning models (verified 2026-08-09, three runs out of three).
        reasoning: dict[str, Any] | None = None
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", ""))
            if item_type == "message":
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        parts.append(TextOutput(text=str(content.get("text", ""))))
                        for note in content.get("annotations") or []:
                            if isinstance(note, dict) and note.get("type") == "url_citation":
                                citations.append(
                                    Citation(
                                        url=str(note.get("url", "")),
                                        title=note.get("title"),
                                    )
                                )
                    elif isinstance(content, dict):
                        parts.append(UnknownOutput(provider_kind=str(content.get("type", "?"))))
            elif item_type == "function_call":
                parts.append(
                    ToolCall(
                        id=str(item.get("call_id", "")),
                        name=str(item.get("name", "")),
                        arguments=self.parse_tool_arguments(item.get("arguments")),
                        opaque=_call_echo(item, reasoning),
                    )
                )
            elif item_type == "reasoning":
                # Not output content, but required echo data for any call
                # that follows it in this response.
                reasoning = item
            elif item_type == "web_search_call":
                continue  # trace of server-side work, not output content
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
            provider_request_id=safe_request_id(headers.get("x-request-id")),
            finish_reason=str(finish) if finish else None,
            citations=dedupe_citations(citations),
            warnings=tuple(warnings),
        )
