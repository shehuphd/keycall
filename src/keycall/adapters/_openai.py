"""OpenAI adapter: GET /models, POST /responses (Responses API)."""

from __future__ import annotations

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
    InvocationResult,
    Model,
    OutputPart,
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextGenerationRequest,
    TextOutput,
    UnknownOutput,
    UnknownStreamEvent,
    Usage,
)
from ._base import InbandStreamError, ProviderAdapter, StreamAssembler

# Stream plumbing events that carry no content of their own; the terminal
# response.completed/incomplete event carries the whole final response.
# Live-verified 2026-08-08.
_STREAM_PLUMBING = frozenset(
    {
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
    }
)


class _OpenAIStreamAssembler(StreamAssembler):
    def __init__(self, resolved, request, adapter: OpenAIAdapter) -> None:
        super().__init__(resolved, request)
        self._adapter = adapter
        self._final: InvocationResult | None = None

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
        import dataclasses

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
        if request.web_search:
            body["tools"] = [{"type": "web_search"}]
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
            elif item_type in ("reasoning", "web_search_call"):
                continue  # traces of server-side work, not output content
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
            citations=tuple(citations),
            warnings=tuple(warnings),
        )
