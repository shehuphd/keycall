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
from .._registry import ResolvedProvider
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    Citation,
    CitationFound,
    CodeExecutionOutput,
    FileInput,
    ImageInput,
    InvocationResult,
    Model,
    OutputPart,
    ReasoningDelta,
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextGenerationRequest,
    TextInput,
    TextOutput,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallComplete,
    ToolCallStarted,
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
    released_at,
)

# Stream plumbing events that carry no content of their own; the terminal
# response.completed/incomplete event carries the whole final response.
# Live-verified 2026-08-08 (apply_patch_call_operation_diff.done added
# 2026-08-22): its diff also arrives complete on response.output_item.done,
# which is what the assembler completes the call from, so the dedicated
# done event is redundant confirmation, not a second source.
_STREAM_PLUMBING = frozenset(
    {
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
        "response.apply_patch_call_operation_diff.done",
        # code_interpreter_call's code and status arrive incrementally, but
        # the terminal response.completed/incomplete event carries the same
        # item complete (live-verified 2026-08-22), which is what
        # finalize() parses via parse_generation_response — so these are
        # progress notices only, never a second source of truth.
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call.completed",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.custom_tool_call_input.done",
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
    """Provider echo data a function_call or apply_patch_call needs back
    verbatim on replay: its own item id, plus the reasoning item it
    belongs to when the model produced one. Reasoning items appear only
    when the model reasons, so a response can legitimately carry none."""
    echo: dict[str, Any] = {}
    if item.get("id"):
        echo["id"] = item["id"]
    if reasoning is not None:
        echo["reasoning"] = reasoning
    return json.dumps(echo) if echo else None


def _parse_apply_patch_call(item: dict[str, Any], reasoning: dict[str, Any] | None) -> ToolCall:
    """Build the ToolCall for one apply_patch_call item. Shared by the
    non-streaming parser and the streaming assembler's completion handler
    (response.output_item.done), since both see the identical final item
    shape — streaming's incremental diff deltas are for live display only,
    never for reassembling the result. ``arguments`` carries the operation
    dict verbatim (create_file/update_file/delete_file, path, and diff —
    diff omitted for delete_file), not caller-defined JSON to parse."""
    return ToolCall(
        id=str(item.get("call_id", "")),
        name="apply_patch",
        arguments=dict(item.get("operation") or {}),
        opaque=_call_echo(item, reasoning),
    )


def _parse_custom_tool_call(item: dict[str, Any], reasoning: dict[str, Any] | None) -> ToolCall:
    """Build the ToolCall for one custom_tool_call item. A custom tool has
    no JSON Schema, so the model's call arrives as a plain string
    (``input``) rather than parsed arguments — carried as the single key
    ``arguments["input"]`` so ToolCall's Mapping-typed arguments field
    needs no new shape to hold it."""
    return ToolCall(
        id=str(item.get("call_id", "")),
        name=str(item.get("name", "")),
        arguments={"input": str(item.get("input", ""))},
        opaque=_call_echo(item, reasoning),
    )


def _parse_code_interpreter_call(item: dict[str, Any]) -> CodeExecutionOutput:
    """Build the CodeExecutionOutput for one code_interpreter_call item.
    ``outputs`` is null even for a successful run that printed something
    (live-verified 2026-08-22) — the human-readable answer arrives
    separately as the following message's output_text, not here. When
    outputs is a non-null list of {type: "logs", logs: str} entries, its
    text is joined; otherwise output is empty and the following text part
    carries the answer on its own."""
    outputs = item.get("outputs")
    output = (
        "".join(str(o.get("logs", "")) for o in outputs if isinstance(o, dict))
        if isinstance(outputs, list)
        else ""
    )
    return CodeExecutionOutput(code=str(item.get("code", "")), output=output, language="python")


def _apply_patch_result_fields(content: str | Mapping[str, Any]) -> tuple[str, str]:
    """apply_patch_call_output needs a status ("completed" or "failed")
    alongside its output text — unlike an ordinary function_call_output,
    which is output-only. A caller that only has text to report gets
    "completed" by default; one that knows the patch failed passes a
    mapping with an explicit status instead."""
    if isinstance(content, Mapping):
        return str(content.get("status", "completed")), str(content.get("output", ""))
    return "completed", str(content)


class _OpenAIStreamAssembler(StreamAssembler):
    def __init__(
        self,
        resolved: ResolvedProvider,
        request: TextGenerationRequest,
        adapter: OpenAIAdapter,
    ) -> None:
        super().__init__(resolved, request)
        self._adapter = adapter
        self._final: InvocationResult | None = None
        # Reasoning items arrive as their own output item before the call
        # they belong to; the call can't be replayed without one.
        self._reasoning: dict[str, Any] | None = None
        # apply_patch_call item id -> call_id, so a diff delta (keyed by
        # item id) can be labeled with the id a caller replies to.
        # The operation itself (type/path/diff) isn't tracked here: unlike
        # function_call's arguments, response.output_item.done delivers it
        # complete in one shot for every operation kind, including
        # delete_file, which streams no diff deltas at all.
        self._pending_patch_calls: dict[str, str] = {}
        # Same purpose, for custom_tool_call: item id -> call_id, so an
        # input delta (keyed by item id) can be labeled with the id a
        # caller replies to.
        self._pending_custom_calls: dict[str, str] = {}

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
        if kind == "response.reasoning_summary_text.delta":
            # The visible reasoning trace, streamed ahead of the answer —
            # surfaced so a long think doesn't read as a hang.
            return [ReasoningDelta(text=str(payload.get("delta", "")))]
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
            elif isinstance(item, dict) and item.get("type") == "apply_patch_call":
                item_id = str(item.get("id", ""))
                call_id = str(item.get("call_id", ""))
                self._pending_patch_calls[item_id] = call_id
                return [ToolCallStarted(id=call_id, name="apply_patch")]
            elif isinstance(item, dict) and item.get("type") == "custom_tool_call":
                item_id = str(item.get("id", ""))
                call_id = str(item.get("call_id", ""))
                self._pending_custom_calls[item_id] = call_id
                return [ToolCallStarted(id=call_id, name=str(item.get("name", "")))]
            return []
        if kind == "response.function_call_arguments.delta":
            return self.append_tool_arguments(
                str(payload.get("item_id", "")), str(payload.get("delta", ""))
            )
        if kind == "response.function_call_arguments.done":
            return self.complete_tool_call(
                str(payload.get("item_id", "")), arguments=payload.get("arguments")
            )
        if kind == "response.apply_patch_call_operation_diff.delta":
            patch_call_id = self._pending_patch_calls.get(str(payload.get("item_id", "")))
            if patch_call_id is None:
                return []
            return [
                ToolCallArgumentsDelta(id=patch_call_id, fragment=str(payload.get("delta", "")))
            ]
        if kind == "response.custom_tool_call_input.delta":
            custom_call_id = self._pending_custom_calls.get(str(payload.get("item_id", "")))
            if custom_call_id is None:
                return []
            return [
                ToolCallArgumentsDelta(id=custom_call_id, fragment=str(payload.get("delta", "")))
            ]
        if kind == "response.output_item.done":
            item = payload.get("item")
            if not isinstance(item, dict):
                return []
            if item.get("type") == "apply_patch_call":
                self._pending_patch_calls.pop(str(item.get("id", "")), None)
                call = _parse_apply_patch_call(item, self._reasoning)
                self.tool_calls.append(call)
                return [ToolCallComplete(tool_call=call)]
            if item.get("type") == "custom_tool_call":
                self._pending_custom_calls.pop(str(item.get("id", "")), None)
                call = _parse_custom_tool_call(item, self._reasoning)
                self.tool_calls.append(call)
                return [ToolCallComplete(tool_call=call)]
            return []  # plumbing for every other item type
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
            headers=spec.headers,
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
                    released_at=released_at(entry),
                    classification_source="keycall_rule",
                )
            )
        return models, None  # GET /models is unpaginated

    def build_image_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["image_generation"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            json_body={"model": request.model, "prompt": request.prompt},
        )

    def parse_image_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise KeyCallError(
                "image response missing 'data' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.IMAGE_GENERATION.value,
            )
        # The response names its own format; b64_json is what the images
        # endpoint returns by default (verified 2026-08-10).
        media_type = f"image/{payload.get('output_format', 'png')}"
        images = [
            (str(entry["b64_json"]), media_type)
            for entry in payload["data"]
            if isinstance(entry, dict) and entry.get("b64_json")
        ]
        usage_raw = payload.get("usage") or {}
        return self.image_result(
            images,
            usage=Usage(
                input_tokens=usage_raw.get("input_tokens"),
                output_tokens=usage_raw.get("output_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            ),
            model=model,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(
                headers.get(self.resolved.provider_request_id_header or "")
            ),
        )

    def build_speech_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["speech_generation"]
        body: dict[str, Any] = {"model": request.model, "input": request.text}
        if request.voice:
            body["voice"] = request.voice
        return RequestSpec(method=op["method"], path=op["path"], json_body=body)

    def parse_speech_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        # This endpoint answers with the audio file itself, not a JSON
        # envelope (verified live 2026-08-12), so the transport hands the
        # raw bytes straight through rather than a parsed dict — this is
        # the one adapter method in the package whose payload is not JSON.
        if not isinstance(payload, bytes) or not payload:
            raise KeyCallError(
                "speech response was not audio bytes",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.SPEECH_GENERATION.value,
            )
        # The response's own Content-Type is authoritative for what format
        # came back; nothing here should assume a fixed format,
        # since response_format on this endpoint isn't sent by KeyCall and
        # OpenAI's own default has changed across model generations.
        media_type = (headers.get("content-type") or "audio/mpeg").split(";")[0].strip()
        return self.speech_result(
            base64_data=base64.b64encode(payload).decode("ascii"),
            media_type=media_type,
            # No usage is reported anywhere on this response: no JSON body
            # to carry it in, and no dedicated usage header either. None
            # here means "not reported", the same as everywhere else.
            usage=Usage(),
            model=model,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(
                headers.get(self.resolved.provider_request_id_header or "")
            ),
        )

    def build_embedding_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["embeddings"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            json_body={"model": request.model, "input": list(request.inputs)},
        )

    def parse_embedding_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
        expected: int,
    ) -> InvocationResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise KeyCallError(
                "embedding response missing 'data' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.EMBEDDING.value,
            )
        # data entries carry an index; order by it rather than trusting
        # arrival order to line vectors up with their inputs.
        entries = sorted(payload["data"], key=lambda e: e.get("index", 0))
        vectors = [tuple(float(v) for v in e.get("embedding", ())) for e in entries]
        usage_raw = payload.get("usage") or {}
        return self.embedding_result(
            vectors,
            usage=Usage(
                input_tokens=usage_raw.get("prompt_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            ),
            model=str(payload.get("model", model)),
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(
                headers.get(self.resolved.provider_request_id_header or "")
            ),
            expected=expected,
        )

    def realtime_plan(self, config: Any) -> tuple[str, Any]:
        if not self.resolved.capabilities.realtime or "realtime" not in self.resolved.operations:
            return super().realtime_plan(config)
        from urllib.parse import quote

        from ._realtime import OpenAIRealtimeTranslator

        path = self.resolved.operations["realtime"]["path"].format(
            model=quote(config.model, safe="")
        )
        translator = OpenAIRealtimeTranslator(
            config, provider=self.resolved.provider, ga_session=True
        )
        return path, translator

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        input_items: list[dict[str, Any]] = []
        replayed_reasoning: set[str] = set()
        custom_tool_names = {tool.name for tool in request.tools if tool.input_schema is None}
        any_cache_marker = False
        for message in request.messages:
            # Responses API: assistant history uses output_text parts, and
            # tool calls/results are top-level input items, not message
            # content (live-verified 2026-08-08).
            part_type = "output_text" if message.role == "assistant" else "input_text"
            content: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextInput):
                    text_item: dict[str, Any] = {"type": part_type, "text": part.text}
                    if part.cacheable:
                        any_cache_marker = True
                        text_item["prompt_cache_breakpoint"] = {"mode": "explicit"}
                    content.append(text_item)
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
                    item: dict[str, Any]
                    if part.name == "apply_patch":
                        item = {
                            "type": "apply_patch_call",
                            "call_id": part.id,
                            "status": "completed",
                            "operation": dict(part.arguments),
                        }
                    elif part.name in custom_tool_names:
                        item = {
                            "type": "custom_tool_call",
                            "call_id": part.id,
                            "name": part.name,
                            "input": str(part.arguments.get("input", "")),
                        }
                    else:
                        item = {
                            "type": "function_call",
                            "call_id": part.id,
                            "name": part.name,
                            "arguments": json.dumps(dict(part.arguments)),
                        }
                    item.update(echo)
                    input_items.append(item)
                elif isinstance(part, ToolResult):
                    if part.name == "apply_patch":
                        status, output = _apply_patch_result_fields(part.content)
                        input_items.append(
                            {
                                "type": "apply_patch_call_output",
                                "call_id": part.tool_call_id,
                                "status": status,
                                "output": output,
                            }
                        )
                    elif part.name in custom_tool_names:
                        input_items.append(
                            {
                                "type": "custom_tool_call_output",
                                "call_id": part.tool_call_id,
                                "output": self.tool_result_text(part.content),
                            }
                        )
                    else:
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": part.tool_call_id,
                                "output": self.tool_result_text(part.content),
                            }
                        )
        body: dict[str, Any] = {"model": request.model, "input": input_items}
        if any_cache_marker:
            # Left unset otherwise: OpenAI's implicit (automatic) caching
            # already runs on every request with no marker at all, and this
            # mode switch only needs sending when a caller asks for an
            # explicit breakpoint.
            body["prompt_cache_options"] = {"mode": "explicit"}
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
        body.update(self.sampling_fields(request))
        if request.reasoning_effort is not None:
            # The Responses effort control (live-verified 2026-08-14);
            # xAI's responses surface takes the same shape.
            body["reasoning"] = {"effort": request.reasoning_effort}
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if tool.input_schema is None:
                tools.append({"type": "custom", "name": tool.name, "description": tool.description})
                continue
            function_tool: dict[str, Any] = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            }
            if tool.defer_loading:
                function_tool["defer_loading"] = True
            tools.append(function_tool)
        if request.web_search:
            tools.append({"type": "web_search"})
        if request.apply_patch:
            tools.append({"type": "apply_patch"})
        if request.code_interpreter:
            tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        if any(tool.defer_loading for tool in request.tools):
            tools.append({"type": "tool_search"})
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
            elif item_type == "apply_patch_call":
                parts.append(_parse_apply_patch_call(item, reasoning))
            elif item_type == "custom_tool_call":
                parts.append(_parse_custom_tool_call(item, reasoning))
            elif item_type == "code_interpreter_call":
                parts.append(_parse_code_interpreter_call(item))
            elif item_type == "reasoning":
                # Not output content, but required echo data for any call
                # that follows it in this response.
                reasoning = item
            elif item_type in ("web_search_call", "tool_search_call", "tool_search_output"):
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
