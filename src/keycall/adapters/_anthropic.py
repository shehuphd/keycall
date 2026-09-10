"""Anthropic adapter: GET /v1/models (cursor pagination), POST /v1/messages."""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import urlsplit

from .._classify import classify_model_id
from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._registry import ResolvedProvider
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    BatchCounts,
    BatchJob,
    BatchStatus,
    Citation,
    CitationFound,
    CodeExecutionOutput,
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
    BatchItemOutcome,
    BatchSubmission,
    InbandStreamError,
    ProviderAdapter,
    StreamAssembler,
    batch_line_entries,
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

# Beta feature flag code_interpreter needs (live-verified 2026-08-22); sent
# only on a request that asks for it, never on every Anthropic request.
_CODE_EXECUTION_BETA_HEADER = "code-execution-2025-08-25"

# Anthropic's own two TTL strings; the pre-flight gate in _base.py already
# refuses any other cache_ttl_seconds value before this is ever reached.
_CACHE_TTL_LABEL = {300: "5m", 3600: "1h"}


def _text_block(part: TextInput) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": part.text}
    if part.cacheable:
        block["cache_control"] = {
            "type": "ephemeral",
            "ttl": _CACHE_TTL_LABEL[part.cache_ttl_seconds],
        }
    return block


def _bash_code_execution_output(
    call_block: Mapping[str, Any], result_block: Mapping[str, Any]
) -> CodeExecutionOutput:
    """Build a CodeExecutionOutput from a paired server_tool_use(name=
    bash_code_execution)/bash_code_execution_tool_result block, matched by
    tool_use_id. Only this pair is normalized; a chained
    text_editor_code_execution call (Anthropic sometimes authors a file
    with one server tool, then runs it with this one) falls through to
    UnknownOutput — closer in kind to apply_patch's file editing than to
    code execution, and out of scope here."""
    command = (call_block.get("input") or {}).get("command", "")
    content = result_block.get("content")
    stdout = content.get("stdout", "") if isinstance(content, Mapping) else ""
    return CodeExecutionOutput(code=str(command), output=str(stdout), language="bash")


class _AnthropicStreamAssembler(StreamAssembler):
    """Event names and formats live-verified 2026-08-08: message_start,
    content_block_start/delta/stop, message_delta (usage + stop_reason),
    message_stop terminal, ping keep-alives, in-band error events."""

    def __init__(
        self,
        resolved: ResolvedProvider,
        request: TextGenerationRequest,
        adapter: AnthropicAdapter,
    ) -> None:
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
                return [
                    self.begin_tool_call(
                        index, call_id=str(block.get("id", "")), name=name
                    )
                ]
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
            headers=spec.headers,
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
        system_parts: list[TextInput] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                # Anthropic takes system content as a top-level parameter.
                system_parts.extend(
                    part for part in message.content if isinstance(part, TextInput)
                )
                continue
            blocks: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextInput):
                    blocks.append(_text_block(part))
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
        if system_parts:
            # A block array only when a marker is present: Anthropic
            # reads a plain string and an all-plain block array as the same
            # prompt, but switching unconditionally would be a needless
            # shape change for every caller who never touches caching.
            if any(part.cacheable for part in system_parts):
                body["system"] = [_text_block(part) for part in system_parts]
            else:
                body["system"] = "\n\n".join(part.text for part in system_parts)
        body.update(self.sampling_fields(request))
        if request.reasoning_effort is not None:
            # Anthropic's control is output_config.effort; a top-level
            # `effort` field is refused with "Extra inputs are not
            # permitted" (live-verified 2026-08-14 on claude-opus-4-5).
            body["output_config"] = {"effort": request.reasoning_effort}
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            tool_def: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                # input_schema=None (custom tool) is OpenAI-only and gated
                # before this point on every other provider, Anthropic
                # included.
                "input_schema": dict(tool.input_schema or {}),
            }
            if tool.defer_loading:
                tool_def["defer_loading"] = True
            tools.append(tool_def)
        if request.web_search:
            tools.append({"type": "web_search_20250305", "name": "web_search"})
        if request.code_interpreter:
            tools.append({"type": "code_execution_20250825", "name": "code_execution"})
        if any(tool.defer_loading for tool in request.tools):
            # BM25 (natural-language queries), the friendlier of the two
            # search variants for a caller not hand-writing regex.
            tools.append(
                {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
            )
        if tools:
            body["tools"] = tools
        if request.tool_choice is not None:
            # Anthropic's spellings, live-verified 2026-08-08.
            body["tool_choice"] = (
                {"type": "any"} if request.tool_choice == "required"
                else {"type": request.tool_choice}
            )
        if request.response_schema is not None:
            # Anthropic's native structured output, rather than forcing a
            # synthetic tool: claude-fable-5-1 refuses tool_choice types
            # "tool" and "any" outright, while the native format holds on
            # every currently listed model down to the 4.5 snapshots,
            # streaming included, and composes with caller tools and
            # web_search where a forced tool cannot (all live-verified
            # 2026-09-10). setdefault: reasoning_effort may already have
            # opened output_config above.
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": dict(request.response_schema),
            }
        headers = (
            {"anthropic-beta": _CODE_EXECUTION_BETA_HEADER} if request.code_interpreter else {}
        )
        return RequestSpec(method=op["method"], path=op["path"], json_body=body, headers=headers)

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
        pending_bash_exec: dict[str, dict[str, Any]] = {}
        for block in payload.get("content", []):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "server_tool_use" and block.get("name") == "bash_code_execution":
                pending_bash_exec[str(block.get("id", ""))] = block
                continue
            if block_type == "bash_code_execution_tool_result":
                call_block = pending_bash_exec.pop(str(block.get("tool_use_id", "")), None)
                parts.append(
                    _bash_code_execution_output(call_block, block)
                    if call_block is not None
                    else UnknownOutput(provider_kind="bash_code_execution_tool_result")
                )
            elif block_type == "text":
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
            elif block_type == "tool_use":
                parts.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=self.parse_tool_arguments(block.get("input", {})),
                    )
                )
            elif block_type in (
                "thinking",
                "web_search_tool_result",
                "tool_search_tool_result",
            ):
                continue  # traces of server-side work, not output content
            elif block_type == "server_tool_use" and block.get("name") in (
                "web_search",
                "tool_search_tool_regex",
                "tool_search_tool_bm25",
            ):
                continue  # ditto — the call side of the same trace
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

    # --- batch generation ---
    #
    # The inline dialect: requests ride the create call itself (custom_id
    # plus ordinary Messages params, so one batch can mix models), results
    # stream as JSONL from results_url, out of submission order. All
    # wire facts live-verified 2026-09-02.

    batch_mixed_models = True

    _BATCH_STATUS: ClassVar[dict[str, BatchStatus]] = {
        "in_progress": "running",
        "canceling": "running",
        "ended": "finished",
    }

    _BATCH_ERROR_CODES: ClassVar[dict[str, str]] = {
        "authentication_error": "invalid_api_key",
        "permission_error": "permission_denied",
        "billing_error": "permission_denied",
        "not_found_error": "model_not_available",
        "rate_limit_error": "rate_limited",
        "overloaded_error": "provider_unavailable",
        "api_error": "provider_unavailable",
    }

    def build_batch_submit_spec(
        self, submission: BatchSubmission, prelude: str | None
    ) -> RequestSpec:
        op = self.resolved.operations["batch_create"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            json_body={
                "requests": [
                    {"custom_id": key, "params": dict(body)}
                    for key, body in submission.items
                ]
            },
        )

    def parse_batch_submit(
        self, payload: Any, *, submission: BatchSubmission, prelude: str | None
    ) -> BatchJob:
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not job_id:
            raise KeyCallError(
                "batch create returned no batch id",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=submission.operation,
            )
        job = BatchJob(
            provider=self.resolved.provider,
            job_id=str(job_id),
            operation=submission.operation,
            model=submission.model,
            request_keys=tuple(key for key, _ in submission.items),
        )
        return self.parse_batch_status(payload, job=job)

    def build_batch_status_spec(self, job: BatchJob) -> RequestSpec:
        op = self.resolved.operations["batch_status"]
        return RequestSpec(method=op["method"], path=op["path"].replace("{batch_id}", job.job_id))

    def parse_batch_status(self, payload: Any, *, job: BatchJob) -> BatchJob:
        provider_status = (
            str(payload.get("processing_status", "")) if isinstance(payload, dict) else ""
        )
        counts_raw = payload.get("request_counts") if isinstance(payload, dict) else None
        counts = None
        if isinstance(counts_raw, dict):
            counts = BatchCounts(
                pending=counts_raw.get("processing"),
                succeeded=counts_raw.get("succeeded"),
                errored=counts_raw.get("errored"),
                cancelled=counts_raw.get("canceled"),
                expired=counts_raw.get("expired"),
                total=len(job.request_keys) or None,
            )
        results_url = payload.get("results_url") if isinstance(payload, dict) else None
        return dataclasses.replace(
            job,
            status=self._BATCH_STATUS.get(provider_status, "running"),
            provider_status=provider_status or None,
            counts=counts,
            results_ref=self._pinned_results_path(results_url) if results_url else None,
        )

    def _pinned_results_path(self, results_url: Any) -> str:
        """results_url must stay on the provider's own API host — the
        credential rides the request, so a URL pointing anywhere else is
        refused rather than followed."""
        parsed = urlsplit(str(results_url))
        own = urlsplit(self.resolved.base_url)
        if parsed.scheme != "https" or parsed.netloc != own.netloc:
            raise KeyCallError(
                "batch results_url points off the provider's API host; refusing to follow it",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.BATCH_GENERATION.value,
            )
        return parsed.path + (f"?{parsed.query}" if parsed.query else "")

    def build_batch_results_spec(self, job: BatchJob, cursor: str | None) -> RequestSpec:
        if not job.results_ref:
            raise KeyCallError(
                "this batch reported no results_url yet; poll check_batch() until it ends",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=job.operation,
            )
        return RequestSpec(method="GET", path=job.results_ref)

    def parse_batch_results(
        self, payload: Any, *, job: BatchJob
    ) -> tuple[list[BatchItemOutcome], str | None]:
        outcomes: list[BatchItemOutcome] = []
        for entry in batch_line_entries(payload):
            if not entry.get("custom_id"):
                continue
            key = str(entry["custom_id"])
            raw_result = entry.get("result")
            result = raw_result if isinstance(raw_result, dict) else {}
            kind = result.get("type")
            if kind == "succeeded" and isinstance(result.get("message"), dict):
                outcomes.append(BatchItemOutcome(key=key, body=result["message"]))
                continue
            if kind == "errored":
                # The per-request error nests one envelope deeper than the
                # HTTP error body: result.error.error carries type/message
                # (observed live 2026-09-02).
                inner = result.get("error")
                if isinstance(inner, dict) and isinstance(inner.get("error"), dict):
                    inner = inner["error"]
                error_type = str(inner.get("type", "")) if isinstance(inner, dict) else ""
                message = str(inner.get("message", "")) if isinstance(inner, dict) else ""
                outcomes.append(
                    BatchItemOutcome(
                        key=key,
                        error_code=self._BATCH_ERROR_CODES.get(error_type, error_type or None),
                        error_message=message or "request errored",
                    )
                )
                continue
            outcomes.append(
                BatchItemOutcome(
                    key=key,
                    error_code=str(kind) if kind else None,
                    error_message=f"request {kind or 'returned an unrecognized result'}",
                )
            )
        return outcomes, None

    def build_batch_cancel_spec(self, job: BatchJob) -> RequestSpec:
        op = self.resolved.operations["batch_cancel"]
        return RequestSpec(method=op["method"], path=op["path"].replace("{batch_id}", job.job_id))

    def parse_batch_cancel(self, payload: Any, *, job: BatchJob) -> BatchJob:
        return self.parse_batch_status(payload, job=job)

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        message = ""
        error_type = ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", ""))
            error_type = str(payload["error"].get("type", ""))
        if error_type == "authentication_error" or status_code == 401:
            return ErrorCode.INVALID_API_KEY, False, message or "invalid API key"
        if "credit balance" in message.lower():
            # Anthropic sends its unfunded-account refusal as a 400
            # invalid_request_error — the same status and type as a
            # malformed request (message observed live 2026-09-01) — so
            # the message content is the only signal that routes it to
            # the billing code rather than a request-format one.
            return ErrorCode.PERMISSION_DENIED, False, message
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
