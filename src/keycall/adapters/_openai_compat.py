"""OpenAI-compatible adapter: GET /models, POST /chat/completions.

Serves DeepSeek, Perplexity, Moonshot, and explicit custom targets. Only
the conventional Chat Completions surface is assumed; provider-specific
extensions aren't (registry research section 9).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from .._classify import classify_model_id
from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._registry import ResolvedProvider
from .._transport import RequestSpec
from .._types import (
    Citation,
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
    ToolResult,
    Usage,
)
from ._base import (
    ProviderAdapter,
    StreamAssembler,
    context_limit,
    dedupe_citations,
    image_media_type,
    released_at,
)


def _provider_units(usage_raw: dict[str, Any]) -> tuple[tuple[str, float], ...] | None:
    """Billing the token counts don't cover. Perplexity reports a `cost`
    object whose `request_cost` is charged per call rather than per token
    (verified 2026-08-09), which is money a token budget can't see. Only
    numeric entries are carried; a descriptive field like
    search_context_size is not a unit."""
    cost = usage_raw.get("cost")
    if not isinstance(cost, dict):
        return None
    units = tuple(
        (str(name), float(value))
        for name, value in cost.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return units or None


def _image_url(part: ImageInput) -> str:
    """The compat family takes a URL or an inline data URL in one field.
    Moonshot rejects remote URLs, which the validation gate refuses before
    the request is built, so only data URLs reach it here."""
    if part.url is not None:
        return part.url
    encoded = base64.b64encode(part.data or b"").decode()
    return f"data:{image_media_type(part, provider='compat')};base64,{encoded}"

# Providers confirmed to honor stream_options include_usage (live-verified
# 2026-08-08). Unverified custom targets don't get the extra field: an
# unknown endpoint may reject it, and a missing-usage warning is the safer
# failure.
_STREAM_USAGE_PROVIDERS = frozenset({"deepseek", "moonshot", "xai"})


class CompatStreamAssembler(StreamAssembler):
    """Chat Completions chunk stream: choices[0].delta.content fragments,
    usage on the chunk that carries it, `data: [DONE]` terminal."""

    def __init__(self, resolved: ResolvedProvider, request: TextGenerationRequest) -> None:
        super().__init__(resolved, request)
        self._started = False
        self._saw_reasoning = False

    def _chunk_events(self, payload: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if payload.get("model"):
            self.model = str(payload["model"])
        if not self._started:
            self._started = True
            events.append(StreamStart(model=self.model))
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(choice, dict):
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if delta.get("content"):
                    text = str(delta["content"])
                    self.append_text(text)
                    events.append(TextDelta(text=text))
                elif delta.get("reasoning_content"):
                    self._saw_reasoning = True
                    # Surfaced so a long think is visible progress, not a
                    # hang: grok-4.6 reasoned 40 s before its first answer
                    # token (observed 2026-08-14).
                    events.append(ReasoningDelta(text=str(delta["reasoning_content"])))
                events.extend(self._tool_call_events(delta))
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
                # This protocol has no per-call end marker: the message
                # finishing is what closes every call still open.
                events.extend(self.flush_tool_calls())
        usage_raw = payload.get("usage")
        if isinstance(usage_raw, dict) and usage_raw.get("completion_tokens") is not None:
            details = usage_raw.get("prompt_tokens_details") or {}
            self.usage = Usage(
                input_tokens=usage_raw.get("prompt_tokens"),
                output_tokens=usage_raw.get("completion_tokens"),
                cached_input_tokens=details.get("cached_tokens")
                or usage_raw.get("prompt_cache_hit_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            )
            self.usage_reported = True
        return events

    def _tool_call_events(self, delta: dict[str, Any]) -> list[StreamEvent]:
        """delta.tool_calls entries are index-keyed: the first fragment for
        an index carries id and name, later ones append argument text
        (live-verified 2026-08-08)."""
        raw_calls = delta.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []
        events: list[StreamEvent] = []
        for entry in raw_calls:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index", 0)
            function = entry.get("function")
            function = function if isinstance(function, dict) else {}
            if index not in self._pending_calls:
                events.append(
                    self.begin_tool_call(
                        index,
                        call_id=str(entry.get("id", "")),
                        name=str(function.get("name", "")),
                    )
                )
            events.extend(self.append_tool_arguments(index, str(function.get("arguments") or "")))
        return events

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        if data.strip() == "[DONE]":
            self.saw_terminal = True
            events = self.flush_tool_calls()
            events.append(StreamFinish(finish_reason=self.finish_reason, usage=self.usage))
            return events
        payload = self._parse_data(data)
        if not isinstance(payload, dict):
            return []
        return self._chunk_events(payload)

    def finalize(self, *, round_trip_duration_ms: float) -> InvocationResult:
        if self._saw_reasoning and not self.text:
            self.warnings.append(
                "provider produced a reasoning trace but no final answer — "
                "max_output_tokens was likely too small for this model to "
                "finish reasoning and still emit content; try a larger value"
            )
        return super().finalize(round_trip_duration_ms=round_trip_duration_ms)


class OpenAICompatibleAdapter(ProviderAdapter):
    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        spec = self.build_generation_spec(request)
        body = {**(spec.json_body or {}), "stream": True}
        if self.resolved.provider in _STREAM_USAGE_PROVIDERS:
            body["stream_options"] = {"include_usage": True}
        return RequestSpec(
            method=spec.method,
            path=spec.path,
            params=spec.params,
            json_body=body,
            headers=spec.headers,
        )

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        return CompatStreamAssembler(self.resolved, request)

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
                    # Moonshot reports a unix `created` and a
                    # `context_length`; DeepSeek and other compat providers
                    # report neither, and get None for both. One code path
                    # covers the family because the readers key off the
                    # field being present, not off which provider it is.
                    released_at=released_at(entry),
                    context_limit=context_limit(entry),
                    classification_source="keycall_rule",
                )
            )
        return models, None

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            text = "\n".join(
                part.text for part in message.content if isinstance(part, TextInput)
            )
            calls = [part for part in message.content if isinstance(part, ToolCall)]
            results = [part for part in message.content if isinstance(part, ToolResult)]
            if message.role == "assistant" and calls:
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(dict(call.arguments)),
                            },
                        }
                        for call in calls
                    ],
                }
                messages.append(entry)
                continue
            # Chat Completions carries tool results as their own messages
            # with a role nothing else uses; a user turn mixing results and
            # text splits, results first (they answer the preceding
            # assistant turn).
            for result in results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": self.tool_result_text(result.content),
                    }
                )
            images = [part for part in message.content if isinstance(part, ImageInput)]
            if images:
                # Multimodal turns take the array content form; text-only
                # turns keep the plain string every compat endpoint accepts.
                content: list[dict[str, Any]] = []
                if text:
                    content.append({"type": "text", "text": text})
                for image in images:
                    content.append(
                        {"type": "image_url", "image_url": {"url": _image_url(image)}}
                    )
                messages.append({"role": message.role, "content": content})
            elif text or not results:
                messages.append({"role": message.role, "content": text})
        body: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        body.update(self.sampling_fields(request))
        if request.reasoning_effort is not None:
            # Only providers whose catalog entry records a live-verified
            # binding control reach this line; the gate refuses the rest
            # (DeepSeek answers 200 to this field but its reasoning token
            # counts do not follow the value, measured 2026-08-14).
            body["reasoning_effort"] = request.reasoning_effort
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        # input_schema=None (custom tool) is OpenAI's own
                        # Responses-API convention and gated before this
                        # point on the compat family, xAI included.
                        "parameters": dict(tool.input_schema or {}),
                    },
                }
                for tool in request.tools
            ]
        if request.tool_choice is not None:
            # Passed through as-is; support varies by provider and model
            # (DeepSeek thinking models reject "required", live-verified
            # 2026-08-08) and the provider's typed error answers.
            body["tool_choice"] = request.tool_choice
        if request.response_schema is not None:
            from .._capabilities import JSON_SCHEMA_COMPAT_PROVIDERS

            if self.resolved.provider in JSON_SCHEMA_COMPAT_PROVIDERS:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "keycall_response",
                        "schema": dict(request.response_schema),
                        "strict": True,
                    },
                }
            else:
                # Not every OpenAI-compatible endpoint supports strict
                # json_schema (DeepSeek returns 400 for it, live-verified
                # 2026-08-06); json_object is the broadly-supported floor.
                # The caller finds out via a result warning, not a guess
                # that later 400s on them.
                body["response_format"] = {"type": "json_object"}
                # DeepSeek hard-requires the literal word "json" to appear
                # in the prompt for this response_format or it 400s outright
                # (live-verified 2026-08-06: "Prompt must contain the word
                # 'json' in some form..."). OpenAI's own docs recommend the
                # same for json_object mode generally (unenforced, quality
                # risk rather than a hard error there) — applying the same
                # defensive instruction to any unverified custom target
                # covers that case too. Inserted first, conventional
                # position for a system instruction; a matching result
                # warning is added in _client.py so this isn't silent.
                from .._capabilities import mentions_json

                if not mentions_json(request.messages):
                    messages.insert(
                        0, {"role": "system", "content": "Respond only with a single valid JSON object."}
                    )
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
                if isinstance(message, dict):
                    for raw_call in message.get("tool_calls") or []:
                        if not isinstance(raw_call, dict):
                            continue
                        function = raw_call.get("function")
                        if not isinstance(function, dict):
                            continue
                        parts.append(
                            ToolCall(
                                id=str(raw_call.get("id", "")),
                                name=str(function.get("name", "")),
                                arguments=self.parse_tool_arguments(function.get("arguments")),
                            )
                        )
                if isinstance(message, dict) and message.get("content"):
                    parts.append(TextOutput(text=str(message["content"])))
                elif isinstance(message, dict) and message.get("reasoning_content"):
                    # Reasoning-capable models (Moonshot/Kimi) spend part of
                    # max_output_tokens on a visible reasoning trace before
                    # the final answer; too small a budget truncates before
                    # any content is emitted at all, leaving reasoning_content
                    # populated but content empty (live-verified 2026-08-06:
                    # reproduced at max_output_tokens=100, resolved at 200).
                    # Silent empty output here would be exactly the kind of
                    # unexplained failure this whole codebase avoids.
                    warnings.append(
                        "provider produced a reasoning trace but no final answer — "
                        "max_output_tokens was likely too small for this model to "
                        "finish reasoning and still emit content; try a larger value"
                    )

        usage_raw = payload.get("usage")
        if isinstance(usage_raw, dict):
            details = usage_raw.get("prompt_tokens_details") or {}
            usage = Usage(
                input_tokens=usage_raw.get("prompt_tokens"),
                output_tokens=usage_raw.get("completion_tokens"),
                cached_input_tokens=details.get("cached_tokens")
                or usage_raw.get("prompt_cache_hit_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
                provider_units=_provider_units(usage_raw),
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
            citations=dedupe_citations(citations),
            warnings=tuple(warnings),
        )
