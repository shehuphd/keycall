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
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextGenerationRequest,
    TextOutput,
    Usage,
)
from ._base import ProviderAdapter, StreamAssembler

# Providers confirmed to honor stream_options include_usage (live-verified
# 2026-08-08). Unverified custom targets don't get the extra field: an
# unknown endpoint may reject it, and a missing-usage warning is the safer
# failure.
_STREAM_USAGE_PROVIDERS = frozenset({"deepseek", "moonshot"})


class CompatStreamAssembler(StreamAssembler):
    """Chat Completions chunk stream: choices[0].delta.content fragments,
    usage on the chunk that carries it, `data: [DONE]` terminal."""

    def __init__(self, resolved, request) -> None:
        super().__init__(resolved, request)
        self._started = False
        self._saw_reasoning = False

    def _chunk_events(self, payload: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if payload.get("model"):
            self.model = str(payload["model"])
        if not self._started:
            self._started = True
            events.append(StreamStart(model=self.model))
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(choice, dict):
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if delta.get("content"):
                    text = str(delta["content"])
                    self.append_text(text)
                    events.append(TextDelta(text=text))
                elif delta.get("reasoning_content"):
                    self._saw_reasoning = True
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

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        if data.strip() == "[DONE]":
            self.saw_terminal = True
            return [StreamFinish(finish_reason=self.finish_reason, usage=self.usage)]
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
        return RequestSpec(method=spec.method, path=spec.path, params=spec.params, json_body=body)

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
