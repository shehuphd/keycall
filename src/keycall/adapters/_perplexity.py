"""Perplexity adapter.

Perplexity needs its own adapter because its two surfaces don't line up:
``GET /v1/models`` is scoped to the Agent API and returns vendor-prefixed
router models (``anthropic/claude-...``, ``perplexity/sonar``) that the
Sonar generation route rejects outright. Sonar's own models aren't
API-discoverable.

So the list call still runs — it proves the credential is valid, which is
half of what list_models is for — but the models KeyCall returns come from
maintained catalog metadata, the classification tier used where provider
metadata is inadequate.

Verified live 2026-08-05: /v1/sonar accepts bare ``sonar``/``sonar-pro``/
``sonar-reasoning-pro``, rejects ``perplexity/sonar`` and every
``anthropic/*`` entry from the list endpoint; ``sonar-reasoning`` is
deprecated provider-side; and max_tokens must be >= 16.
"""

from __future__ import annotations

from typing import Any

from .._enums import ModelCategory
from .._errors import ErrorCode, KeyCallError
from .._transport import RequestSpec
from .._types import (
    Citation,
    CitationFound,
    Model,
    StreamEvent,
    StreamFinish,
    TextGenerationRequest,
)
from ._base import StreamAssembler
from ._openai_compat import CompatStreamAssembler, OpenAICompatibleAdapter


class _PerplexityStreamAssembler(CompatStreamAssembler):
    """Perplexity streams chat.completion.chunk objects but ends with a
    chat.completion.done object instead of `data: [DONE]` (live-verified
    2026-08-08). Citations and search_results ride on every chunk."""

    def _collect_citations(self, payload: dict[str, Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        seen = {c.url for c in self.citations}
        for entry in payload.get("search_results") or []:
            if isinstance(entry, dict) and entry.get("url") and str(entry["url"]) not in seen:
                citation = Citation(
                    url=str(entry["url"]),
                    title=entry.get("title"),
                    cited_text=entry.get("snippet"),
                )
                seen.add(citation.url)
                self.citations.append(citation)
                events.append(CitationFound(citation=citation))
        for url in payload.get("citations") or []:
            if isinstance(url, str) and url not in seen:
                citation = Citation(url=url)
                seen.add(url)
                self.citations.append(citation)
                events.append(CitationFound(citation=citation))
        return events

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        if data.strip() == "[DONE]":
            return super().feed(event_name, data)
        payload = self._parse_data(data)
        if not isinstance(payload, dict):
            return []
        events = self._chunk_events(payload)
        events.extend(self._collect_citations(payload))
        if payload.get("object") == "chat.completion.done":
            self.saw_terminal = True
            events.append(StreamFinish(finish_reason=self.finish_reason, usage=self.usage))
        return events


class PerplexityAdapter(OpenAICompatibleAdapter):
    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        # The response itself is discarded: reaching a 2xx here proves the
        # credential works, which is all this call can honestly establish.
        if not isinstance(payload, (dict, list)):
            raise KeyCallError(
                "model list response was not JSON",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = [
            Model(
                id=str(entry["id"]),
                provider=self.resolved.provider,
                categories=frozenset(
                    ModelCategory(category) for category in entry.get("categories", [])
                ),
                classification_source="keycall_catalog",
                warnings=("sonar models are not API-discoverable; list is maintained by KeyCall",),
            )
            for entry in self.resolved.catalog_models
        ]
        return models, None

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        minimum = self.resolved.min_max_output_tokens
        if (
            minimum is not None
            and request.max_output_tokens is not None
            and request.max_output_tokens < minimum
        ):
            raise KeyCallError(
                f"{self.resolved.provider} requires max_output_tokens >= {minimum}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation="text_generation",
            )
        return super().build_generation_spec(request)

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        message = ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", ""))
        # Perplexity reports an unusable model as 400, not 404.
        if status_code == 400 and ("Invalid model" in message or "deprecated" in message):
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message
        return super().translate_error(status_code, payload)

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        return _PerplexityStreamAssembler(self.resolved, request)
