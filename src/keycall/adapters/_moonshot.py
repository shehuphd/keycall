"""Moonshot (Kimi) adapter.

Chat, streaming, and tool calling ride the openai-compatible protocol
unchanged. The override exists for web search, which Moonshot serves as
the ``$web_search`` builtin function (verified live 2026-08-14 on
kimi-k2.6): the request declares the builtin as a tool, the search runs
server-side, and the model answers ``finish_reason: "tool_calls"`` with
a ``$web_search`` call whose arguments carry a search-result handle. The
caller must echo that call back — assistant turn plus a tool message
whose content is the arguments verbatim — before the model will answer
from the results. This adapter builds those echo turns; the client runs
the bounded round loop, so ``web_search=True`` behaves like every other
provider from the caller's seat. Moonshot returns no structured
citations, so ``result.citations`` stays empty on this provider.
"""

from __future__ import annotations

import json
from typing import Any

from .._transport import RequestSpec
from .._types import (
    InvocationResult,
    Message,
    TextGenerationRequest,
    ToolCall,
    ToolResult,
)
from ._openai_compat import OpenAICompatibleAdapter

_BUILTIN_WEB_SEARCH = "$web_search"


class MoonshotAdapter(OpenAICompatibleAdapter):
    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        spec = super().build_generation_spec(request)
        if not request.web_search:
            return spec
        body = dict(spec.json_body or {})
        tools = list(body.get("tools", []))
        tools.append({"type": "builtin_function", "function": {"name": _BUILTIN_WEB_SEARCH}})
        body["tools"] = tools
        return RequestSpec(
            method=spec.method, path=spec.path, params=spec.params, json_body=body
        )

    def server_tool_continuation(
        self, request: TextGenerationRequest, result: InvocationResult
    ) -> TextGenerationRequest | None:
        calls = [
            part
            for part in result.parts
            if isinstance(part, ToolCall) and part.name == _BUILTIN_WEB_SEARCH
        ]
        if not calls:
            return None
        echoes: list[Any] = [
            ToolResult(
                tool_call_id=call.id,
                name=call.name,
                # The arguments carry the provider's search-result handle;
                # they go back as the tool's "output", verbatim.
                content=json.dumps(dict(call.arguments)),
            )
            for call in calls
        ]
        import dataclasses

        return dataclasses.replace(
            request,
            messages=[
                *request.messages,
                result.to_assistant_message(),
                Message(role="user", content=echoes),
            ],
        )
