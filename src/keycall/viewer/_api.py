"""Pure request handlers: Registry + JSON-serializable input -> JSON-
serializable output. No HTTP here, so these are testable without a socket.

Every function's return value is what actually reaches the browser. This is
the second-to-last place credentials could leak (the last is `_server.py`'s
response encoder); nothing here ever touches `Target.key` or
`Credential.reveal()`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

from .._enums import ModelCategory
from .._errors import KeyCallError
from .._sources import SourceError, load_targets
from .._types import (
    Message,
    MessageRole,
    TextGenerationRequest,
    TextInput,
    Tool,
    ToolCall,
    ToolResult,
)
from .._verify_core import DEFAULT_ATTEMPTS, run_verify
from ._registry import Registry

__all__ = [
    "add_source",
    "browse_models",
    "check_target",
    "error_body",
    "generate",
    "generate_stream_events",
    "list_targets",
    "verify_target",
]


def error_body(error: KeyCallError) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code.value,
            "message": error.message,
            "retryable": error.retryable,
            "retry_after": error.retry_after,
        }
    }


def list_targets(registry: Registry) -> dict[str, Any]:
    return {"targets": [dataclasses.asdict(view) for view in registry.views()]}


def _model_dict(model: Any) -> dict[str, Any]:
    return {
        "id": model.id,
        "provider": model.provider,
        "display_name": model.display_name,
        "categories": sorted(c.value for c in model.categories),
        "context_limit": model.context_limit,
        "classification_source": model.classification_source,
        "warnings": list(model.warnings),
    }


def _discovery_dict(discovery: Any) -> dict[str, Any]:
    return {
        "provider": discovery.provider,
        "models": [_model_dict(m) for m in discovery.models],
        "categories": sorted(c.value for c in discovery.categories),
        "from_cache": discovery.from_cache,
        "catalog_version": discovery.catalog_version,
        "warnings": list(discovery.warnings),
    }


def check_target(registry: Registry, target_id: int) -> dict[str, Any]:
    """Live list_models() call across every category, cached for browse_models."""
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}
    try:
        discovery = client.list_models(categories=set(ModelCategory), refresh=True)
    except KeyCallError as error:
        return error_body(error)
    registry.set_cached_discovery(target_id, discovery)
    return _discovery_dict(discovery)


def browse_models(
    registry: Registry, target_id: int, *, category: str | None, refresh: bool
) -> dict[str, Any]:
    discovery = None if refresh else registry.cached_discovery(target_id)
    if discovery is None:
        # Single-flight: concurrent callers (dashboard + playground booting
        # together) share one upstream fetch instead of racing duplicates.
        try:
            lock = registry.fetch_lock(target_id)
        except KeyError:
            return {"error": {"code": "not_found", "message": "unknown target id"}}
        with lock:
            discovery = None if refresh else registry.cached_discovery(target_id)
            if discovery is None:
                checked = check_target(registry, target_id)
                if "error" in checked:
                    return checked
                discovery = registry.cached_discovery(target_id)

    if category is None:
        return _discovery_dict(discovery)
    try:
        wanted = ModelCategory(category)
    except ValueError:
        return {"error": {"code": "bad_request", "message": f"unknown category {category!r}"}}
    filtered = [m for m in discovery.models if wanted in m.categories]
    body = _discovery_dict(discovery)
    body["models"] = [_model_dict(m) for m in filtered]
    body["categories"] = [wanted.value]
    return body


_ROLES: tuple[MessageRole, ...] = ("system", "user", "assistant")


class _BadRequest(Exception):
    """A malformed browser payload. Carries the message the user sees, so
    a typo in a tool schema reads as a fixable mistake rather than a
    generic failure."""


def _parse_tools(raw: Any) -> list[Tool]:
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise _BadRequest("tools must be a JSON array of tool definitions")
    tools = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise _BadRequest(f"tool {index} must be an object")
        try:
            tools.append(
                Tool(
                    name=str(entry["name"]),
                    description=str(entry.get("description", "")),
                    input_schema=entry["input_schema"],
                )
            )
        except KeyError as missing:
            raise _BadRequest(f"tool {index} is missing {missing}") from None
        except (TypeError, ValueError) as error:
            raise _BadRequest(f"tool {index}: {error}") from None
    return tools


def _parse_history(raw: Any) -> list[Message]:
    """Rebuild prior turns the browser is replaying. KeyCall never runs the
    tool loop, so the Playground holds the conversation and sends it back;
    ToolCall.opaque must survive that round trip untouched or providers
    that require it (Gemini's thought signature) reject the next turn."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise _BadRequest("history must be a JSON array of messages")
    messages: list[Message] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("parts"), list):
            raise _BadRequest(f"history[{index}] must be an object with a parts array")
        parts: list[Any] = []
        for part in entry["parts"]:
            if not isinstance(part, dict):
                raise _BadRequest(f"history[{index}] has a non-object part")
            kind = part.get("kind")
            if kind == "text":
                parts.append(TextInput(text=str(part.get("text", ""))))
            elif kind == "tool_call":
                arguments = part.get("arguments")
                parts.append(
                    ToolCall(
                        id=str(part.get("id", "")),
                        name=str(part.get("name", "")),
                        arguments=arguments if isinstance(arguments, dict) else {},
                        opaque=part.get("opaque"),
                    )
                )
            elif kind == "tool_result":
                parts.append(
                    ToolResult(
                        tool_call_id=str(part.get("tool_call_id", "")),
                        name=str(part.get("name", "")),
                        content=part.get("content", ""),
                    )
                )
            else:
                raise _BadRequest(f"history[{index}] has an unsupported part kind {kind!r}")
        requested = str(entry.get("role", ""))
        role = next((valid for valid in _ROLES if valid == requested), None)
        if role is None:
            raise _BadRequest(f"history[{index}] has an unsupported role {requested!r}")
        try:
            messages.append(Message(role=role, content=parts))
        except (TypeError, ValueError) as error:
            raise _BadRequest(f"history[{index}]: {error}") from None
    return messages


def _generation_fields(body: dict[str, Any]) -> dict[str, Any] | None:
    """Shared request fields for the streamed and non-streamed generate
    paths, or None when the body is unusable."""
    model = body.get("model")
    prompt = body.get("prompt")
    if not model or not isinstance(model, str):
        return None
    history = _parse_history(body.get("history"))
    # A continuation replays the whole conversation, so the prompt is only
    # required when there is no history to continue.
    if not history and (not prompt or not isinstance(prompt, str)):
        return None
    messages = []
    system = body.get("system")
    if system:
        messages.append(Message(role="system", content=[TextInput(text=str(system))]))
    if prompt and isinstance(prompt, str):
        messages.append(Message(role="user", content=[TextInput(text=prompt)]))
    messages.extend(history)
    tool_choice = body.get("tool_choice") or None
    return {
        "model": model,
        "messages": messages,
        "max_output_tokens": body.get("max_output_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "web_search": bool(body.get("web_search", False)),
        "tools": _parse_tools(body.get("tools")),
        "tool_choice": tool_choice,
    }


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "model": result.model,
        "operation": result.operation.value,
        "text": result.text,
        "usage": dataclasses.asdict(result.usage),
        "round_trip_duration_ms": result.round_trip_duration_ms,
        "provider_request_id": result.provider_request_id,
        "finish_reason": result.finish_reason,
        "citations": [dataclasses.asdict(c) for c in result.citations],
        "warnings": list(result.warnings),
        "tool_calls": [_tool_call_dict(call) for call in result.tool_calls],
    }


def _tool_call_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "kind": "tool_call",
        "id": call.id,
        "name": call.name,
        "arguments": dict(call.arguments),
        # Provider echo data, passed back verbatim on the next turn.
        "opaque": call.opaque,
    }


def generate(registry: Registry, target_id: int, body: dict[str, Any]) -> dict[str, Any]:
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}

    try:
        fields = _generation_fields(body)
    except _BadRequest as error:
        return {"error": {"code": "bad_request", "message": str(error)}}
    if fields is None:
        return {"error": {"code": "bad_request", "message": "model and prompt are required"}}

    try:
        request = TextGenerationRequest(**fields)
    except (ValueError, TypeError) as error:
        return {"error": {"code": "bad_request", "message": str(error)}}

    try:
        result = client.invoke(request)
    except KeyCallError as error:
        return error_body(error)

    return _result_dict(result)


def generate_stream_events(
    registry: Registry, target_id: int, body: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Streamed generation as JSON-serializable events, ending with either
    a full `result` event (same shape as generate()) or an error event.
    Every failure surfaces as an event so the browser never hangs on a
    silently dropped stream."""
    try:
        client = registry.client(target_id)
    except KeyError:
        yield {"error": {"code": "not_found", "message": "unknown target id"}}
        return

    try:
        fields = _generation_fields(body)
    except _BadRequest as error:
        yield {"error": {"code": "bad_request", "message": str(error)}}
        return
    if fields is None:
        yield {"error": {"code": "bad_request", "message": "model and prompt are required"}}
        return

    try:
        with client.stream_text(**fields) as stream:
            for event in stream:
                if event.kind == "text_delta":
                    yield {"kind": "text_delta", "text": event.text}
                elif event.kind == "stream_start":
                    yield {"kind": "stream_start", "model": event.model}
                elif event.kind == "citation":
                    yield {"kind": "citation", "citation": dataclasses.asdict(event.citation)}
                elif event.kind == "tool_call_started":
                    yield {"kind": "tool_call_started", "id": event.id, "name": event.name}
                elif event.kind == "tool_call_complete":
                    yield {
                        "kind": "tool_call_complete",
                        "tool_call": _tool_call_dict(event.tool_call),
                    }
                # stream_finish carries nothing the final result event lacks.
            result = stream.result()
        yield {"kind": "result", **_result_dict(result)}
    except (ValueError, TypeError) as error:
        yield {"error": {"code": "bad_request", "message": str(error)}}
    except KeyCallError as error:
        yield error_body(error)


def _attempt_dict(attempt: Any) -> dict[str, Any]:
    return dataclasses.asdict(attempt)


def verify_target(
    registry: Registry, target_id: int, *, generate: bool, attempts: int = DEFAULT_ATTEMPTS
) -> dict[str, Any]:
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}

    target = registry.target(target_id)
    result = run_verify(target, generate=generate, attempts=attempts, client=client)
    body = dataclasses.asdict(result)
    body["attempts"] = [_attempt_dict(a) for a in result.attempts]
    body["target_id"] = target_id
    return body


def add_source(registry: Registry, body: dict[str, Any]) -> dict[str, Any]:
    """Load a key file by filesystem path, server-side. The browser only
    ever sends the path; the keys are read and held in this process."""
    path = body.get("path")
    if not path or not isinstance(path, str) or path.strip() in ("", "-"):
        return {"error": {"code": "bad_request", "message": "a file path is required"}}
    try:
        targets, warnings = load_targets(path.strip())
    except SourceError as error:
        return {"error": {"code": "bad_source", "message": str(error)}}
    try:
        registry.add_targets(targets)
    except KeyCallError as error:
        return error_body(error)
    result = list_targets(registry)
    result["warnings"] = [w.message for w in warnings]
    return result
