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
from .._types import Message, TextGenerationRequest, TextInput
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


def _generation_fields(body: dict[str, Any]) -> dict[str, Any] | None:
    """Shared request fields for the streamed and non-streamed generate
    paths, or None when the body is unusable."""
    model = body.get("model")
    prompt = body.get("prompt")
    if not model or not isinstance(model, str):
        return None
    if not prompt or not isinstance(prompt, str):
        return None
    messages = []
    system = body.get("system")
    if system:
        messages.append(Message(role="system", content=[TextInput(text=str(system))]))
    messages.append(Message(role="user", content=[TextInput(text=prompt)]))
    return {
        "model": model,
        "messages": messages,
        "max_output_tokens": body.get("max_output_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "web_search": bool(body.get("web_search", False)),
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
    }


def generate(registry: Registry, target_id: int, body: dict[str, Any]) -> dict[str, Any]:
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}

    fields = _generation_fields(body)
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

    fields = _generation_fields(body)
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
