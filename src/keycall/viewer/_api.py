"""Pure request handlers: Registry + JSON-serializable input -> JSON-
serializable output. No HTTP here, so these are testable without a socket.

Every function's return value is what actually reaches the browser. This is
the second-to-last place credentials could leak (the last is `_server.py`'s
response encoder); nothing here ever touches `Target.key` or
`Credential.reveal()`.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
from collections.abc import Iterator
from typing import Any

from .._enums import ModelCategory
from .._errors import KeyCallError
from .._registry import providers_with, resolve_provider, supported_providers
from .._sources import SourceError, _target_from_mapping, load_targets
from .._types import (
    AudioInput,
    FileInput,
    ImageInput,
    ImageOutput,
    Message,
    MessageRole,
    TextGenerationRequest,
    TextInput,
    Tool,
    ToolCall,
    ToolResult,
    VideoOutput,
)
from .._verify_core import DEFAULT_ATTEMPTS, order_candidates, run_verify
from ._registry import MAX_READ_TIMEOUT, MIN_READ_TIMEOUT, Registry

__all__ = [
    "add_key",
    "add_source",
    "browse_models",
    "check_target",
    "clear_conversations",
    "error_body",
    "generate",
    "generate_image",
    "generate_stream_events",
    "generate_video",
    "get_conversation",
    "list_conversations",
    "list_targets",
    "save_conversation",
    "set_settings",
    "verify_target",
]

# The viewer is a local single-user tool where a route blocking one request
# thread for a while is fine; render times observed live range from 10s to
# over 11 minutes, so this is generous rather than tight. This is a separate
# budget from `registry.read_timeout` (each individual start/check/download
# call within the poll loop still respects that, much shorter, setting).
VIDEO_JOB_TIMEOUT = 900.0


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
    return {
        "targets": [dataclasses.asdict(view) for view in registry.views()],
        # Which providers take each attachment kind at all, loaded or not,
        # so the Playground can name a provider that would work when none
        # of the loaded keys can. Read from the catalog the gates read, so
        # the suggestion can't drift from what the adapters enforce.
        "providers_accepting": {
            kind: sorted(providers_with(f"{kind}_input"))
            for kind in ("image", "audio", "file")
        },
        # Per-provider capability flags, read from the same catalog the
        # adapters gate on, so every Playground control can gate itself on
        # a key switch instead of failing after a billable round trip.
        "provider_capabilities": {
            name: {
                "web_search": caps.web_search,
                "tool_calling": caps.tool_calling,
                "image_generation": caps.image_generation,
                "video_generation": caps.video_generation,
                "reasoning_effort": caps.reasoning_effort,
                "prompt_caching": caps.prompt_caching,
                "realtime": caps.realtime,
                # Keyed by the model category name rather than the
                # capability flag's, so the page can use one string for
                # both the provider gate and the model-list filter, the
                # same way "realtime" already doubles for voice.
                "transcription": caps.streaming_transcription,
            }
            for name in supported_providers()
            for caps in (resolve_provider(name).capabilities,)
        },
        # Every provider a key can be added for, straight from the catalog,
        # so the form's dropdown can't drift from what the library accepts.
        "providers": list(supported_providers()),
        # The current provider read timeout, so the settings control shows
        # what the server is holding rather than assuming its own default.
        "read_timeout": registry.read_timeout,
    }


def set_settings(registry: Registry, body: dict[str, Any]) -> dict[str, Any]:
    """Apply viewer settings. Currently one: the provider read timeout,
    an integer number of seconds within the accepted range."""
    seconds = body.get("read_timeout")
    if not isinstance(seconds, int) or isinstance(seconds, bool) or not (
        MIN_READ_TIMEOUT <= seconds <= MAX_READ_TIMEOUT
    ):
        return {
            "error": {
                "code": "bad_request",
                "message": (
                    "read_timeout must be an integer number of seconds from "
                    f"{MIN_READ_TIMEOUT} to {MAX_READ_TIMEOUT}"
                ),
            }
        }
    registry.set_read_timeout(seconds)
    return {"read_timeout": seconds}


def _conversation_summary(conversation: Any) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "mode": conversation.mode,
        "target": conversation.target_id,
        "model": conversation.model_id,
        "updated_at": conversation.updated_at,
    }


def _conversation_full(conversation: Any) -> dict[str, Any]:
    return {
        **_conversation_summary(conversation),
        "history": conversation.history,
        "transcript_html": conversation.transcript_html,
    }


def list_conversations(registry: Registry) -> dict[str, Any]:
    """Metadata only, newest first: enough for the Playground's history
    list without shipping every saved transcript on every page load."""
    return {"conversations": [_conversation_summary(c) for c in registry.list_conversations()]}


def get_conversation(registry: Registry, conversation_id: int) -> dict[str, Any]:
    conversation = registry.get_conversation(conversation_id)
    if conversation is None:
        return {"error": {"code": "not_found", "message": "unknown conversation id"}}
    return {"conversation": _conversation_full(conversation)}


def save_conversation(registry: Registry, body: dict[str, Any]) -> dict[str, Any]:
    """Create or overwrite a conversation. The body is what the Playground
    already holds client-side (title, mode, target, model, the replay
    history, and the rendered transcript) plus an optional id from a
    previous save of the same conversation, so repeated saves as a chat
    grows update one slot instead of piling up copies."""
    conversation_id = body.get("id")
    if conversation_id is not None and not isinstance(conversation_id, int):
        return {"error": {"code": "bad_request", "message": "id must be an integer"}}
    title = body.get("title")
    mode = body.get("mode")
    history = body.get("history")
    transcript_html = body.get("transcript_html")
    if (
        not isinstance(title, str)
        or not isinstance(mode, str)
        or not isinstance(history, list)
        or not isinstance(transcript_html, str)
    ):
        return {
            "error": {
                "code": "bad_request",
                "message": "title, mode, history, and transcript_html are required",
            }
        }
    target_id = body.get("target")
    if target_id is not None and not isinstance(target_id, int):
        return {"error": {"code": "bad_request", "message": "target must be an integer or null"}}
    model_id = body.get("model")
    if model_id is not None and not isinstance(model_id, str):
        return {"error": {"code": "bad_request", "message": "model must be a string or null"}}
    conversation = registry.save_conversation(
        id=conversation_id,
        title=title,
        mode=mode,
        target_id=target_id,
        model_id=model_id,
        history=history,
        transcript_html=transcript_html,
    )
    return {"conversation": _conversation_full(conversation)}


def clear_conversations(registry: Registry) -> dict[str, Any]:
    registry.clear_conversations()
    return {"cleared": True}


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
                if discovery is None:
                    # check_target stores the listing, but the entry carries
                    # the same 5-minute TTL the library cache uses and can
                    # expire between the write and this read. Rare, and a
                    # retry fixes it; reading `.models` off None here would
                    # have been a 500 with no explanation.
                    return {
                        "error": {
                            "code": "cache_expired",
                            "message": "the model list expired while loading; try again",
                        }
                    }

    if category is None:
        return _discovery_dict(discovery)
    try:
        wanted = ModelCategory(category)
    except ValueError:
        return {"error": {"code": "bad_request", "message": f"unknown category {category!r}"}}
    # Same order the verify walk uses: newest first where the provider dates
    # its models, maintained aliases first where it doesn't. Without this
    # the Playground's model picker defaulted to whatever the provider
    # happened to list first, which on Gemini is a model the walk already
    # knows to skip — so the first generation a person tried could fail on a
    # key that works perfectly.
    filtered = order_candidates([m for m in discovery.models if wanted in m.categories])
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


# The three attachment kinds the Playground can send, keyed by the field
# name the browser posts. Audio and documents follow exactly the path
# images already took: the browser holds the bytes and sends base64,
# KeyCall decodes here, and the adapters see the same input any library
# caller would construct. Nothing about the refusal rules is re-implemented
# for the viewer, so a provider that rejects a sound file rejects it here
# for the same reason and with the same message.
_MEDIA_KINDS: dict[str, tuple[str, type[AudioInput | FileInput | ImageInput]]] = {
    "images": ("image", ImageInput),
    "audio": ("audio", AudioInput),
    "files": ("file", FileInput),
}


def _parse_media(
    raw: Any, field: str
) -> list[AudioInput | FileInput | ImageInput]:
    """Attachments the browser sent: base64 for a picked file, or a URL."""
    noun, part_type = _MEDIA_KINDS[field]
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise _BadRequest(f"{field} must be a JSON array")
    parts: list[AudioInput | FileInput | ImageInput] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise _BadRequest(f"{noun} {index} must be an object")
        url = entry.get("url")
        encoded = entry.get("data_base64")
        if bool(url) == bool(encoded):
            raise _BadRequest(f"{noun} {index} needs exactly one of url or data_base64")
        try:
            if url:
                parts.append(part_type(url=str(url)))
            else:
                decoded = base64.b64decode(str(encoded), validate=True)
                # Only a document carries a filename; providers show it to
                # the model, so a picked file keeps the name it had on disk.
                extra = (
                    {"filename": entry["filename"]}
                    if part_type is FileInput and entry.get("filename")
                    else {}
                )
                parts.append(
                    part_type(
                        data=decoded,
                        media_type=entry.get("media_type"),
                        **extra,
                    )
                )
        except (binascii.Error, TypeError, ValueError) as error:
            raise _BadRequest(f"{noun} {index}: {error}") from None
    return parts


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
    attachments: list[Any] = []
    for field in _MEDIA_KINDS:
        attachments.extend(_parse_media(body.get(field), field))
    # A continuation replays the whole conversation and an attachment can
    # carry a turn on its own, so the prompt is required only when neither
    # is present.
    if not history and not attachments and (not prompt or not isinstance(prompt, str)):
        return None
    messages = []
    system = body.get("system")
    if system:
        system_part = TextInput(text=str(system), cacheable=bool(body.get("cache_system", False)))
        messages.append(Message(role="system", content=[system_part]))
    # Prior turns first, then the turn being asked now: the conversation
    # must reach the provider in the order it happened.
    messages.extend(history)
    user_parts: list[Any] = []
    if prompt and isinstance(prompt, str):
        user_parts.append(TextInput(text=prompt))
    user_parts.extend(attachments)
    if user_parts:
        messages.append(Message(role="user", content=user_parts))
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
        "reasoning_effort": body.get("reasoning_effort") or None,
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


def generate_image(registry: Registry, target_id: int, body: dict[str, Any]) -> dict[str, Any]:
    """Image generation is its own operation, not a flag on text: the
    request shape, the result parts, and the models are all different."""
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}

    model = body.get("model")
    prompt = body.get("prompt")
    if not model or not isinstance(model, str) or not prompt or not isinstance(prompt, str):
        return {"error": {"code": "bad_request", "message": "model and prompt are required"}}

    try:
        result = client.generate_image(model=model, prompt=prompt)
    except KeyCallError as error:
        return error_body(error)

    body_out = _result_dict(result)
    body_out["images"] = [
        {"base64_data": part.base64_data, "media_type": part.media_type}
        for part in result.parts
        if isinstance(part, ImageOutput)
    ]
    return body_out


def generate_video(registry: Registry, target_id: int, body: dict[str, Any]) -> dict[str, Any]:
    """Video generation is job-based under the hood (start/poll/download),
    but `client.generate_video()` already collapses that into one blocking
    call, so this route stays a single round trip from the browser's view,
    same shape as `generate_image`."""
    try:
        client = registry.client(target_id)
    except KeyError:
        return {"error": {"code": "not_found", "message": "unknown target id"}}

    model = body.get("model")
    prompt = body.get("prompt")
    if not model or not isinstance(model, str) or not prompt or not isinstance(prompt, str):
        return {"error": {"code": "bad_request", "message": "model and prompt are required"}}

    duration_seconds = body.get("duration_seconds")
    if duration_seconds is not None and not isinstance(duration_seconds, int):
        return {"error": {"code": "bad_request", "message": "duration_seconds must be an integer"}}

    try:
        result = client.generate_video(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            timeout=VIDEO_JOB_TIMEOUT,
        )
    except KeyCallError as error:
        return error_body(error)

    body_out = _result_dict(result)
    body_out["videos"] = [
        {"base64_data": part.base64_data, "media_type": part.media_type, "url": part.url}
        for part in result.parts
        if isinstance(part, VideoOutput)
    ]
    return body_out


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
                elif event.kind == "reasoning_delta":
                    # The page shows progress, not the trace itself, so
                    # only the size is sent: reasoning can run to
                    # thousands of characters before the first answer
                    # token, and the length is what proves liveness.
                    yield {"kind": "reasoning_delta", "chars": len(event.text)}
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
                elif event.kind == "unknown":
                    # Bounded provider kind only, never a payload. This is
                    # how the page learns a server-side search is running:
                    # those frames are activity, not answer content, and
                    # without them a 30-second search reads as a hang.
                    yield {"kind": "provider_event", "what": event.provider_kind[:60]}
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


def add_key(registry: Registry, body: dict[str, Any]) -> dict[str, Any]:
    """Load one target from a key typed into the viewer.

    The key reaches this process over loopback and stays in memory for the
    life of the run, the same way a key read from a file does: nothing is
    written to disk, and the response is the ordinary target list, which
    carries no credential. `keycall view` already accepted a pasted key at
    the terminal prompt; the browser had no way in, so anyone holding a key
    had to write a TOML file before they could click anything.
    """
    provider = body.get("provider")
    key = body.get("key")
    if not isinstance(provider, str) or not provider.strip():
        return {"error": {"code": "bad_request", "message": "a provider is required"}}
    if not isinstance(key, str) or not key.strip():
        return {"error": {"code": "bad_request", "message": "a key is required"}}
    fields = {"provider": provider, "key": key}
    for optional in ("name", "protocol", "base_url"):
        value = body.get(optional)
        if isinstance(value, str) and value.strip():
            fields[optional] = value.strip()
    try:
        # The same constructor the file parsers use, so a pasted target gets
        # the same field validation rather than a second, looser path.
        target = _target_from_mapping(fields, where="the key you entered")
    except SourceError as error:
        return {"error": {"code": "bad_source", "message": str(error)}}
    try:
        registry.add_targets([target])
    except KeyCallError as error:
        return error_body(error)
    return list_targets(registry)


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
