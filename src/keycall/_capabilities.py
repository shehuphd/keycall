"""Capability lookups over the maintained provider registry.

The evidence itself lives in the bundled catalog (`_catalog/catalog.json`)
under each provider's `capabilities` block, with the date it was last
verified against the live API. This module is the typed read side: it
answers "can this provider do X" and "does this model restrict sampling",
so a gate and the error message listing the alternatives always read the
same data and can't drift apart.

Adding a capability claim means editing the catalog, not this file, and it
requires confirmed live evidence: an entry here blocks a request before it
reaches the provider, so a wrong one blocks valid calls. When unsure,
leave it out and let the provider's own 400 answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ._registry import (
    ResolvedProvider,
    providers_enforcing_schema,
    providers_with,
    schema_mechanism,
)
from ._types import Message

__all__ = [
    "APPLY_PATCH_PROVIDERS",
    "CODE_INTERPRETER_PROVIDERS",
    "CUSTOM_TOOL_PROVIDERS",
    "JSON_SCHEMA_COMPAT_PROVIDERS",
    "SCHEMA_ENFORCING_PROVIDERS",
    "STREAMING_TRANSCRIPTION_PROVIDERS",
    "TOOL_CALLING_PROVIDERS",
    "TOOL_SEARCH_PROVIDERS",
    "WEB_SEARCH_PROVIDERS",
    "mentions_json",
    "sampling_violation",
]

WEB_SEARCH_PROVIDERS = providers_with("web_search")
APPLY_PATCH_PROVIDERS = providers_with("apply_patch")
CODE_INTERPRETER_PROVIDERS = providers_with("code_interpreter")
CUSTOM_TOOL_PROVIDERS = providers_with("custom_tool")
TOOL_SEARCH_PROVIDERS = providers_with("tool_search")
TOOL_CALLING_PROVIDERS = providers_with("tool_calling")
STREAMING_TRANSCRIPTION_PROVIDERS = providers_with("streaming_transcription")
SCHEMA_ENFORCING_PROVIDERS = providers_enforcing_schema()
# Enforcement mechanism differs within the OpenAI-compatible family:
# "json_schema" providers accept response_format={"type":"json_schema"},
# the rest fall back to json_object (valid JSON, any shape) with a warning.
JSON_SCHEMA_COMPAT_PROVIDERS = frozenset(
    name
    for name in SCHEMA_ENFORCING_PROVIDERS
    if schema_mechanism(name) == "json_schema"
)


def sampling_violation(
    resolved: ResolvedProvider,
    model: str,
    *,
    temperature: float | None,
    top_p: float | None,
) -> str | None:
    """The reason this model won't accept these sampling parameters, or
    None. Some families reject an explicit value entirely; others pin it to
    one permitted value (every Moonshot kimi model takes temperature=1.0
    and top_p=0.95 and 400s on anything else, verified 2026-08-09). Naming
    the permitted value turns a dead end into a one-line fix."""
    lowered = model.lower()
    for constraint in resolved.capabilities.sampling_constraints:
        if not re.match(constraint.pattern, lowered):
            continue
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if value is None or constraint.accepts(name, value):
                continue
            permitted = constraint.permitted(name)
            if permitted is None:
                return (
                    f"model {model!r} does not accept an explicit {name}; "
                    "remove it and the model's own default applies"
                )
            return (
                f"model {model!r} accepts only {name}={permitted:g}, not {value:g}"
            )
    return None


def mentions_json(messages: Sequence[Message]) -> bool:
    """Whether the literal word 'json' (any case) appears anywhere in a
    request's messages. DeepSeek hard-requires this for its json_object
    response_format and 400s otherwise (live-verified 2026-08-06); OpenAI's
    own docs recommend the same for json_object mode generally. Shared
    between the compat adapter (decides whether to inject a JSON
    instruction) and the client (decides whether to warn that it did).
    """
    for message in messages:
        for part in message.content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and "json" in text.lower():
                return True
    return False
