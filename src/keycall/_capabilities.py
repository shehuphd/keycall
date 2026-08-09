"""Maintained per-model capability evidence.

Small and deliberately conservative: an entry here blocks a request before
it reaches the provider, so entries require confirmed evidence that the
provider rejects the parameter. Wrong entries block valid calls — when
unsure, leave the model out and let the provider's own 400 answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ._types import Message

__all__ = [
    "JSON_SCHEMA_COMPAT_PROVIDERS",
    "SCHEMA_ENFORCING_PROVIDERS",
    "mentions_json",
    "rejects_sampling_params",
]

# Model families whose APIs reject temperature/top_p outright.
# OpenAI reasoning families are documented: o-series and gpt-5 accept only
# the default temperature and error on an explicit value.
_SAMPLING_REJECTING = (
    re.compile(r"^o\d"),        # o1, o3, o4-mini, ...
    re.compile(r"^gpt-5"),      # gpt-5 family (Responses API)
    # Anthropic deprecated temperature/top_p/top_k for newer reasoning and
    # Opus models: sending them returns 400. Claude Opus 4.7+ and the newer
    # Sonnet generations are affected; behavior is now controlled by prompting.
    re.compile(r"^claude-opus-4-[7-9]"),
    re.compile(r"^claude-opus-[5-9]"),
    re.compile(r"^claude-sonnet-[5-9]"),
    # Gemini is deliberately absent. Google's changelog announced
    # temperature/top_p/top_k as deprecated on its latest models on
    # 2026-07-21, but deprecation there is an announcement, not a
    # rejection: gemini-3.6-flash, 3.5-flash, 3.1-flash-lite, and
    # gemini-flash-latest all still accept both parameters (verified
    # 2026-08-09). Gate on what a provider rejects, never on what it says
    # it plans to stop supporting — the wrong gate breaks working calls.
)


def rejects_sampling_params(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(pattern.match(lowered) for pattern in _SAMPLING_REJECTING)


# Web search support is an adapter property, not a per-model one: the
# provider either exposes a native server-side search tool on its
# generation surface or it doesn't. Live-verified 2026-08-05: OpenAI
# (web_search tool, Responses), Anthropic (web_search_20250305, Messages),
# Gemini (google_search tool, generateContent), Perplexity (Sonar always
# searches). DeepSeek and Moonshot document no equivalent; custom
# OpenAI-compatible targets can't be assumed to have one (registry research
# section 9: capabilities must be declared or verified, never inferred from
# the protocol label).
WEB_SEARCH_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "perplexity"})

# Structured-output (response_schema) enforcement, per provider, not per
# protocol. OpenAI, Anthropic, and Gemini each have their own dedicated
# adapter with a bespoke enforced mechanism (json_schema text.format,
# forced tool_choice, and responseSchema respectively) and always enforce.
# Within the OpenAI-compatible protocol family, capability genuinely
# differs by provider — live-verified 2026-08-06: Moonshot and Perplexity
# both accept response_format={"type":"json_schema",...} and return
# schema-conformant JSON; DeepSeek returns 400 "This response_format type
# is unavailable now" for the identical request and only supports the
# generic response_format={"type":"json_object"} (valid JSON, any shape).
# A provider absent from this set — DeepSeek, or any custom
# OpenAI-compatible target we've never tested — falls back to json_object
# with a result warning that the schema was not enforced, rather than
# assuming it works and finding out from a live 400.
JSON_SCHEMA_COMPAT_PROVIDERS = frozenset({"moonshot", "perplexity"})

# Tool calling, live-verified 2026-08-08 with a full definition -> call ->
# result -> answer round per provider. Perplexity's Sonar rejects tools
# outright ("Tool calling is not supported for this model", HTTP 400), so
# it is gated before any network call; the live suite carries a drift probe
# that fails if that ever changes. Custom OpenAI-compatible targets are not
# listed: they pass through unverified with a result warning, because the
# tools field is protocol-standard and endpoints like vLLM commonly
# support it.
TOOL_CALLING_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "deepseek", "moonshot"})
SCHEMA_ENFORCING_PROVIDERS = frozenset({"openai", "anthropic", "gemini"}) | JSON_SCHEMA_COMPAT_PROVIDERS


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
