"""Structured core of `keycall verify`: walks filtered text models in
provider order, reporting every attempt until one succeeds or the budget
runs out (PRD section 14.2/14.3's "reported fallthrough").

Returns data, not printable strings — `_cli.py` renders this to the
terminal, the viewer renders it to JSON/SSE. One walk, two presentations.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._client import KeyCall
from ._enums import ModelCategory
from ._errors import ErrorCode, KeyCallError
from ._sanitize import safe_display_name
from ._sources import Target
from ._types import Message, TextInput

__all__ = ["ModelAttempt", "VerifyResult", "run_verify"]

DEFAULT_GENERATION_PROMPT = "Reply with the single word: ok"
DEFAULT_GENERATION_MAX_TOKENS = 16
DEFAULT_ATTEMPTS = 8

# The credential itself is the problem: stop immediately, no other model
# will fare better with a key the provider has rejected.
_CREDENTIAL_FAILURES = frozenset({ErrorCode.INVALID_API_KEY, ErrorCode.PERMISSION_DENIED})
# Everything else is model-scoped and worth trying the next candidate for.
# Rate limits included, deliberately: providers meter per model and tier, so
# a 429 on one model says nothing about the next.


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelAttempt:
    model_id: str
    position: int
    ok: bool
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    round_trip_duration_ms: float | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifyResult:
    label: str
    provider: str
    listed_ok: bool
    text_model_count: int | None = None
    list_error_code: str | None = None
    list_error_message: str | None = None
    generate_requested: bool = False
    generate_ok: bool = False
    attempts: tuple[ModelAttempt, ...] = ()
    # "listed" | "generated" | "no_text_models" | "credential_rejected" |
    # "rate_limited_unverified" | "no_model_invocable" | "list_failed"
    outcome: str = "listed"


def run_verify(
    target: Target,
    *,
    generate: bool,
    attempts: int = DEFAULT_ATTEMPTS,
    client: KeyCall | None = None,
) -> VerifyResult:
    """Never raises for provider failures. Pass an already-open `client` to
    reuse a live connection (the viewer does this); otherwise one is opened
    and closed for the duration of the call."""
    label = safe_display_name(target.display_name)
    owns_client = client is None
    if owns_client:
        client = KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        )
    try:
        try:
            # Verification must hit the live provider, never cached data.
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
        except KeyCallError as error:
            return VerifyResult(
                label=label,
                provider=target.provider,
                listed_ok=False,
                list_error_code=error.code.value,
                list_error_message=error.message,
                outcome="list_failed",
            )

        text_models = discovery.models
        if not generate:
            return VerifyResult(
                label=label,
                provider=client.provider,
                listed_ok=True,
                text_model_count=len(text_models),
                outcome="listed",
            )
        if not text_models:
            return VerifyResult(
                label=label,
                provider=client.provider,
                listed_ok=True,
                text_model_count=0,
                generate_requested=True,
                outcome="no_text_models",
            )

        messages = [Message(role="user", content=[TextInput(text=DEFAULT_GENERATION_PROMPT)])]
        collected: list[ModelAttempt] = []
        rate_limited = False
        for position, candidate in enumerate(text_models[:attempts]):
            try:
                result = client.generate_text(
                    model=candidate.id,
                    messages=messages,
                    max_output_tokens=DEFAULT_GENERATION_MAX_TOKENS,
                )
            except KeyCallError as error:
                collected.append(
                    ModelAttempt(
                        model_id=candidate.id,
                        position=position,
                        ok=False,
                        error_code=error.code.value,
                        error_message=error.message,
                        retryable=error.retryable,
                    )
                )
                if error.code in _CREDENTIAL_FAILURES:
                    return VerifyResult(
                        label=label,
                        provider=client.provider,
                        listed_ok=True,
                        text_model_count=len(text_models),
                        generate_requested=True,
                        attempts=tuple(collected),
                        outcome="credential_rejected",
                    )
                if error.code is ErrorCode.RATE_LIMITED:
                    rate_limited = True
                continue

            collected.append(
                ModelAttempt(
                    model_id=candidate.id,
                    position=position,
                    ok=True,
                    round_trip_duration_ms=result.round_trip_duration_ms,
                    total_tokens=result.usage.total_tokens,
                    finish_reason=result.finish_reason,
                )
            )
            return VerifyResult(
                label=label,
                provider=client.provider,
                listed_ok=True,
                text_model_count=len(text_models),
                generate_requested=True,
                generate_ok=True,
                attempts=tuple(collected),
                outcome="generated",
            )

        return VerifyResult(
            label=label,
            provider=client.provider,
            listed_ok=True,
            text_model_count=len(text_models),
            generate_requested=True,
            attempts=tuple(collected),
            outcome="rate_limited_unverified" if rate_limited else "no_model_invocable",
        )
    finally:
        if owns_client:
            client.close()
