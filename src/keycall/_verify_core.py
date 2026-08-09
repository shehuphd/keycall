"""Structured core of `keycall verify`: walks filtered text models in
provider order, reporting every attempt until one succeeds or the budget
runs out: reported fallthrough, never silent.

Returns data, not printable strings — `_cli.py` renders this to the
terminal, the viewer renders it to JSON/SSE. One walk, two presentations.
"""

from __future__ import annotations

import hashlib
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

# Bumped whenever the candidate-selection procedure changes, so an old
# report can be read against the rule that produced it. "1" selected the
# first filtered candidate and made exactly one attempt; "2" is the
# bounded, fully-reported walk; "3" adds maintained-alias-first ordering
# within that walk.
SELECTION_RULE_VERSION = "3"


def _is_maintained_alias(model_id: str) -> bool:
    """A provider-maintained pointer to a current model rather than a dated
    snapshot — `gemini-flash-latest`, `chatgpt-4o-latest`. Deliberately
    narrow: only the explicit suffix counts, because a false positive here
    promotes a model that may not exist and wastes an attempt."""
    return model_id.lower().endswith("-latest")


def _model_list_digest(model_ids: list[str]) -> str:
    """Safe identity for the raw model-list snapshot: a digest over the
    ordered provider model IDs. Two runs against the same advertised
    surface produce the same digest; no credential-derived input."""
    joined = "\n".join(model_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]

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
    # Zero-based index in the provider's raw, unfiltered model list; both
    # positions together make a failure reconstructable.
    raw_position: int
    # Why this model was considered a text-generation candidate:
    # provider_metadata, keycall_rule, keycall_catalog, or a combination.
    classification_source: str
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
    # Identity of the raw model-list snapshot the walk ran against, and the
    # version of the selection procedure that produced these attempts.
    model_list_digest: str | None = None
    selection_rule_version: str = SELECTION_RULE_VERSION
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
            # All categories are requested so each text candidate's position
            # in the raw provider list is known, not just its filtered one.
            discovery = client.list_models(categories=set(ModelCategory), refresh=True)
        except KeyCallError as error:
            return VerifyResult(
                label=label,
                provider=target.provider,
                listed_ok=False,
                list_error_code=error.code.value,
                list_error_message=error.message,
                outcome="list_failed",
            )

        digest = _model_list_digest([model.id for model in discovery.models])
        text_models = [
            (raw_position, model)
            for raw_position, model in enumerate(discovery.models)
            if ModelCategory.TEXT_GENERATION in model.categories
        ]
        # Try the provider's own maintained aliases first. A provider that
        # publishes a "-latest" pointer keeps it aimed at a model that
        # currently works, which is exactly what verification needs; the
        # dated ids around it are the ones that get retired. This is not
        # cosmetic: on 2026-08-09 the first six text models Gemini
        # advertised to a new key were all withdrawn, so a walk in list
        # order spent most of its budget on models Google had already shut
        # down. Order is otherwise the provider's own, and every attempt
        # still reports both its filtered and raw position.
        text_models.sort(key=lambda entry: not _is_maintained_alias(entry[1].id))
        if not generate:
            return VerifyResult(
                label=label,
                provider=client.provider,
                listed_ok=True,
                text_model_count=len(text_models),
                model_list_digest=digest,
                outcome="listed",
            )
        if not text_models:
            return VerifyResult(
                label=label,
                provider=client.provider,
                listed_ok=True,
                text_model_count=0,
                generate_requested=True,
                model_list_digest=digest,
                outcome="no_text_models",
            )

        messages = [Message(role="user", content=[TextInput(text=DEFAULT_GENERATION_PROMPT)])]
        collected: list[ModelAttempt] = []
        rate_limited = False
        for position, (raw_position, candidate) in enumerate(text_models[:attempts]):
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
                        raw_position=raw_position,
                        classification_source=candidate.classification_source,
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
                        model_list_digest=digest,
                        outcome="credential_rejected",
                    )
                if error.code is ErrorCode.RATE_LIMITED:
                    rate_limited = True
                continue

            collected.append(
                ModelAttempt(
                    model_id=candidate.id,
                    position=position,
                    raw_position=raw_position,
                    classification_source=candidate.classification_source,
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
                model_list_digest=digest,
                outcome="generated",
            )

        return VerifyResult(
            label=label,
            provider=client.provider,
            listed_ok=True,
            text_model_count=len(text_models),
            generate_requested=True,
            attempts=tuple(collected),
            model_list_digest=digest,
            outcome="rate_limited_unverified" if rate_limited else "no_model_invocable",
        )
    finally:
        if owns_client:
            client.close()
