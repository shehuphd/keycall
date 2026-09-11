"""KeyCall and AsyncKeyCall clients.

Provider, credential, protocol, and base URL are immutable client identity,
bound once at construction with no setters and no per-call override.
The raw credential is wrapped in the redacting Credential type here, at
its single entry boundary; only the transport layer ever reveals it again.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._live import AsyncLiveSession, LiveSession
    from ._realtime import AsyncRealtimeSession, RealtimeSession
    from ._transcription import AsyncTranscriptionSession, TranscriptionSession

import httpx

from . import _cache, _capabilities, _classify, _tracing
from ._cache import CachedModels
from ._credential import Credential
from ._enums import ModelCategory, Operation, ProviderProtocol
from ._errors import (
    BatchJobTimeout,
    ErrorCode,
    KeyCallError,
    TranscriptionJobTimeout,
    VideoJobTimeout,
)
from ._registry import (
    ResolvedProvider,
    catalog_age_days,
    catalog_is_stale,
    catalog_version,
    resolve_provider,
    retired_model_fact,
    supported_service_providers,
)
from ._transport import AsyncTransport, Transport
from ._types import (
    BatchJob,
    BatchRequest,
    BatchResult,
    EmbeddingRequest,
    ImageGenerationRequest,
    InvocationResult,
    LiveConfig,
    Message,
    Model,
    ModelDiscovery,
    RealtimeConfig,
    ServiceReport,
    ServiceStatus,
    SpeechGenerationRequest,
    StreamEvent,
    TextGenerationRequest,
    Tool,
    TranscriptionConfig,
    TranscriptionJob,
    TranscriptionRequest,
    TranscriptionResult,
    Usage,
    VideoGenerationRequest,
    VideoJob,
    Voice,
    WithheldModel,
)
from .adapters import ProviderAdapter, adapter_for
from .adapters._base import BatchSubmission, InbandStreamError, StreamAssembler

__all__ = ["AsyncKeyCall", "AsyncTextStream", "KeyCall", "TextStream"]


def _catalog_voice_records(resolved: Any) -> tuple[Voice, ...]:
    """Catalog-recorded voice sets (fixed named lists) as Voice records."""
    return tuple(
        Voice(
            provider=resolved.provider,
            id=str(entry["id"]),
            name=str(entry.get("name") or entry["id"]),
            description=entry.get("description"),
            models=tuple(entry["models"]) if entry.get("models") else None,
        )
        for entry in resolved.catalog_voices
    )

_MAX_LIST_PAGES = 10
_DEFAULT_CATEGORIES = frozenset({ModelCategory.TEXT_GENERATION})


def _validate_categories(
    categories: set[ModelCategory] | frozenset[ModelCategory] | None,
) -> frozenset[ModelCategory]:
    if categories is None:
        return _DEFAULT_CATEGORIES
    validated = set()
    for category in categories:
        if not isinstance(category, ModelCategory):
            raise TypeError(
                "categories accepts ModelCategory members only; plain strings are "
                "validated at config/CLI boundaries, not here "
                f"(got {type(category).__name__})"
            )
        validated.add(category)
    if not validated:
        return _DEFAULT_CATEGORIES
    return frozenset(validated)


def _with_schema_warning(
    invocation: InvocationResult, request: TextGenerationRequest, provider: str
) -> InvocationResult:
    """Append a warning when response_schema was requested but this
    provider only guarantees valid JSON, not schema conformance: never
    claim enforcement that isn't delivered — the same posture as the
    unreported-usage and stale-catalog warnings elsewhere."""
    if request.response_schema is None or provider in _capabilities.SCHEMA_ENFORCING_PROVIDERS:
        return invocation
    warnings = list(invocation.warnings)
    warnings.append(
        f"provider {provider!r} does not enforce response_schema; the response "
        "is guaranteed valid JSON but not guaranteed to match the schema — "
        "validate client-side"
    )
    if not _capabilities.mentions_json(request.messages):
        warnings.append(
            "keycall added a 'respond only with JSON' system instruction "
            f"because {provider!r} requires the word 'json' to appear in the "
            "prompt for its JSON response mode"
        )
    return dataclasses.replace(invocation, warnings=tuple(warnings))


# Every provider says "I ran out of output budget" in its own words:
# OpenAI Responses reports incomplete_details.reason, which the adapter
# renders as "incomplete:max_output_tokens"; Anthropic says "max_tokens";
# Gemini "MAX_TOKENS"; the Chat Completions family "length". A caller
# should not have to learn four spellings to notice its answer was cut off.
_TRUNCATION_REASONS = frozenset(
    {"incomplete:max_output_tokens", "max_tokens", "length"}
)


def was_truncated(finish_reason: str | None) -> bool:
    """Whether the provider stopped because the output budget ran out,
    normalized across the four wire protocols. Case-insensitive because
    Gemini shouts its finish reasons and the rest do not."""
    if not finish_reason:
        return False
    return finish_reason.lower() in _TRUNCATION_REASONS


def _with_truncation_warning(invocation: InvocationResult) -> InvocationResult:
    """Say plainly that the answer is incomplete and what to change.

    The finish reason already carries this, but only to a reader who knows
    that provider's vocabulary, and it appears beside timing and token counts
    where it reads as one more statistic. Reasoning models make it easy to
    hit: their hidden reasoning is billed against the same output budget,
    so a small max_output_tokens can be spent before any answer is emitted.
    """
    if not was_truncated(invocation.finish_reason):
        return invocation
    return dataclasses.replace(
        invocation,
        warnings=(
            *invocation.warnings,
            (
                "the reply stopped because max_output_tokens ran out, so it is "
                "cut off mid-answer — raise max_output_tokens and send again. "
                "On a reasoning model the hidden reasoning is charged to the "
                "same budget, so it can be used up before any text appears"
            ),
        ),
    )


def _with_custom_tool_warning(
    invocation: InvocationResult,
    request: TextGenerationRequest,
    provider: str,
    *,
    is_custom: bool,
) -> InvocationResult:
    """Custom targets get tools passed through unverified, so say so on
    every result rather than implying the endpoint honored them."""
    if not request.tools or not is_custom:
        return invocation
    return dataclasses.replace(
        invocation,
        warnings=(
            *invocation.warnings,
            (
                f"tool calling on custom target {provider!r} is unverified; "
                "keycall passes the standard tools field through without "
                "evidence the endpoint honors it"
            ),
        ),
    )


def _filter_models(
    models: tuple[Model, ...], categories: frozenset[ModelCategory]
) -> tuple[Model, ...]:
    # Unknown models never enter a picker unless UNKNOWN was requested.
    return tuple(model for model in models if model.categories & categories)


def _build_discovery(
    *,
    provider: str,
    cached: CachedModels,
    categories: frozenset[ModelCategory],
    from_cache: bool,
) -> ModelDiscovery:
    warnings = list(cached.warnings)
    stale = catalog_is_stale()
    if stale:
        # The bundled catalog carries endpoints, auth schemes, and
        # capability evidence. When it is this old the evidence predates
        # provider changes KeyCall hasn't seen, so say so instead of
        # presenting it as current.
        warnings.append(
            f"keycall's bundled provider catalog was last verified "
            f"{catalog_age_days()} days ago (version {catalog_version()}); "
            "provider endpoints and capabilities may have changed since — "
            "upgrade keycall for the current catalog"
        )
    return ModelDiscovery(
        provider=provider,
        models=_filter_models(cached.models, categories),
        categories=categories,
        fetched_at=cached.fetched_at,
        from_cache=from_cache,
        catalog_version=catalog_version(),
        catalog_stale=stale,
        warnings=tuple(warnings),
        withheld=cached.withheld,
    )


def _require_declared_credential_fields(resolved: ResolvedProvider, credential: Credential) -> None:
    """Refuse a credential whose field set differs from what the catalog
    declares for this provider, naming the missing or unrecognized field
    names (catalog vocabulary, never values) and the constructor shape
    that fixes it — before any network call could fail less legibly."""
    declared = resolved.credential_fields
    supplied = credential.field_names()
    missing = [name for name in declared if name not in supplied]
    unrecognized = [name for name in supplied if name not in declared]
    if not missing and not unrecognized:
        return
    shape = ", ".join(f"'{name}': ..." for name in declared)
    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if unrecognized:
        problems.append("not a field it takes: " + ", ".join(unrecognized))
    raise KeyCallError(
        f"{resolved.provider}'s credential carries {', '.join(declared)} "
        f"({'; '.join(problems)}). Pass credential={{{shape}}}"
        + (" or the api_key shorthand" if declared == ("api_key",) else ""),
        code=ErrorCode.INVALID_API_KEY,
        provider=resolved.provider,
        retryable=False,
    )


class _BaseClient:
    __slots__ = ("_adapter", "_credential", "_resolved", "_transport")

    _resolved: ResolvedProvider
    _credential: Credential | None
    _adapter: ProviderAdapter
    _transport: Any

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        credential: Mapping[str, str] | None = None,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        # One mechanism, one spelling: api_key is the single-key shorthand
        # for credential={"api_key": ...}, so passing both is undefined and
        # refused rather than one silently winning.
        if (api_key is None) == (credential is None):
            raise ValueError(
                "pass api_key for a single-key provider, or credential "
                "with the provider's named secret fields; never both"
            )
        # Wrap the secret fields first so no later failure path ever
        # handles a raw string.
        wrapped = Credential(api_key if api_key is not None else dict(credential or {}))
        resolved = resolve_provider(
            provider,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        _require_declared_credential_fields(resolved, wrapped)
        object.__setattr__(self, "_resolved", resolved)
        object.__setattr__(self, "_credential", wrapped)
        object.__setattr__(self, "_adapter", adapter_for(resolved))
        object.__setattr__(self, "_transport", None)

    # Immutable identity: no attribute may be rebound after construction.
    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError(
            f"{type(self).__name__} identity is immutable; construct a new client instead"
        )

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(
            f"{type(self).__name__} identity is immutable; construct a new client instead"
        )

    @property
    def provider(self) -> str:
        return self._resolved.provider

    @property
    def protocol(self) -> ProviderProtocol:
        return self._resolved.protocol

    @property
    def base_url(self) -> str:
        return self._resolved.base_url

    @property
    def kind(self) -> str:
        """"model" for a provider that lists and invokes models, "service"
        for one validated by live category probes (probe_services())."""
        return self._resolved.kind

    # Deliberately no api_key property.

    @property
    def closed(self) -> bool:
        return self._credential is None

    def _require_open(self) -> Credential:
        credential = self._credential
        if credential is None:
            raise RuntimeError(f"{type(self).__name__} is closed; construct a new client")
        return credential

    def _require_service_kind(self) -> None:
        """probe_services() is the service-provider validation surface; a
        model provider validates by listing, so pointing there beats a
        probe that has no categories to check."""
        if self._resolved.kind == "service":
            return
        raise KeyCallError(
            f"{self.provider} is a model provider; probe_services() checks "
            "the service categories of a service provider ("
            + ", ".join(supported_service_providers())
            + "). Validate this key by listing models: list_models(), or "
            "keycall verify",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.provider,
            operation=Operation.SERVICE_PROBE.value,
        )

    def _service_probe_status(
        self, trace: Any, category: str, outcome: ServiceStatus
    ) -> ServiceStatus:
        trace.event(
            "app",
            operation="service_probe",
            status="ok" if outcome.status == "enabled" else "error",
            result={"category": category, "status": outcome.status},
        )
        return outcome

    def _require_model_not_retired(self, model: str | None) -> None:
        """Refuse a model the catalog records as shut down, before any
        network call: the provider would only answer with a bare not-found
        after a bill-nothing round trip, and this error can name the
        retirement date and the provider's own recommended replacement.
        Matches the id and its recorded alias spellings. Every entry is
        backed by a live release-suite probe, so a stale record fails the
        suite rather than silently blocking a model that came back."""
        if model is None:
            return
        fact = retired_model_fact(self._resolved.retired_models, model)
        if fact is None:
            return
        when = fact.get("retired")
        replacement = fact.get("replacement")
        raise KeyCallError(
            f"{model} was retired by {self.provider}"
            + (f" on {when}" if when else "")
            + (
                f"; the provider recommends {replacement}"
                if replacement
                else "; the provider named no replacement"
            ),
            code=ErrorCode.MODEL_RETIRED,
            provider=self.provider,
            retryable=False,
        )

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"protocol={self.protocol.value!r}, {state})"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("clients hold credentials and cannot be pickled or copied")

    # --- pure logic shared by the sync and async clients; only the awaits
    # --- differ in the public methods below.

    def _cached_discovery(
        self, categories: frozenset[ModelCategory], fingerprint: str, trace: Any
    ) -> ModelDiscovery | None:
        cached = _cache.shared_cache.get(self.provider, self.base_url, fingerprint)
        if cached is None:
            return None
        trace.event("app", operation="cache_hit", status="ok")
        return _build_discovery(
            provider=self.provider, cached=cached, categories=categories, from_cache=True
        )

    def _parse_page(self, trace: Any, spec: Any, result: Any) -> tuple[list[Model], Any]:
        trace.event(
            "http",
            operation=f"{spec.method} {spec.path}",
            target=self.provider,
            status=str(result.status_code),
            duration_ms=result.duration_ms,
        )
        return self._adapter.parse_model_page(result.payload)

    def _store_discovery(
        self,
        models: list[Model],
        *,
        truncated: bool,
        categories: frozenset[ModelCategory],
        fingerprint: str,
        trace: Any,
    ) -> ModelDiscovery:
        warnings: tuple[str, ...] = ()
        if truncated:
            warnings = (
                (
                    f"provider reported more model pages after the "
                    f"{_MAX_LIST_PAGES}-page limit; this list is truncated"
                ),
            )
        # Models the catalog records as retired are withheld: some
        # providers keep shut-down models in their listing while requests
        # to them fail (OpenAI does), so offering one is offering a dead
        # end. Each withheld id is reported both as a warning and as a
        # record, never dropped silently. One site for both clients, before
        # caching, so cached reads carry the same filtered view.
        withheld: tuple[WithheldModel, ...] = ()
        if self._resolved.retired_models:
            kept: list[Model] = []
            for model in models:
                fact = retired_model_fact(self._resolved.retired_models, model.id)
                if fact is None:
                    kept.append(model)
                    continue
                when = fact.get("retired")
                replacement = fact.get("replacement")
                warnings += (
                    f"{model.id} was retired by {self.provider}"
                    + (f" on {when}" if when else "")
                    + (f"; the provider recommends {replacement}" if replacement else "")
                    + "; withheld from this listing",
                )
                # The same fact as data, for a reader that lays these out
                # rather than printing the sentence.
                withheld += (
                    WithheldModel(
                        id=model.id,
                        provider=self.provider,
                        retired_on=when,
                        replacement=replacement,
                    ),
                )
            models = kept
        # One annotation site for both clients, before caching, so cached
        # reads carry the fact too. Only ids matching a recorded convention
        # gain one; everything else keeps alias=None.
        if self._resolved.alias_conventions:
            models = [
                dataclasses.replace(model, alias=_classify.alias_fact(self.provider, model.id))
                for model in models
            ]
        cached = CachedModels(
            models=tuple(models),
            fetched_at=datetime.now(timezone.utc),
            warnings=warnings,
            withheld=withheld,
        )
        _cache.shared_cache.put(self.provider, self.base_url, fingerprint, cached)
        discovery = _build_discovery(
            provider=self.provider, cached=cached, categories=categories, from_cache=False
        )
        trace.event(
            "model",
            operation="normalize",
            status="ok",
            result={"models": len(models), "filtered": len(discovery.models)},
        )
        return discovery

    def _image_spec(self, request: ImageGenerationRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_image_spec, whose
        # default covers every adapter without an implementation.
        self._require_model_not_retired(request.model)
        return self._adapter.build_image_spec(request)

    def _parse_image(
        self, request: ImageGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_image_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        trace.event(
            "model",
            operation="image_generation",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"images": len(invocation.parts)},
        )
        return invocation

    def _speech_spec(self, request: SpeechGenerationRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_speech_spec, whose
        # default covers every adapter without an implementation.
        self._require_model_not_retired(request.model)
        return self._adapter.build_speech_spec(request)

    def _parse_speech(
        self, request: SpeechGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_speech_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        trace.event(
            "model",
            operation="speech_generation",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"clips": len(invocation.parts)},
        )
        return invocation

    def _require_video_job(self, job: VideoJob) -> None:
        if not isinstance(job, VideoJob):
            raise TypeError(f"expected a VideoJob, got {type(job).__name__}")
        if job.provider != self.provider:
            raise KeyCallError(
                f"this job belongs to provider {job.provider!r}; this client is bound "
                f"to {self.provider!r} and its credential must not poll another "
                "provider's job",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.provider,
                operation="video_generation",
            )

    def _require_batch_job(self, job: BatchJob) -> None:
        if not isinstance(job, BatchJob):
            raise TypeError(f"expected a BatchJob, got {type(job).__name__}")
        if job.provider != self.provider:
            raise KeyCallError(
                f"this batch belongs to provider {job.provider!r}; this client is "
                f"bound to {self.provider!r} and its credential must not poll "
                "another provider's batch",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.provider,
                operation=job.operation,
            )

    def _raise_batch_failure(self, job: BatchJob) -> NoReturn:
        detail = job.error_message or "no detail from the provider"
        state = f" ({job.provider_status})" if job.provider_status else ""
        raise KeyCallError(
            f"the batch failed{state}: {detail}",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.provider,
            operation=job.operation,
        )

    def _batch_submission(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        items = tuple(requests)
        if not items:
            raise ValueError("start_batch needs at least one request")
        for request in items:
            if not isinstance(request, BatchRequest):
                raise TypeError(f"expected BatchRequest entries, got {type(request).__name__}")
            self._require_model_not_retired(request.model)
        models = {request.model for request in items}
        if len(models) > 1 and not self._adapter.batch_mixed_models:
            raise KeyCallError(
                f"{self.provider} runs one model per batch, and these requests "
                f"name {len(models)}: {', '.join(sorted(models))}. Split them "
                "into one batch per model — Anthropic's batch lane is the one "
                "that takes mixed models.",
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.provider,
                operation=Operation.BATCH_GENERATION.value,
            )
        wired = []
        for index, request in enumerate(items):
            generation = TextGenerationRequest(
                model=request.model,
                messages=request.messages,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
            wired.append((f"kc-{index}", self._adapter.batch_item_body(generation)))
        return BatchSubmission(
            operation=Operation.BATCH_GENERATION.value,
            model=items[0].model if len(models) == 1 else None,
            items=tuple(wired),
        )

    def _embed_batch_submission(self, model: str, inputs: Sequence[str]) -> BatchSubmission:
        self._require_model_not_retired(model)
        texts = tuple(inputs)
        if not texts:
            raise ValueError("start_embedding_batch needs at least one input")
        return BatchSubmission(
            operation=Operation.BATCH_EMBEDDING.value,
            model=model,
            items=tuple(
                (f"kc-{index}", self._adapter.batch_embed_item_body(model, text))
                for index, text in enumerate(texts)
            ),
        )

    def _assemble_batch_results(
        self, job: BatchJob, outcomes: Sequence[Any]
    ) -> tuple[BatchResult, ...]:
        by_key: dict[str, Any] = {}
        for outcome in outcomes:
            by_key.setdefault(outcome.key, outcome)
        results = []
        for index, key in enumerate(job.request_keys):
            outcome = by_key.get(key)
            if outcome is None:
                results.append(
                    BatchResult(
                        index=index,
                        key=key,
                        error_message="the provider returned no result for this request",
                    )
                )
            elif outcome.body is not None:
                try:
                    invocation = self._adapter.parse_batch_item(outcome.body, key=key, job=job)
                except KeyCallError as error:
                    results.append(
                        BatchResult(
                            index=index,
                            key=key,
                            error_code=error.code.value,
                            error_message=error.message,
                        )
                    )
                else:
                    results.append(BatchResult(index=index, key=key, result=invocation))
            else:
                results.append(
                    BatchResult(
                        index=index,
                        key=key,
                        error_code=outcome.error_code,
                        error_message=outcome.error_message,
                    )
                )
        return tuple(results)

    def _transcription_request(
        self,
        *,
        model: str,
        audio: bytes | None,
        url: str | None,
        media_type: str | None,
        language: str | None,
        diarize: bool,
    ) -> TranscriptionRequest:
        self._require_model_not_retired(model)
        request = TranscriptionRequest(
            model=model,
            data=audio,
            url=url,
            media_type=media_type,
            language=language,
            diarize=diarize,
        )
        self._adapter.validate_transcription_request(request)
        return request

    def _require_transcription_job(self, job: TranscriptionJob) -> None:
        if not isinstance(job, TranscriptionJob):
            raise TypeError(f"expected a TranscriptionJob, got {type(job).__name__}")
        if job.provider != self.provider:
            raise KeyCallError(
                f"this transcription belongs to provider {job.provider!r}; this "
                f"client is bound to {self.provider!r} and its credential must "
                "not poll another provider's job",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.provider,
                operation=Operation.TRANSCRIPTION.value,
            )

    def _raise_transcription_failure(self, job: TranscriptionJob) -> NoReturn:
        detail = job.error_message or "no detail from the provider"
        state = f" ({job.provider_status})" if job.provider_status else ""
        raise KeyCallError(
            f"the transcription failed{state}: {detail}",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.provider,
            operation=Operation.TRANSCRIPTION.value,
        )

    def _refuse_transcription_jobs_here(self) -> NoReturn:
        # The one job-shaped transcription provider today is AssemblyAI;
        # this message names the pattern rather than enumerating, so a
        # second job-shaped provider needs no edit here.
        raise KeyCallError(
            f"provider {self.provider!r} answers transcription in one round "
            "trip, so no job handle exists — call transcribe()",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.provider,
            operation=Operation.TRANSCRIPTION.value,
        )

    def _raise_video_failure(self, job: VideoJob) -> NoReturn:
        detail = job.error_message or "no detail from the provider"
        state = f" ({job.provider_status})" if job.provider_status else ""
        raise KeyCallError(
            f"video generation failed{state}: {detail}",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.provider,
            operation="video_generation",
        )

    def _video_result_from_download(self, job: VideoJob, result: Any) -> InvocationResult:
        import base64 as _b64

        if not isinstance(result.payload, bytes) or not result.payload:
            raise KeyCallError(
                "video download did not return the file's bytes",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.provider,
                operation="video_generation",
            )
        media_type = (
            str(result.headers.get("content-type", "video/mp4")).split(";")[0].strip()
        )
        return self._adapter.video_result(
            base64_data=_b64.b64encode(result.payload).decode("ascii"),
            media_type=media_type,
            url=job.video_url,
            model=job.model,
            round_trip_duration_ms=result.duration_ms,
        )

    def _embedding_spec(self, request: EmbeddingRequest) -> Any:
        # The refusal lives in ProviderAdapter.build_embedding_spec, whose
        # default raises for every adapter that hasn't implemented one.
        # Gating here as well would duplicate the message in a second
        # place that could drift from it.
        self._require_model_not_retired(request.model)
        return self._adapter.build_embedding_spec(request)

    def _parse_embedding(
        self, request: EmbeddingRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_embedding_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
            expected=len(request.inputs),
        )
        trace.event(
            "model",
            operation="embedding",
            target=invocation.model,
            duration_ms=invocation.round_trip_duration_ms,
            result={"inputs": len(request.inputs), "vectors": len(invocation.parts)},
        )
        return invocation

    def _generation_spec(self, request: TextGenerationRequest) -> Any:
        if not isinstance(request, TextGenerationRequest):
            raise KeyCallError(
                f"invoke() accepts typed request objects, got {type(request).__name__}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        self._require_model_not_retired(request.model)
        return self._adapter.build_generation_spec(request)

    def _parse_invocation(
        self, request: TextGenerationRequest, result: Any, trace: Any
    ) -> InvocationResult:
        invocation = self._adapter.parse_generation_response(
            result.payload,
            headers=result.headers,
            round_trip_duration_ms=result.duration_ms,
            model=request.model,
        )
        invocation = _with_schema_warning(invocation, request, self.provider)
        invocation = _with_truncation_warning(invocation)
        invocation = _with_custom_tool_warning(
            invocation, request, self.provider, is_custom=self._resolved.is_custom
        )
        # Token counts ride as event kwargs, not inside result=:
        # TraceAct's sanitiser redacts any result field whose name
        # contains "token", and its cost estimator reads model events'
        # provider/tokens_in/tokens_out kwargs (traceact 1.1.0
        # convention, verified against the installed signature
        # 2026-09-02).
        trace.event(
            "model",
            operation="text_generation",
            target=invocation.model,
            status=invocation.finish_reason or "ok",
            duration_ms=invocation.round_trip_duration_ms,
            provider=self.provider,
            tokens_in=invocation.usage.input_tokens,
            tokens_out=invocation.usage.output_tokens,
            result={"parts": len(invocation.parts)},
        )
        return invocation


# A provider that keeps demanding server-tool echo rounds past this many
# billable calls is broken; refusing beats an unbounded spend.
_SERVER_TOOL_ROUND_BUDGET = 5


def _added(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _merged_usage(carried: Usage, current: Usage) -> Usage:
    """Token spend across the rounds of one logical call (Moonshot's
    web-search echo loop). Sums where both rounds report, keeps the
    reporting side where only one does."""
    return Usage(
        input_tokens=_added(carried.input_tokens, current.input_tokens),
        output_tokens=_added(carried.output_tokens, current.output_tokens),
        cached_input_tokens=_added(carried.cached_input_tokens, current.cached_input_tokens),
        reasoning_tokens=_added(carried.reasoning_tokens, current.reasoning_tokens),
        total_tokens=_added(carried.total_tokens, current.total_tokens),
        provider_units=current.provider_units or carried.provider_units,
    )


def _hide_server_tool_event(event: StreamEvent, hidden_ids: set[str]) -> bool:
    """Whether a stream event belongs to a server-side builtin tool's echo
    handshake rather than to the answer. Those calls are KeyCall's to
    complete, and surfacing them would tell the caller to act on a call
    that is not theirs."""
    if event.kind == "tool_call_started" and event.name.startswith("$"):
        hidden_ids.add(event.id)
        return True
    if event.kind == "tool_call_arguments_delta" and event.id in hidden_ids:
        return True
    return bool(
        event.kind == "tool_call_complete" and event.tool_call.name.startswith("$")
    )


class _StreamCore:
    """State shared by the sync and async stream wrappers."""

    def __init__(self, client: _BaseClient, request: TextGenerationRequest) -> None:
        self._client = client
        self._request = request
        client._require_model_not_retired(request.model)
        self._assembler: StreamAssembler = client._adapter.stream_assembler(request)
        self._spec = client._adapter.build_stream_spec(request)
        self._started_at: float | None = None
        self._result: InvocationResult | None = None
        self._failed = False
        # Server-tool echo rounds (Moonshot web search): usage carried
        # from finished rounds, and how many rounds have run.
        self._carry_usage: Usage | None = None
        self._rounds = 1

    def _continuation(self) -> TextGenerationRequest | None:
        """After a round's terminal event: the follow-up request when the
        provider still owes the answer (a server-side tool wants its echo),
        None when this round's result is the answer. Advancing rebuilds the
        assembler and spec for the next round and banks this round's
        usage."""
        interim = self._assembler.finalize(round_trip_duration_ms=0.0)
        follow_up: TextGenerationRequest | None = (
            self._client._adapter.server_tool_continuation(self._request, interim)
        )
        if follow_up is None:
            return None
        if self._rounds >= _SERVER_TOOL_ROUND_BUDGET:
            self._failed = True
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self._client.provider,
                operation="text_generation",
                retryable=True,
            )
        self._carry_usage = (
            interim.usage
            if self._carry_usage is None
            else _merged_usage(self._carry_usage, interim.usage)
        )
        self._rounds += 1
        self._request = follow_up
        self._assembler = self._client._adapter.stream_assembler(follow_up)
        self._assembler.response_headers = {}
        self._spec = self._client._adapter.build_stream_spec(follow_up)
        return follow_up

    def _feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        try:
            return self._assembler.feed(event_name, data)
        except InbandStreamError as exc:
            self._failed = True
            raise KeyCallError(
                self._client._transport._scrub(exc.raw_message),
                code=exc.code,
                provider=self._client.provider,
                operation="text_generation",
                retryable=exc.retryable,
            ) from None
        except KeyCallError:
            self._failed = True
            raise

    def _check_terminal(self) -> None:
        """The stream closed; without the provider's terminal signal that is
        a truncation, never a completion."""
        if not self._assembler.saw_terminal:
            self._failed = True
            raise KeyCallError(
                "the stream ended before the provider's terminal event; "
                "the response is incomplete",
                code=ErrorCode.NETWORK_ERROR,
                provider=self._client.provider,
                operation="text_generation",
                retryable=True,
            )

    def _build_result(self) -> InvocationResult:
        if self._failed or not self._assembler.saw_terminal:
            raise KeyCallError(
                "the stream did not complete; no result is available",
                code=ErrorCode.NETWORK_ERROR,
                provider=self._client.provider,
                operation="text_generation",
            )
        if self._result is None:
            duration = (
                (time.monotonic() - self._started_at) * 1000.0 if self._started_at else 0.0
            )
            invocation = self._assembler.finalize(round_trip_duration_ms=duration)
            if self._carry_usage is not None:
                # The clock spans every round already; only tokens carry.
                invocation = dataclasses.replace(
                    invocation, usage=_merged_usage(self._carry_usage, invocation.usage)
                )
            invocation = _with_schema_warning(invocation, self._request, self._client.provider)
            invocation = _with_truncation_warning(invocation)
            self._result = _with_custom_tool_warning(
                invocation,
                self._request,
                self._client.provider,
                is_custom=self._client._resolved.is_custom,
            )
        return self._result


class TextStream(_StreamCore):
    """Iterate typed stream events; call result() after exhaustion for the
    full InvocationResult. The context manager owns the connection: leaving
    the block closes it, even on early break or exception."""

    def __enter__(self) -> Self:
        # Before the request goes out: time to first byte is part of the
        # round trip, and on providers that buffer it is most of it.
        self._started_at = time.monotonic()
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = self._ctx.__enter__()
        self._assembler.response_headers = headers
        self._events = events
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._ctx.__exit__(exc_type, exc, tb)

    def __iter__(self) -> Any:
        while True:
            # The round's finish event is held back until it is known to be
            # the last round: an echo round's finish is plumbing, not the
            # end of the answer.
            held: list[StreamEvent] = []
            hidden_ids: set[str] = set()
            for event_name, data in self._events:
                for event in self._feed(event_name, data):
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
                if self._assembler.saw_terminal:
                    break
            if not self._assembler.saw_terminal:
                for event in self._assembler.on_close():
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
            self._check_terminal()
            if self._continuation() is None:
                yield from held
                return
            self._ctx.__exit__(None, None, None)
            self._ctx = self._client._transport.stream_request(
                self._spec,
                operation="text_generation",
                translate_error=self._client._adapter.translate_error,
            )
            headers, events = self._ctx.__enter__()
            self._assembler.response_headers = headers
            self._events = events

    def result(self) -> InvocationResult:
        return self._build_result()


class AsyncTextStream(_StreamCore):
    """Async twin of TextStream."""

    async def __aenter__(self) -> Self:
        # See TextStream.__enter__: the clock starts before the request.
        self._started_at = time.monotonic()
        self._ctx = self._client._transport.stream_request(
            self._spec,
            operation="text_generation",
            translate_error=self._client._adapter.translate_error,
        )
        headers, events = await self._ctx.__aenter__()
        self._assembler.response_headers = headers
        self._events = events
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._ctx.__aexit__(exc_type, exc, tb)

    async def __aiter__(self) -> Any:
        while True:
            # See TextStream.__iter__: echo rounds hold their finish back.
            held: list[StreamEvent] = []
            hidden_ids: set[str] = set()
            async for event_name, data in self._events:
                for event in self._feed(event_name, data):
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
                if self._assembler.saw_terminal:
                    break
            if not self._assembler.saw_terminal:
                for event in self._assembler.on_close():
                    if _hide_server_tool_event(event, hidden_ids):
                        continue
                    if event.kind == "stream_finish":
                        held.append(event)
                        continue
                    yield event
            self._check_terminal()
            if self._continuation() is None:
                for event in held:
                    yield event
                return
            await self._ctx.__aexit__(None, None, None)
            self._ctx = self._client._transport.stream_request(
                self._spec,
                operation="text_generation",
                translate_error=self._client._adapter.translate_error,
            )
            headers, events = await self._ctx.__aenter__()
            self._assembler.response_headers = headers
            self._events = events

    def result(self) -> InvocationResult:
        return self._build_result()


class KeyCall(_BaseClient):
    """Synchronous client. See AsyncKeyCall for the awaitable equivalent."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        credential: Mapping[str, str] | None = None,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        trust_env: bool = True,
        httpx_transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            credential=credential,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        object.__setattr__(
            self,
            "_transport",
            Transport(
                self._resolved,
                self._require_open(),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_response_bytes=max_response_bytes,
                trust_env=trust_env,
                allow_private_network=allow_private_network,
                httpx_transport=httpx_transport,
            ),
        )

    def close(self) -> None:
        """Release the credential reference and the HTTP client."""
        self._transport.close()
        object.__setattr__(self, "_credential", None)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def list_models(
        self,
        *,
        categories: set[ModelCategory] | frozenset[ModelCategory] | None = None,
        refresh: bool = False,
    ) -> ModelDiscovery:
        credential = self._require_open()
        requested = _validate_categories(categories)
        fingerprint = credential.fingerprint()

        with _tracing.span(
            "keycall.list_models", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            if not refresh:
                cached = self._cached_discovery(requested, fingerprint, trace)
                if cached is not None:
                    return cached

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            next_spec = spec
            for _ in range(_MAX_LIST_PAGES):
                result = self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page_models, next_spec = self._parse_page(trace, spec, result)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec
            return self._store_discovery(
                models,
                truncated=next_spec is not None,
                categories=requested,
                fingerprint=fingerprint,
                trace=trace,
            )

    def probe_services(self) -> ServiceReport:
        """One billable live probe per catalog service category, reporting
        each category's standing on this key. Reaching a report at all
        means the credential authenticated; a bad key (or, on LiveKit, a
        secret that fails the signature) raises instead."""
        self._require_open()
        self._require_service_kind()
        with _tracing.span(
            "keycall.probe_services", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            statuses: list[ServiceStatus] = []
            for category, spec in self._adapter.service_probe_specs():
                try:
                    result = self._transport.request(
                        spec,
                        operation=Operation.SERVICE_PROBE.value,
                        retry_policy="list",
                        translate_error=self._adapter.translate_error,
                    )
                except KeyCallError as exc:
                    if exc.code is ErrorCode.INVALID_API_KEY:
                        # Credential-level, not a category fact: every
                        # other category would fail the same way.
                        raise
                    statuses.append(
                        self._service_probe_status(
                            trace,
                            category,
                            self._adapter.service_status_from_error(category, exc),
                        )
                    )
                else:
                    statuses.append(
                        self._service_probe_status(
                            trace,
                            category,
                            self._adapter.service_status_from_payload(category, result.payload),
                        )
                    )
            return ServiceReport(provider=self.provider, services=tuple(statuses))

    def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            carried: Usage | None = None
            carried_ms = 0.0
            for _round in range(_SERVER_TOOL_ROUND_BUDGET):
                spec = self._generation_spec(request)
                result = self._transport.request(
                    spec,
                    operation="text_generation",
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                invocation = self._parse_invocation(request, result, trace)
                follow_up = self._adapter.server_tool_continuation(request, invocation)
                if follow_up is None:
                    if carried is not None:
                        invocation = dataclasses.replace(
                            invocation,
                            usage=_merged_usage(carried, invocation.usage),
                            round_trip_duration_ms=carried_ms
                            + invocation.round_trip_duration_ms,
                        )
                    return invocation
                # A server-side tool wants its echo before the answer
                # comes (Moonshot web search); the tokens and time spent
                # this round belong to the one logical call.
                carried = (
                    invocation.usage
                    if carried is None
                    else _merged_usage(carried, invocation.usage)
                )
                carried_ms += invocation.round_trip_duration_ms
                request = follow_up
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self.provider,
                operation="text_generation",
                retryable=True,
            )

    def generate_image(self, *, model: str, prompt: str) -> InvocationResult:
        """Generate a picture. The result's parts are ImageOutput values
        carrying base64 data and the media type the provider produced."""
        self._require_open()
        request = ImageGenerationRequest(model=model, prompt=prompt)
        spec = self._image_spec(request)
        with _tracing.span(
            "keycall.image_generation", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="image_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_image(request, result, trace)

    def embed(self, *, model: str, inputs: Sequence[str]) -> InvocationResult:
        """Embed one or more strings. The result's parts are EmbeddingOutput
        values in the order the inputs were given, so they zip together."""
        self._require_open()
        request = EmbeddingRequest(model=model, inputs=inputs)
        spec = self._embedding_spec(request)
        with _tracing.span("keycall.embedding", provider=self.provider, model=model) as trace:
            result = self._transport.request(
                spec,
                operation="embedding",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_embedding(request, result, trace)

    def list_voices(self) -> tuple[Voice, ...]:
        """The voices this provider can speak with, as normalized Voice
        records. Providers whose voice set is a fixed named list answer
        from the bundled catalog without a network call (OpenAI, Gemini);
        providers with a live voices endpoint are asked (ElevenLabs, whose
        voices are account-scoped). A provider without speech generation
        refuses before the network."""
        self._require_open()
        if self._resolved.catalog_voices:
            return _catalog_voice_records(self._resolved)
        spec = self._adapter.build_voices_spec()
        result = self._transport.request(
            spec,
            operation="list_voices",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_voices_response(result.payload)

    def _require_voice(self, request: SpeechGenerationRequest) -> None:
        # Where the provider routes by voice there is no default to fall
        # back on; refuse with the actual choices rather than a bare
        # "voice required".
        if request.voice or not getattr(self._adapter, "requires_voice", False):
            return
        voices = self.list_voices()
        shown = ", ".join(f"{v.name} ({v.id})" for v in voices[:5])
        remainder = f", and {len(voices) - 5} more via list_voices()" if len(voices) > 5 else ""
        raise KeyCallError(
            f"{self.provider} routes speech by voice and has no default; "
            f"pass voice=<voice_id>. This key's voices include: {shown}{remainder}",
            code=ErrorCode.MODEL_NOT_SUITABLE,
            provider=self.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    def generate_speech(
        self, *, model: str, text: str, voice: str | None = None
    ) -> InvocationResult:
        """Speak text aloud. The result's one part is an AudioOutput
        carrying base64 data and the media type the provider produced —
        not necessarily a playable container; Gemini answers with raw PCM
        and says so in the media type."""
        self._require_open()
        request = SpeechGenerationRequest(model=model, text=text, voice=voice)
        self._require_voice(request)
        spec = self._speech_spec(request)
        with _tracing.span(
            "keycall.speech_generation", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="speech_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_speech(request, result, trace)

    def start_video(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> VideoJob:
        """Start a video render and return its job handle immediately.
        Rendering takes anywhere from seconds to many minutes depending on
        provider load; poll with check_video(), then fetch_video() once
        the job reports succeeded — or let generate_video() do all three
        against a waiting budget."""
        self._require_open()
        self._require_model_not_retired(model)
        request = VideoGenerationRequest(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        spec = self._adapter.build_video_start_spec(request)
        with _tracing.span(
            "keycall.video_generation.start", provider=self.provider, model=model
        ) as trace:
            result = self._transport.request(
                spec,
                operation="video_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_video_start(result.payload, model=request.model)
            trace.event(
                "model",
                operation="video_generation",
                target=model,
                duration_ms=result.duration_ms,
                result={"job": "started"},
            )
            return job

    def check_video(self, job: VideoJob) -> VideoJob:
        """Ask the provider where a render stands. Returns a new VideoJob
        rather than mutating; a job that already finished is returned
        as-is without a network call."""
        self._require_open()
        self._require_video_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_video_status_spec(job)
        result = self._transport.request(
            spec,
            operation="video_generation",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_video_status(result.payload, job=job)

    def fetch_video(self, job: VideoJob) -> InvocationResult:
        """Download a finished render. The result's one part is a
        VideoOutput carrying base64 data, the media type the provider
        served, and the provider's own download URL for as long as the
        provider keeps the file alive."""
        self._require_open()
        self._require_video_job(job)
        if job.status == "failed":
            self._raise_video_failure(job)
        if job.status != "succeeded" or not job.video_url:
            raise ValueError(
                "this job has not succeeded yet; call check_video() until it does"
            )
        plan = self._adapter.video_download_plan(job)
        with _tracing.span(
            "keycall.video_generation.fetch", provider=self.provider, model=job.model
        ) as trace:
            result = self._transport.download(
                plan,
                operation="video_generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._video_result_from_download(job, result)
            trace.event(
                "model",
                operation="video_generation",
                target=job.model,
                duration_ms=result.duration_ms,
                result={"videos": len(invocation.parts)},
            )
            return invocation

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
        poll_interval: float = 10.0,
    ) -> InvocationResult:
        """Start, poll, and download in one call. ``timeout`` is the
        caller's waiting budget in seconds and has no default: render
        times observed live range from 10 seconds to over 11 minutes, so
        only the caller can say how long is too long. When the budget
        runs out the raised VideoJobTimeout carries the still-valid job —
        the render keeps going provider-side and check_video() resumes
        where the wait left off."""
        job = self.start_video(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        deadline = time.monotonic() + timeout
        while True:
            job = self.check_video(job)
            if job.status == "succeeded":
                return self.fetch_video(job)
            if job.status == "failed":
                self._raise_video_failure(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoJobTimeout(
                    f"video render still going after {timeout:g}s; the job remains "
                    "valid — poll it with check_video(error.job)",
                    provider=self.provider,
                    job=job,
                )
            time.sleep(min(poll_interval, remaining))

    def start_batch(self, requests: Sequence[BatchRequest]) -> BatchJob:
        """Submit a batch of text-generation requests to the provider's
        asynchronous batch lane — half-price tokens on the three majors,
        with results arriving minutes to hours later — and return its job
        handle immediately. Poll with check_batch(), read outcomes with
        fetch_batch_results() once the job reports finished — or let
        generate_batch() run all three against a waiting budget."""
        self._require_open()
        return self._submit_batch(self._batch_submission(requests))

    def start_embedding_batch(self, *, model: str, inputs: Sequence[str]) -> BatchJob:
        """Submit a batch of embedding inputs (one vector per input, in
        input order, same as embed()) to the batch lane. OpenAI and Gemini
        support it; every other provider refuses before the network."""
        self._require_open()
        return self._submit_batch(self._embed_batch_submission(model, inputs))

    def _submit_batch(self, submission: BatchSubmission) -> BatchJob:
        with _tracing.span(
            "keycall.batch.start", provider=self.provider, operation=submission.operation
        ) as trace:
            prelude = None
            prelude_spec = self._adapter.build_batch_prelude_spec(submission)
            if prelude_spec is not None:
                result = self._transport.request(
                    prelude_spec,
                    operation=submission.operation,
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                prelude = self._adapter.parse_batch_prelude(result.payload)
            spec = self._adapter.build_batch_submit_spec(submission, prelude)
            result = self._transport.request(
                spec,
                operation=submission.operation,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_batch_submit(
                result.payload, submission=submission, prelude=prelude
            )
            trace.event(
                "model",
                operation=submission.operation,
                status="submitted",
                result={"requests": len(submission.items)},
            )
            return job

    def check_batch(self, job: BatchJob) -> BatchJob:
        """Ask the provider where a batch stands. Returns a new BatchJob
        rather than mutating; a job that already ended is returned as-is
        without a network call."""
        self._require_open()
        self._require_batch_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_batch_status_spec(job)
        result = self._transport.request(
            spec,
            operation=job.operation,
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_batch_status(result.payload, job=job)

    def fetch_batch_results(self, job: BatchJob) -> tuple[BatchResult, ...]:
        """Every request's outcome, in submission order: an
        InvocationResult where the request succeeded, the provider's
        per-request error where it didn't. A batch that ended cancelled or
        expired still reports whatever completed before the end."""
        self._require_open()
        self._require_batch_job(job)
        if job.status == "running":
            raise ValueError("this batch has not ended yet; call check_batch() until it does")
        if job.status == "failed":
            self._raise_batch_failure(job)
        with _tracing.span(
            "keycall.batch.results", provider=self.provider, operation=job.operation
        ) as trace:
            outcomes: list[Any] = []
            cursor: str | None = None
            while True:
                spec = self._adapter.build_batch_results_spec(job, cursor)
                if spec is None:
                    break
                result = self._transport.request(
                    spec,
                    operation=job.operation,
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page, cursor = self._adapter.parse_batch_results(result.payload, job=job)
                outcomes.extend(page)
                if not cursor:
                    break
            error_spec = self._adapter.build_batch_error_spec(job)
            if error_spec is not None:
                result = self._transport.request(
                    error_spec,
                    operation=job.operation,
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                outcomes.extend(self._adapter.parse_batch_error_file(result.payload, job=job))
            results = self._assemble_batch_results(job, outcomes)
            trace.event(
                "model",
                operation=job.operation,
                status="ok",
                result={
                    "requests": len(results),
                    "succeeded": sum(1 for entry in results if entry.succeeded),
                },
            )
            return results

    def cancel_batch(self, job: BatchJob) -> BatchJob:
        """Ask the provider to stop a running batch. Requests already
        processed stay billed and readable; the rest report cancelled in
        the results. A job that already ended is returned as-is."""
        self._require_open()
        self._require_batch_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_batch_cancel_spec(job)
        result = self._transport.request(
            spec,
            operation=job.operation,
            retry_policy="generation",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_batch_cancel(result.payload, job=job)

    def generate_batch(
        self,
        requests: Sequence[BatchRequest],
        *,
        timeout: float,
        poll_interval: float = 10.0,
    ) -> tuple[BatchResult, ...]:
        """Submit, poll, and fetch in one call. ``timeout`` is the
        caller's waiting budget in seconds and has no default: providers
        promise completion within 24 hours and usually finish in minutes,
        so only the caller can say how long is too long. When the budget
        runs out the raised BatchJobTimeout carries the still-valid job —
        the batch keeps processing provider-side and check_batch() resumes
        where the wait left off."""
        job = self.start_batch(requests)
        deadline = time.monotonic() + timeout
        while True:
            job = self.check_batch(job)
            if job.status == "failed":
                self._raise_batch_failure(job)
            if job.status != "running":
                return self.fetch_batch_results(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BatchJobTimeout(
                    f"batch still processing after {timeout:g}s; the job remains "
                    "valid — poll it with check_batch(error.job)",
                    provider=self.provider,
                    job=job,
                )
            time.sleep(min(poll_interval, remaining))

    def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> InvocationResult:
        return self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    def stream_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> TextStream:
        """Stream a text generation. Use as a context manager; iterate the
        typed events, then call result() for the full InvocationResult."""
        self._require_open()
        return TextStream(
            self,
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def realtime(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        provider_config: Mapping[str, Any] | None = None,
    ) -> RealtimeSession:
        """A live WebSocket session with a realtime model. Use as a
        context manager; push turns with send_text/send_audio and read
        normalized events from events()."""
        self._require_open()
        self._require_model_not_retired(model)
        config = RealtimeConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            provider_config=provider_config,
        )
        path, translator = self._adapter.realtime_plan(config)
        from ._realtime import RealtimeSession

        return RealtimeSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def live(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        backend_model: str | None = None,
        backend_tools: Sequence[Mapping[str, Any]] = (),
        provider_config: Mapping[str, Any] | None = None,
    ) -> LiveSession:
        """A full-duplex live voice session (OpenAI's gpt-live). A sibling
        of realtime() on its own endpoint: the caller's audio and the
        model's overlap, and reasoning is delegated to a backend model
        (backend_model plus backend_tools). Use as a context manager;
        stream caller audio with send_audio and read normalized events
        from events(). The model does its own endpointing, so no explicit
        turn boundary is required."""
        self._require_open()
        self._require_model_not_retired(model)
        config = LiveConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            backend_model=backend_model,
            backend_tools=tuple(backend_tools),
            provider_config=provider_config,
        )
        path, translator = self._adapter.live_plan(config)
        from ._live import LiveSession

        return LiveSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def transcribe_stream(
        self,
        *,
        model: str | None = None,
        sample_rate: int = 16000,
        diarize: bool = False,
    ) -> TranscriptionSession:
        """A live speech-to-text session (AssemblyAI, Deepgram,
        ElevenLabs). Use as a context manager; push raw 16-bit mono PCM
        with send_audio, call finish() when the audio ends, and read
        normalized events from events(). model None takes the provider's
        default streaming model. diarize=True labels each finalized word
        with its speaker on the providers whose streaming wire reports
        one (AssemblyAI, Deepgram); elsewhere it refuses before the
        socket opens rather than returning unlabelled words."""
        self._require_open()
        self._require_model_not_retired(model)
        config = TranscriptionConfig(
            model=model, sample_rate=sample_rate, diarize=diarize
        )
        path, translator = self._adapter.transcription_plan(config)
        from ._transcription import TranscriptionSession

        return TranscriptionSession(self._transport, path=path, translator=translator)

    def transcribe(
        self,
        *,
        model: str,
        audio: bytes | None = None,
        url: str | None = None,
        media_type: str | None = None,
        language: str | None = None,
        diarize: bool = False,
        timeout: float | None = None,
        poll_interval: float = 2.0,
    ) -> TranscriptionResult:
        """Transcribe a stored audio file (OpenAI, ElevenLabs, Deepgram,
        AssemblyAI). Pass one of ``audio`` (bytes) or ``url`` (a
        location the provider fetches itself — only some providers take
        one). On the sync providers the transcript comes back in one round
        trip and ``timeout`` does not apply; on a job-shaped provider
        (AssemblyAI) ``timeout`` is your required waiting budget in
        seconds, and running out raises TranscriptionJobTimeout carrying
        the still-valid job. Gemini refuses here: send the audio as an
        AudioInput on generate_text() with your own instruction."""
        self._require_open()
        request = self._transcription_request(
            model=model, audio=audio, url=url, media_type=media_type,
            language=language, diarize=diarize,
        )
        if not self._adapter.transcription_is_job_shaped:
            if timeout is not None:
                raise ValueError(
                    f"timeout applies to providers that transcribe as a job; "
                    f"{self.provider} answers in one round trip — set "
                    "read_timeout on the client instead"
                )
            return self._transcribe_sync(request)
        if timeout is None:
            raise ValueError(
                f"{self.provider} transcribes as a job; pass timeout=<seconds> "
                "as your waiting budget"
            )
        job = self._submit_transcription(request)
        deadline = time.monotonic() + timeout
        while True:
            job = self.check_transcription(job)
            if job.status == "failed":
                self._raise_transcription_failure(job)
            if job.status == "finished":
                return self.fetch_transcription(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptionJobTimeout(
                    f"transcription still processing after {timeout:g}s; the "
                    "job remains valid — poll it with "
                    "check_transcription(error.job)",
                    provider=self.provider,
                    job=job,
                )
            time.sleep(min(poll_interval, remaining))

    def _transcribe_sync(self, request: TranscriptionRequest) -> TranscriptionResult:
        with _tracing.span(
            "keycall.transcription", provider=self.provider, operation="transcription"
        ) as trace:
            spec = self._adapter.build_transcription_spec(request)
            result = self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            parsed = self._adapter.parse_transcription_response(
                result.payload,
                headers=result.headers,
                round_trip_duration_ms=result.duration_ms,
                model=request.model,
            )
            trace.event(
                "model",
                operation="transcription",
                target=request.model,
                status="ok",
                duration_ms=result.duration_ms,
                provider=self.provider,
                result={
                    "words": len(parsed.words),
                    "audio_seconds": parsed.audio_duration_seconds,
                },
            )
            return parsed

    def start_transcription(
        self,
        *,
        model: str,
        audio: bytes | None = None,
        url: str | None = None,
        media_type: str | None = None,
        language: str | None = None,
        diarize: bool = False,
    ) -> TranscriptionJob:
        """Submit a transcription to a job-shaped provider (AssemblyAI)
        and return its handle immediately. Poll with check_transcription()
        and read the result with fetch_transcription() — or let
        transcribe(timeout=) run all three against a waiting budget.
        Providers that answer in one round trip have no job to hand back
        and refuse here."""
        self._require_open()
        request = self._transcription_request(
            model=model, audio=audio, url=url, media_type=media_type,
            language=language, diarize=diarize,
        )
        if not self._adapter.transcription_is_job_shaped:
            self._refuse_transcription_jobs_here()
        return self._submit_transcription(request)

    def _submit_transcription(self, request: TranscriptionRequest) -> TranscriptionJob:
        with _tracing.span(
            "keycall.transcription.start",
            provider=self.provider,
            operation="transcription",
        ) as trace:
            audio_ref = None
            upload_spec = self._adapter.build_transcription_upload_spec(request)
            if upload_spec is not None:
                result = self._transport.request(
                    upload_spec,
                    operation=Operation.TRANSCRIPTION.value,
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                audio_ref = self._adapter.parse_transcription_upload(result.payload)
            spec = self._adapter.build_transcription_submit_spec(request, audio_ref)
            result = self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_transcription_submit(
                result.payload, request=request
            )
            trace.event(
                "model",
                operation="transcription",
                target=request.model,
                status="submitted",
                provider=self.provider,
            )
            return job

    def check_transcription(self, job: TranscriptionJob) -> TranscriptionJob:
        """Ask the provider where a transcription job stands. Returns a
        new TranscriptionJob rather than mutating; a job that already
        ended is returned as-is without a network call."""
        self._require_open()
        self._require_transcription_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_transcription_status_spec(job)
        result = self._transport.request(
            spec,
            operation=Operation.TRANSCRIPTION.value,
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_transcription_status(result.payload, job=job)

    def fetch_transcription(self, job: TranscriptionJob) -> TranscriptionResult:
        """The finished transcript of a job that reported finished. A
        running job is refused (poll check_transcription() first); a
        failed one raises with the provider's own explanation."""
        self._require_open()
        self._require_transcription_job(job)
        if job.status == "running":
            raise ValueError(
                "this transcription has not ended yet; call "
                "check_transcription() until it does"
            )
        if job.status == "failed":
            self._raise_transcription_failure(job)
        with _tracing.span(
            "keycall.transcription.results",
            provider=self.provider,
            operation="transcription",
        ) as trace:
            spec = self._adapter.build_transcription_status_spec(job)
            result = self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="list",
                translate_error=self._adapter.translate_error,
            )
            parsed = self._adapter.parse_transcription_result(result.payload, job=job)
            trace.event(
                "model",
                operation="transcription",
                target=job.model,
                status="ok",
                provider=self.provider,
                result={
                    "words": len(parsed.words),
                    "audio_seconds": parsed.audio_duration_seconds,
                },
            )
            return parsed


class AsyncKeyCall(_BaseClient):
    """Asynchronous client. Same identity rules and methods as KeyCall."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        credential: Mapping[str, str] | None = None,
        protocol: ProviderProtocol | str | None = None,
        base_url: str | None = None,
        allow_insecure_localhost: bool = False,
        allow_private_network: bool = False,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        trust_env: bool = True,
        httpx_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            credential=credential,
            protocol=protocol,
            base_url=base_url,
            allow_insecure_localhost=allow_insecure_localhost,
            allow_private_network=allow_private_network,
        )
        object.__setattr__(
            self,
            "_transport",
            AsyncTransport(
                self._resolved,
                self._require_open(),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_response_bytes=max_response_bytes,
                trust_env=trust_env,
                allow_private_network=allow_private_network,
                httpx_transport=httpx_transport,
            ),
        )

    async def close(self) -> None:
        await self._transport.close()
        object.__setattr__(self, "_credential", None)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def list_models(
        self,
        *,
        categories: set[ModelCategory] | frozenset[ModelCategory] | None = None,
        refresh: bool = False,
    ) -> ModelDiscovery:
        credential = self._require_open()
        requested = _validate_categories(categories)
        fingerprint = credential.fingerprint()

        with _tracing.span(
            "keycall.list_models", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            if not refresh:
                cached = self._cached_discovery(requested, fingerprint, trace)
                if cached is not None:
                    return cached

            models: list[Model] = []
            spec = self._adapter.initial_list_request()
            next_spec = spec
            for _ in range(_MAX_LIST_PAGES):
                result = await self._transport.request(
                    spec,
                    operation="list_models",
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page_models, next_spec = self._parse_page(trace, spec, result)
                models.extend(page_models)
                if next_spec is None:
                    break
                spec = next_spec
            return self._store_discovery(
                models,
                truncated=next_spec is not None,
                categories=requested,
                fingerprint=fingerprint,
                trace=trace,
            )

    async def probe_services(self) -> ServiceReport:
        """Async twin of KeyCall.probe_services()."""
        self._require_open()
        self._require_service_kind()
        with _tracing.span(
            "keycall.probe_services", provider=self.provider, protocol=self.protocol.value
        ) as trace:
            statuses: list[ServiceStatus] = []
            for category, spec in self._adapter.service_probe_specs():
                try:
                    result = await self._transport.request(
                        spec,
                        operation=Operation.SERVICE_PROBE.value,
                        retry_policy="list",
                        translate_error=self._adapter.translate_error,
                    )
                except KeyCallError as exc:
                    if exc.code is ErrorCode.INVALID_API_KEY:
                        raise
                    statuses.append(
                        self._service_probe_status(
                            trace,
                            category,
                            self._adapter.service_status_from_error(category, exc),
                        )
                    )
                else:
                    statuses.append(
                        self._service_probe_status(
                            trace,
                            category,
                            self._adapter.service_status_from_payload(category, result.payload),
                        )
                    )
            return ServiceReport(provider=self.provider, services=tuple(statuses))

    async def invoke(self, request: TextGenerationRequest) -> InvocationResult:
        self._require_open()
        with _tracing.span(
            "keycall.text_generation", provider=self.provider, model=request.model
        ) as trace:
            carried: Usage | None = None
            carried_ms = 0.0
            for _round in range(_SERVER_TOOL_ROUND_BUDGET):
                spec = self._generation_spec(request)
                result = await self._transport.request(
                    spec,
                    operation="text_generation",
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                invocation = self._parse_invocation(request, result, trace)
                follow_up = self._adapter.server_tool_continuation(request, invocation)
                if follow_up is None:
                    if carried is not None:
                        invocation = dataclasses.replace(
                            invocation,
                            usage=_merged_usage(carried, invocation.usage),
                            round_trip_duration_ms=carried_ms
                            + invocation.round_trip_duration_ms,
                        )
                    return invocation
                # See KeyCall.invoke: the echo rounds are one logical call.
                carried = (
                    invocation.usage
                    if carried is None
                    else _merged_usage(carried, invocation.usage)
                )
                carried_ms += invocation.round_trip_duration_ms
                request = follow_up
            raise KeyCallError(
                "the provider kept requesting server-side tool echoes beyond "
                f"the {_SERVER_TOOL_ROUND_BUDGET}-round budget",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                provider=self.provider,
                operation="text_generation",
                retryable=True,
            )

    async def generate_image(self, *, model: str, prompt: str) -> InvocationResult:
        """Async twin of KeyCall.generate_image()."""
        self._require_open()
        request = ImageGenerationRequest(model=model, prompt=prompt)
        spec = self._image_spec(request)
        with _tracing.span(
            "keycall.image_generation", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="image_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_image(request, result, trace)

    async def embed(self, *, model: str, inputs: Sequence[str]) -> InvocationResult:
        """Async twin of KeyCall.embed()."""
        self._require_open()
        request = EmbeddingRequest(model=model, inputs=inputs)
        spec = self._embedding_spec(request)
        with _tracing.span("keycall.embedding", provider=self.provider, model=model) as trace:
            result = await self._transport.request(
                spec,
                operation="embedding",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_embedding(request, result, trace)

    async def list_voices(self) -> tuple[Voice, ...]:
        """Async twin of KeyCall.list_voices()."""
        self._require_open()
        if self._resolved.catalog_voices:
            return _catalog_voice_records(self._resolved)
        spec = self._adapter.build_voices_spec()
        result = await self._transport.request(
            spec,
            operation="list_voices",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_voices_response(result.payload)

    async def _require_voice(self, request: SpeechGenerationRequest) -> None:
        if request.voice or not getattr(self._adapter, "requires_voice", False):
            return
        voices = await self.list_voices()
        shown = ", ".join(f"{v.name} ({v.id})" for v in voices[:5])
        remainder = f", and {len(voices) - 5} more via list_voices()" if len(voices) > 5 else ""
        raise KeyCallError(
            f"{self.provider} routes speech by voice and has no default; "
            f"pass voice=<voice_id>. This key's voices include: {shown}{remainder}",
            code=ErrorCode.MODEL_NOT_SUITABLE,
            provider=self.provider,
            operation=Operation.SPEECH_GENERATION.value,
        )

    async def generate_speech(
        self, *, model: str, text: str, voice: str | None = None
    ) -> InvocationResult:
        """Async twin of KeyCall.generate_speech()."""
        self._require_open()
        request = SpeechGenerationRequest(model=model, text=text, voice=voice)
        await self._require_voice(request)
        spec = self._speech_spec(request)
        with _tracing.span(
            "keycall.speech_generation", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="speech_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            return self._parse_speech(request, result, trace)

    async def start_video(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> VideoJob:
        """Async twin of KeyCall.start_video()."""
        self._require_open()
        self._require_model_not_retired(model)
        request = VideoGenerationRequest(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        spec = self._adapter.build_video_start_spec(request)
        with _tracing.span(
            "keycall.video_generation.start", provider=self.provider, model=model
        ) as trace:
            result = await self._transport.request(
                spec,
                operation="video_generation",
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_video_start(result.payload, model=request.model)
            trace.event(
                "model",
                operation="video_generation",
                target=model,
                duration_ms=result.duration_ms,
                result={"job": "started"},
            )
            return job

    async def check_video(self, job: VideoJob) -> VideoJob:
        """Async twin of KeyCall.check_video()."""
        self._require_open()
        self._require_video_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_video_status_spec(job)
        result = await self._transport.request(
            spec,
            operation="video_generation",
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_video_status(result.payload, job=job)

    async def fetch_video(self, job: VideoJob) -> InvocationResult:
        """Async twin of KeyCall.fetch_video()."""
        self._require_open()
        self._require_video_job(job)
        if job.status == "failed":
            self._raise_video_failure(job)
        if job.status != "succeeded" or not job.video_url:
            raise ValueError(
                "this job has not succeeded yet; call check_video() until it does"
            )
        plan = self._adapter.video_download_plan(job)
        with _tracing.span(
            "keycall.video_generation.fetch", provider=self.provider, model=job.model
        ) as trace:
            result = await self._transport.download(
                plan,
                operation="video_generation",
                translate_error=self._adapter.translate_error,
            )
            invocation = self._video_result_from_download(job, result)
            trace.event(
                "model",
                operation="video_generation",
                target=job.model,
                duration_ms=result.duration_ms,
                result={"videos": len(invocation.parts)},
            )
            return invocation

    async def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
        poll_interval: float = 10.0,
    ) -> InvocationResult:
        """Async twin of KeyCall.generate_video()."""
        import anyio

        job = await self.start_video(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        deadline = time.monotonic() + timeout
        while True:
            job = await self.check_video(job)
            if job.status == "succeeded":
                return await self.fetch_video(job)
            if job.status == "failed":
                self._raise_video_failure(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoJobTimeout(
                    f"video render still going after {timeout:g}s; the job remains "
                    "valid — poll it with check_video(error.job)",
                    provider=self.provider,
                    job=job,
                )
            await anyio.sleep(min(poll_interval, remaining))

    async def start_batch(self, requests: Sequence[BatchRequest]) -> BatchJob:
        """Async twin of KeyCall.start_batch()."""
        self._require_open()
        return await self._submit_batch(self._batch_submission(requests))

    async def start_embedding_batch(
        self, *, model: str, inputs: Sequence[str]
    ) -> BatchJob:
        """Async twin of KeyCall.start_embedding_batch()."""
        self._require_open()
        return await self._submit_batch(self._embed_batch_submission(model, inputs))

    async def _submit_batch(self, submission: BatchSubmission) -> BatchJob:
        with _tracing.span(
            "keycall.batch.start", provider=self.provider, operation=submission.operation
        ) as trace:
            prelude = None
            prelude_spec = self._adapter.build_batch_prelude_spec(submission)
            if prelude_spec is not None:
                result = await self._transport.request(
                    prelude_spec,
                    operation=submission.operation,
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                prelude = self._adapter.parse_batch_prelude(result.payload)
            spec = self._adapter.build_batch_submit_spec(submission, prelude)
            result = await self._transport.request(
                spec,
                operation=submission.operation,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_batch_submit(
                result.payload, submission=submission, prelude=prelude
            )
            trace.event(
                "model",
                operation=submission.operation,
                status="submitted",
                result={"requests": len(submission.items)},
            )
            return job

    async def check_batch(self, job: BatchJob) -> BatchJob:
        """Async twin of KeyCall.check_batch()."""
        self._require_open()
        self._require_batch_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_batch_status_spec(job)
        result = await self._transport.request(
            spec,
            operation=job.operation,
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_batch_status(result.payload, job=job)

    async def fetch_batch_results(self, job: BatchJob) -> tuple[BatchResult, ...]:
        """Async twin of KeyCall.fetch_batch_results()."""
        self._require_open()
        self._require_batch_job(job)
        if job.status == "running":
            raise ValueError("this batch has not ended yet; call check_batch() until it does")
        if job.status == "failed":
            self._raise_batch_failure(job)
        with _tracing.span(
            "keycall.batch.results", provider=self.provider, operation=job.operation
        ) as trace:
            outcomes: list[Any] = []
            cursor: str | None = None
            while True:
                spec = self._adapter.build_batch_results_spec(job, cursor)
                if spec is None:
                    break
                result = await self._transport.request(
                    spec,
                    operation=job.operation,
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                page, cursor = self._adapter.parse_batch_results(result.payload, job=job)
                outcomes.extend(page)
                if not cursor:
                    break
            error_spec = self._adapter.build_batch_error_spec(job)
            if error_spec is not None:
                result = await self._transport.request(
                    error_spec,
                    operation=job.operation,
                    retry_policy="list",
                    translate_error=self._adapter.translate_error,
                )
                outcomes.extend(self._adapter.parse_batch_error_file(result.payload, job=job))
            results = self._assemble_batch_results(job, outcomes)
            trace.event(
                "model",
                operation=job.operation,
                status="ok",
                result={
                    "requests": len(results),
                    "succeeded": sum(1 for entry in results if entry.succeeded),
                },
            )
            return results

    async def cancel_batch(self, job: BatchJob) -> BatchJob:
        """Async twin of KeyCall.cancel_batch()."""
        self._require_open()
        self._require_batch_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_batch_cancel_spec(job)
        result = await self._transport.request(
            spec,
            operation=job.operation,
            retry_policy="generation",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_batch_cancel(result.payload, job=job)

    async def generate_batch(
        self,
        requests: Sequence[BatchRequest],
        *,
        timeout: float,
        poll_interval: float = 10.0,
    ) -> tuple[BatchResult, ...]:
        """Async twin of KeyCall.generate_batch()."""
        import anyio

        job = await self.start_batch(requests)
        deadline = time.monotonic() + timeout
        while True:
            job = await self.check_batch(job)
            if job.status == "failed":
                self._raise_batch_failure(job)
            if job.status != "running":
                return await self.fetch_batch_results(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BatchJobTimeout(
                    f"batch still processing after {timeout:g}s; the job remains "
                    "valid — poll it with check_batch(error.job)",
                    provider=self.provider,
                    job=job,
                )
            await anyio.sleep(min(poll_interval, remaining))

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> InvocationResult:
        return await self.invoke(
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    def stream_text(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        web_search: bool = False,
        apply_patch: bool = False,
        code_interpreter: bool = False,
        reasoning_effort: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Tool] = (),
        tool_choice: str | None = None,
    ) -> AsyncTextStream:
        """Stream a text generation. Use as an async context manager;
        iterate with `async for`, then call result()."""
        self._require_open()
        return AsyncTextStream(
            self,
            TextGenerationRequest(
                model=model,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                web_search=web_search,
                apply_patch=apply_patch,
                code_interpreter=code_interpreter,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def realtime(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        provider_config: Mapping[str, Any] | None = None,
    ) -> AsyncRealtimeSession:
        """A live WebSocket session with a realtime model. Use as an
        async context manager; push turns with send_text/send_audio and
        read normalized events with `async for`."""
        self._require_open()
        self._require_model_not_retired(model)
        config = RealtimeConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            provider_config=provider_config,
        )
        path, translator = self._adapter.realtime_plan(config)
        from ._realtime import AsyncRealtimeSession

        return AsyncRealtimeSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def live(
        self,
        *,
        model: str,
        voice: str | None = None,
        instructions: str | None = None,
        backend_model: str | None = None,
        backend_tools: Sequence[Mapping[str, Any]] = (),
        provider_config: Mapping[str, Any] | None = None,
    ) -> AsyncLiveSession:
        """A full-duplex live voice session (OpenAI's gpt-live), the async
        twin of KeyCall.live(). Use as an async context manager; stream
        caller audio with send_audio and read normalized events with
        `async for`."""
        self._require_open()
        self._require_model_not_retired(model)
        config = LiveConfig(
            model=model,
            voice=voice,
            instructions=instructions,
            backend_model=backend_model,
            backend_tools=tuple(backend_tools),
            provider_config=provider_config,
        )
        path, translator = self._adapter.live_plan(config)
        from ._live import AsyncLiveSession

        return AsyncLiveSession(
            self._transport,
            path=path,
            translator=translator,
            provider=self.provider,
            config=config,
        )

    def transcribe_stream(
        self,
        *,
        model: str | None = None,
        sample_rate: int = 16000,
        diarize: bool = False,
    ) -> AsyncTranscriptionSession:
        """Async twin of KeyCall.transcribe_stream."""
        self._require_open()
        self._require_model_not_retired(model)
        config = TranscriptionConfig(
            model=model, sample_rate=sample_rate, diarize=diarize
        )
        path, translator = self._adapter.transcription_plan(config)
        from ._transcription import AsyncTranscriptionSession

        return AsyncTranscriptionSession(
            self._transport, path=path, translator=translator
        )

    async def transcribe(
        self,
        *,
        model: str,
        audio: bytes | None = None,
        url: str | None = None,
        media_type: str | None = None,
        language: str | None = None,
        diarize: bool = False,
        timeout: float | None = None,
        poll_interval: float = 2.0,
    ) -> TranscriptionResult:
        """Async twin of KeyCall.transcribe()."""
        import anyio

        self._require_open()
        request = self._transcription_request(
            model=model, audio=audio, url=url, media_type=media_type,
            language=language, diarize=diarize,
        )
        if not self._adapter.transcription_is_job_shaped:
            if timeout is not None:
                raise ValueError(
                    f"timeout applies to providers that transcribe as a job; "
                    f"{self.provider} answers in one round trip — set "
                    "read_timeout on the client instead"
                )
            return await self._transcribe_sync(request)
        if timeout is None:
            raise ValueError(
                f"{self.provider} transcribes as a job; pass timeout=<seconds> "
                "as your waiting budget"
            )
        job = await self._submit_transcription(request)
        deadline = time.monotonic() + timeout
        while True:
            job = await self.check_transcription(job)
            if job.status == "failed":
                self._raise_transcription_failure(job)
            if job.status == "finished":
                return await self.fetch_transcription(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptionJobTimeout(
                    f"transcription still processing after {timeout:g}s; the "
                    "job remains valid — poll it with "
                    "check_transcription(error.job)",
                    provider=self.provider,
                    job=job,
                )
            await anyio.sleep(min(poll_interval, remaining))

    async def _transcribe_sync(self, request: TranscriptionRequest) -> TranscriptionResult:
        with _tracing.span(
            "keycall.transcription", provider=self.provider, operation="transcription"
        ) as trace:
            spec = self._adapter.build_transcription_spec(request)
            result = await self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            parsed = self._adapter.parse_transcription_response(
                result.payload,
                headers=result.headers,
                round_trip_duration_ms=result.duration_ms,
                model=request.model,
            )
            trace.event(
                "model",
                operation="transcription",
                target=request.model,
                status="ok",
                duration_ms=result.duration_ms,
                provider=self.provider,
                result={
                    "words": len(parsed.words),
                    "audio_seconds": parsed.audio_duration_seconds,
                },
            )
            return parsed

    async def start_transcription(
        self,
        *,
        model: str,
        audio: bytes | None = None,
        url: str | None = None,
        media_type: str | None = None,
        language: str | None = None,
        diarize: bool = False,
    ) -> TranscriptionJob:
        """Async twin of KeyCall.start_transcription()."""
        self._require_open()
        request = self._transcription_request(
            model=model, audio=audio, url=url, media_type=media_type,
            language=language, diarize=diarize,
        )
        if not self._adapter.transcription_is_job_shaped:
            self._refuse_transcription_jobs_here()
        return await self._submit_transcription(request)

    async def _submit_transcription(
        self, request: TranscriptionRequest
    ) -> TranscriptionJob:
        with _tracing.span(
            "keycall.transcription.start",
            provider=self.provider,
            operation="transcription",
        ) as trace:
            audio_ref = None
            upload_spec = self._adapter.build_transcription_upload_spec(request)
            if upload_spec is not None:
                result = await self._transport.request(
                    upload_spec,
                    operation=Operation.TRANSCRIPTION.value,
                    retry_policy="generation",
                    translate_error=self._adapter.translate_error,
                )
                audio_ref = self._adapter.parse_transcription_upload(result.payload)
            spec = self._adapter.build_transcription_submit_spec(request, audio_ref)
            result = await self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="generation",
                translate_error=self._adapter.translate_error,
            )
            job = self._adapter.parse_transcription_submit(
                result.payload, request=request
            )
            trace.event(
                "model",
                operation="transcription",
                target=request.model,
                status="submitted",
                provider=self.provider,
            )
            return job

    async def check_transcription(self, job: TranscriptionJob) -> TranscriptionJob:
        """Async twin of KeyCall.check_transcription()."""
        self._require_open()
        self._require_transcription_job(job)
        if job.status != "running":
            return job
        spec = self._adapter.build_transcription_status_spec(job)
        result = await self._transport.request(
            spec,
            operation=Operation.TRANSCRIPTION.value,
            retry_policy="list",
            translate_error=self._adapter.translate_error,
        )
        return self._adapter.parse_transcription_status(result.payload, job=job)

    async def fetch_transcription(self, job: TranscriptionJob) -> TranscriptionResult:
        """Async twin of KeyCall.fetch_transcription()."""
        self._require_open()
        self._require_transcription_job(job)
        if job.status == "running":
            raise ValueError(
                "this transcription has not ended yet; call "
                "check_transcription() until it does"
            )
        if job.status == "failed":
            self._raise_transcription_failure(job)
        with _tracing.span(
            "keycall.transcription.results",
            provider=self.provider,
            operation="transcription",
        ) as trace:
            spec = self._adapter.build_transcription_status_spec(job)
            result = await self._transport.request(
                spec,
                operation=Operation.TRANSCRIPTION.value,
                retry_policy="list",
                translate_error=self._adapter.translate_error,
            )
            parsed = self._adapter.parse_transcription_result(result.payload, job=job)
            trace.event(
                "model",
                operation="transcription",
                target=job.model,
                status="ok",
                provider=self.provider,
                result={
                    "words": len(parsed.words),
                    "audio_seconds": parsed.audio_duration_seconds,
                },
            )
            return parsed
