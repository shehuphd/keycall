"""xAI adapter.

Chat, streaming, and tool calling ride the openai-compatible protocol
unchanged (SSE chunks carry a ``reasoning_content`` delta field, which the
compat assembler already handles for DeepSeek). What needs an override:

- Image generation answers with a ``url`` by default; KeyCall asks for
  ``b64_json`` so the result carries bytes like every other provider, and
  each entry names its own ``mime_type`` (verified live 2026-08-13).
- Web search is served by xAI's Agent Tools API on ``POST /v1/responses``
  — the OpenAI Responses shape, with the same streaming event names and
  ``url_citation`` annotations (verified live 2026-08-14) — while plain
  generation stays on chat completions. A request with ``web_search=True``
  is therefore delegated to the OpenAI adapter, rebound to the responses
  path; everything else rides the compat protocol. ``reasoning_effort``
  takes the same detour: chat completions answers 200 to the field but
  reasoning token counts do not follow the value, while the responses
  route's ``reasoning.effort`` binds (both measured live 2026-08-14).
- Model listing appends Grok Voice from the catalog: GET /v1/models
  doesn't list it (checked live 2026-08-15), so a key with realtime
  access would otherwise show none.
- Video generation is the three-phase job lifecycle:
  ``POST /v1/videos/generations`` answers ``{"request_id": ...}``
  immediately, ``GET /v1/videos/{request_id}`` reports ``pending`` /
  ``done`` / ``expired`` / ``failed``, and a finished job names a plain
  MP4 URL on ``vidgen.x.ai`` — unsigned and fetchable with no credential,
  so the URL itself is the only secret and the auth header is never sent
  to that host (all verified live 2026-08-13).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._sanitize import safe_request_id
from .._transport import DownloadPlan, RequestSpec
from .._types import (
    BatchCounts,
    BatchJob,
    BatchStatus,
    InvocationResult,
    Model,
    TextGenerationRequest,
    Usage,
    VideoJob,
)
from ._base import BatchItemOutcome, BatchSubmission, StreamAssembler
from ._openai import OpenAIAdapter
from ._openai_compat import OpenAICompatibleAdapter


class XAIAdapter(OpenAICompatibleAdapter):
    # --- web search via the Responses route ---

    def _responses_adapter(self) -> OpenAIAdapter:
        """An OpenAI adapter over this same provider profile, with text
        generation rebound to the responses path. Capabilities, auth, and
        provider identity stay xAI's — only the route and wire shape
        change, which is the whole difference between the two surfaces."""
        rebound = dataclasses.replace(
            self.resolved,
            operations={
                **self.resolved.operations,
                "text_generation": self.resolved.operations["responses_generation"],
            },
        )
        return OpenAIAdapter(rebound)

    def _needs_responses_route(self, request: TextGenerationRequest) -> bool:
        return (
            request.web_search
            or request.reasoning_effort is not None
            or request.code_interpreter
        )

    def _reject_seed_on_responses_route(self, request: TextGenerationRequest) -> None:
        """xAI's chat-completions route takes a seed; its Agent Tools route
        (POST /v1/responses, taken for web search, reasoning effort, and code
        execution) has no seed field. A seed set alongside one of those would
        be dropped on the way to that route, so it is refused here rather than
        silently lost — the caller keeps or drops the seed deliberately."""
        if request.seed is not None and self._needs_responses_route(request):
            raise KeyCallError(
                "xai cannot combine a seed with web_search, reasoning_effort, "
                "or code_interpreter: those route through xAI's Agent Tools "
                "API, which has no seed field. Drop the seed or the other option.",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider="xai",
                operation=Operation.TEXT_GENERATION.value,
            )

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self._reject_seed_on_responses_route(request)
        if self._needs_responses_route(request):
            return self._responses_adapter().build_generation_spec(request)
        return super().build_generation_spec(request)

    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self._reject_seed_on_responses_route(request)
        if self._needs_responses_route(request):
            return self._responses_adapter().build_stream_spec(request)
        return super().build_stream_spec(request)

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        if self._needs_responses_route(request):
            return self._responses_adapter().stream_assembler(request)
        return super().stream_assembler(request)

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        # The two surfaces are distinguishable from the payload itself:
        # a Responses body carries `output`, a chat completion `choices`.
        if isinstance(payload, dict) and "output" in payload and "choices" not in payload:
            return self._responses_adapter().parse_generation_response(
                payload,
                headers=headers,
                round_trip_duration_ms=round_trip_duration_ms,
                model=model,
            )
        return super().parse_generation_response(
            payload,
            headers=headers,
            round_trip_duration_ms=round_trip_duration_ms,
            model=model,
        )

    # --- model listing ---

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        # Grok Voice is absent from GET /v1/models (checked live
        # 2026-08-15, missing from a 12-model response) despite being a
        # documented, live-verified model, so it never surfaces from
        # discovery alone. Appended from the catalog instead, the same
        # way Perplexity's fully undiscoverable list is built, except
        # here only to cover the one model live discovery misses.
        models, next_spec = super().parse_model_page(payload)
        seen = {model.id for model in models}
        for entry in self.resolved.catalog_models:
            model_id = str(entry["id"])
            if model_id in seen:
                continue
            models.append(
                Model(
                    id=model_id,
                    provider=self.resolved.provider,
                    categories=frozenset(
                        ModelCategory(category) for category in entry.get("categories", [])
                    ),
                    classification_source="keycall_catalog",
                    warnings=("not listed by this key's model endpoint; carried from KeyCall's catalog",),
                )
            )
        return models, next_spec

    # --- realtime ---

    def realtime_plan(self, config: Any) -> tuple[str, Any]:
        if not self.resolved.capabilities.realtime or "realtime" not in self.resolved.operations:
            return super().realtime_plan(config)
        from ._realtime import OpenAIRealtimeTranslator

        path = self.resolved.operations["realtime"]["path"].format(
            model=quote(config.model, safe="")
        )
        # Grok Voice speaks the pre-GA session shape (its session object
        # keys `modalities`, and voice is a top-level session field).
        translator = OpenAIRealtimeTranslator(
            config, provider=self.resolved.provider, ga_session=False
        )
        return path, translator

    # --- image generation ---

    def build_image_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["image_generation"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            json_body={
                "model": request.model,
                "prompt": request.prompt,
                # The default answer is a URL on imgen.x.ai; asking for
                # b64_json keeps the result in bytes like every other
                # image-generating provider (verified live 2026-08-13).
                "response_format": "b64_json",
            },
        )

    def parse_image_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise KeyCallError(
                "image response missing 'data' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.IMAGE_GENERATION.value,
            )
        # Unlike OpenAI's single response-level output_format, each entry
        # names its own mime_type.
        images = [
            (str(entry["b64_json"]), str(entry.get("mime_type", "image/png")))
            for entry in payload["data"]
            if isinstance(entry, dict) and entry.get("b64_json")
        ]
        return self.image_result(
            images,
            usage=Usage(),
            model=model,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(
                headers.get(self.resolved.provider_request_id_header or "")
            ),
        )

    # --- batch generation ---
    #
    # xAI's own dialect (live-verified 2026-09-02): create a named batch
    # container, add the requests in a second call (atomic — one
    # ineligible model rejects the whole add), poll counters (no status
    # string exists; the state is derived from them), then page through
    # results, which are retrievable even before the batch ends. Only a
    # subset of models is batch-eligible per xAI's model pages.

    def build_batch_prelude_spec(self, submission: BatchSubmission) -> RequestSpec | None:
        op = self.resolved.operations["batch_create"]
        return RequestSpec(method=op["method"], path=op["path"], json_body={"name": "keycall batch"})

    def parse_batch_prelude(self, payload: Any) -> str:
        batch_id = payload.get("batch_id") if isinstance(payload, dict) else None
        if not batch_id:
            raise KeyCallError(
                "batch create returned no batch id",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.BATCH_GENERATION.value,
            )
        return str(batch_id)

    def build_batch_submit_spec(
        self, submission: BatchSubmission, prelude: str | None
    ) -> RequestSpec:
        op = self.resolved.operations["batch_add"]
        assert prelude is not None
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{batch_id}", prelude),
            json_body={
                "batch_requests": [
                    {"batch_request_id": key, "batch_request": {"chat_get_completion": dict(body)}}
                    for key, body in submission.items
                ]
            },
        )

    def parse_batch_submit(
        self, payload: Any, *, submission: BatchSubmission, prelude: str | None
    ) -> BatchJob:
        # The add call answers with an empty body; the handle came from
        # the container create.
        assert prelude is not None
        return BatchJob(
            provider=self.resolved.provider,
            job_id=prelude,
            operation=submission.operation,
            model=submission.model,
            request_keys=tuple(key for key, _ in submission.items),
            counts=BatchCounts(total=len(submission.items), pending=len(submission.items)),
        )

    def build_batch_status_spec(self, job: BatchJob) -> RequestSpec:
        op = self.resolved.operations["batch_status"]
        return RequestSpec(method=op["method"], path=op["path"].replace("{batch_id}", job.job_id))

    def parse_batch_status(self, payload: Any, *, job: BatchJob) -> BatchJob:
        state = payload.get("state") if isinstance(payload, dict) else None
        state = state if isinstance(state, dict) else {}
        counts = BatchCounts(
            total=state.get("num_requests"),
            pending=state.get("num_pending"),
            succeeded=state.get("num_success"),
            errored=state.get("num_error"),
            cancelled=state.get("num_cancelled"),
        )
        cancelled = bool(payload.get("cancel_time")) if isinstance(payload, dict) else False
        status: BatchStatus
        if cancelled:
            status = "cancelled"
        elif (counts.total or 0) > 0 and (counts.pending or 0) == 0:
            status = "finished"
        else:
            status = "running"
        message = payload.get("cancel_by_xai_message") if isinstance(payload, dict) else None
        return dataclasses.replace(
            job,
            status=status,
            counts=counts,
            error_message=str(message) if message else None,
        )

    def build_batch_results_spec(self, job: BatchJob, cursor: str | None) -> RequestSpec:
        op = self.resolved.operations["batch_results"]
        params = {"pagination_token": cursor} if cursor else {}
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{batch_id}", job.job_id),
            params=params,
        )

    def parse_batch_results(
        self, payload: Any, *, job: BatchJob
    ) -> tuple[list[BatchItemOutcome], str | None]:
        entries = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise KeyCallError(
                "batch results response carried no results list",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=job.operation,
            )
        outcomes: list[BatchItemOutcome] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("batch_request_id"):
                continue
            key = str(entry["batch_request_id"])
            raw_result = entry.get("batch_result")
            result = raw_result if isinstance(raw_result, dict) else {}
            raw_response = result.get("response")
            response = raw_response if isinstance(raw_response, dict) else {}
            body = response.get("chat_get_completion")
            if isinstance(body, dict):
                outcomes.append(BatchItemOutcome(key=key, body=body))
                continue
            message = entry.get("error_message")
            outcomes.append(
                BatchItemOutcome(
                    key=key,
                    error_message=str(message) if message else "request errored",
                )
            )
        cursor = payload.get("pagination_token") if isinstance(payload, dict) else None
        return outcomes, str(cursor) if cursor else None

    def build_batch_cancel_spec(self, job: BatchJob) -> RequestSpec:
        op = self.resolved.operations["batch_cancel"]
        return RequestSpec(method=op["method"], path=op["path"].replace("{batch_id}", job.job_id))

    def parse_batch_cancel(self, payload: Any, *, job: BatchJob) -> BatchJob:
        return self.parse_batch_status(payload, job=job)

    # --- video generation ---

    def build_video_start_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["video_generation"]
        body: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
        if request.duration_seconds is not None:
            body["duration"] = request.duration_seconds
        if request.aspect_ratio:
            body["aspect_ratio"] = request.aspect_ratio
        return RequestSpec(method=op["method"], path=op["path"], json_body=body)

    def parse_video_start(self, payload: Any, *, model: str) -> VideoJob:
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id:
            raise KeyCallError(
                "provider did not return a request_id for the video job",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.VIDEO_GENERATION.value,
            )
        return VideoJob(provider=self.resolved.provider, model=model, job_id=request_id)

    def build_video_status_spec(self, job: VideoJob) -> RequestSpec:
        op = self.resolved.operations["video_status"]
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{request_id}", quote(job.job_id, safe="")),
        )

    def parse_video_status(self, payload: Any, *, job: VideoJob) -> VideoJob:
        data = payload if isinstance(payload, dict) else {}
        provider_status = str(data.get("status", ""))
        if provider_status in ("", "pending"):
            return job
        if provider_status == "done":
            url = data.get("video", {}).get("url") if isinstance(data.get("video"), dict) else None
            if not isinstance(url, str) or not url:
                raise KeyCallError(
                    "video job reported done without a video URL",
                    code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                    provider=self.resolved.provider,
                    operation=Operation.VIDEO_GENERATION.value,
                )
            return VideoJob(
                provider=job.provider,
                model=job.model,
                job_id=job.job_id,
                status="succeeded",
                provider_status=provider_status,
                video_url=url,
            )
        # `failed` and `expired` both end the job; the provider's own word
        # for the state rides alongside rather than growing KeyCall's
        # closed status set by one vendor vocabulary at a time.
        message = str(data.get("error", data.get("detail", "video generation failed")))[:300]
        return VideoJob(
            provider=job.provider,
            model=job.model,
            job_id=job.job_id,
            status="failed",
            provider_status=provider_status,
            error_message=message,
        )

    def video_download_plan(self, job: VideoJob) -> DownloadPlan:
        # The finished file is served from a storage host pinned in the
        # catalog, as an unsigned public URL: no credential travels there,
        # and the URL should be treated as the secret it is.
        return DownloadPlan(
            url=job.video_url or "",
            allowed_hosts=self.resolved.video_download_hosts,
            send_credential=False,
            allow_same_origin_redirect=False,
        )
