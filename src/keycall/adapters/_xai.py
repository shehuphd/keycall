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

from .._enums import Operation
from .._errors import ErrorCode, KeyCallError
from .._sanitize import safe_request_id
from .._transport import DownloadPlan, RequestSpec
from .._types import InvocationResult, TextGenerationRequest, Usage, VideoJob
from ._base import StreamAssembler
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
        return request.web_search or request.reasoning_effort is not None

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        if self._needs_responses_route(request):
            return self._responses_adapter().build_generation_spec(request)
        return super().build_generation_spec(request)

    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
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
