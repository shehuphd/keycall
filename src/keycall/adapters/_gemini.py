"""Google Gemini adapter: GET /models (pageToken pagination),
POST /models/{model}:generateContent.

Gemini's list response is the strongest provider-native classification
source in the v1 set: supportedGenerationMethods is explicit provider
metadata and takes precedence over identifier rules.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from .._classify import classify_model_id
from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._registry import ResolvedProvider
from .._sanitize import safe_request_id
from .._transport import DownloadPlan, RequestSpec
from .._types import (
    AudioInput,
    Citation,
    CitationFound,
    FileInput,
    ImageInput,
    InvocationResult,
    Model,
    OutputPart,
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    TextGenerationRequest,
    TextInput,
    TextOutput,
    ToolCall,
    ToolResult,
    UnknownOutput,
    UnknownStreamEvent,
    Usage,
    VideoJob,
)
from ._base import (
    ProviderAdapter,
    StreamAssembler,
    context_limit,
    dedupe_citations,
    media_type_for,
    parse_tool_arguments,
)

_PAGE_SIZE = "1000"

# Modalities specific enough that an identifier match outranks the generic
# "supports generateContent" signal.
_DISTINCTIVE_MODALITIES = frozenset(
    {
        ModelCategory.SPEECH_GENERATION,
        ModelCategory.TRANSCRIPTION,
        ModelCategory.IMAGE_GENERATION,
        ModelCategory.VIDEO_GENERATION,
        ModelCategory.EMBEDDING,
    }
)

# Families that advertise generateContent and then refuse a text call are
# maintained in the catalog under gemini.capabilities.non_text_model_families,
# each verified 2026-08-09 by calling it: the Interactions-only models answer
# "This model only supports Interactions API", the computer-use preview
# demands its own tool, and Lyria is music generation that returns a 500
# here. Provider metadata is wrong for these, so the identifier is the
# better evidence and they classify UNKNOWN, never silently entering a
# caller's default text picker.


_METHOD_CATEGORIES = {
    "generateContent": ModelCategory.TEXT_GENERATION,
    "embedContent": ModelCategory.EMBEDDING,
    "embedText": ModelCategory.EMBEDDING,
    "predict": ModelCategory.IMAGE_GENERATION,
    "predictLongRunning": ModelCategory.VIDEO_GENERATION,
    "bidiGenerateContent": ModelCategory.REALTIME,
}


def _strip_prefix(name: str) -> str:
    return name.removeprefix("models/")


class _GeminiStreamAssembler(StreamAssembler):
    """Gemini's SSE stream has no event names and no terminal marker: each
    data line is a full GenerateContentResponse chunk, finishReason arrives
    on or before the last chunk, and the connection then closes
    (live-verified 2026-08-08). Completion is therefore the connection
    close after a finishReason was seen, via on_close()."""

    def __init__(self, resolved: ResolvedProvider, request: TextGenerationRequest) -> None:
        super().__init__(resolved, request)
        self._started = False
        self._seen_citations: set[tuple[str, str | None, str | None]] = set()

    def feed(self, event_name: str | None, data: str) -> list[StreamEvent]:
        payload = self._parse_data(data)
        if not isinstance(payload, dict):
            return []
        events: list[StreamEvent] = []
        if payload.get("modelVersion"):
            self.model = _strip_prefix(str(payload["modelVersion"]))
        if payload.get("responseId"):
            self.provider_request_id = safe_request_id(payload.get("responseId"))
        if not self._started:
            self._started = True
            events.append(StreamStart(model=self.model))

        candidates = payload.get("candidates")
        candidate = candidates[0] if isinstance(candidates, list) and candidates else None
        if isinstance(candidate, dict):
            content = candidate.get("content")
            if isinstance(content, dict):
                for part in content.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        text = str(part["text"])
                        self.append_text(text)
                        events.append(TextDelta(text=text))
                    elif isinstance(part, dict) and isinstance(part.get("functionCall"), dict):
                        # Gemini sends each call whole, with the thought
                        # signature as a sibling of functionCall; the echo
                        # data is required back verbatim on replay.
                        call = part["functionCall"]
                        echo = {
                            key: value
                            for key, value in (
                                ("thoughtSignature", part.get("thoughtSignature")),
                                ("id", call.get("id")),
                            )
                            if value
                        }
                        events.extend(
                            self.record_tool_call(
                                ToolCall(
                                    id=str(call.get("id", "")),
                                    name=str(call.get("name", "")),
                                    arguments=parse_tool_arguments(
                                        call.get("args", {}),
                                        provider=self.resolved.provider,
                                    ),
                                    opaque=json.dumps(echo) if echo else None,
                                )
                            )
                        )
                    elif isinstance(part, dict):
                        kind = next(iter(part.keys()), "?")
                        events.append(UnknownStreamEvent(provider_kind=str(kind)))
            grounding = candidate.get("groundingMetadata")
            if isinstance(grounding, dict):
                for chunk in grounding.get("groundingChunks") or []:
                    web = chunk.get("web") if isinstance(chunk, dict) else None
                    if isinstance(web, dict) and web.get("uri"):
                        # Chunks repeat across stream events. Guard on the
                        # same identity dedupe_citations uses, so the events
                        # a caller sees match the final result exactly.
                        citation = Citation(url=str(web["uri"]), title=web.get("title"))
                        identity = (citation.url, citation.title, citation.cited_text)
                        if identity not in self._seen_citations:
                            self._seen_citations.add(identity)
                            self.citations.append(citation)
                            events.append(CitationFound(citation=citation))

        usage_raw = payload.get("usageMetadata")
        if isinstance(usage_raw, dict) and any(
            usage_raw.get(field) is not None
            for field in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
        ):
            # Chunks repeat usageMetadata; the final chunk is authoritative.
            # A MAX_TOKENS truncation can omit candidatesTokenCount while
            # still reporting the other counts (live-verified 2026-08-08).
            self.usage = Usage(
                input_tokens=usage_raw.get("promptTokenCount"),
                output_tokens=usage_raw.get("candidatesTokenCount"),
                cached_input_tokens=usage_raw.get("cachedContentTokenCount"),
                reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                total_tokens=usage_raw.get("totalTokenCount"),
            )
            self.usage_reported = True

        if isinstance(candidate, dict) and candidate.get("finishReason"):
            # Not terminal yet: a trailing chunk can still carry the final
            # usageMetadata (live-verified 2026-08-08), so the stream close
            # completes the response via on_close().
            self.finish_reason = str(candidate["finishReason"])
        return events

    def on_close(self) -> list[StreamEvent]:
        if self.finish_reason is None:
            return []
        self.saw_terminal = True
        return [StreamFinish(finish_reason=self.finish_reason, usage=self.usage)]


class GeminiAdapter(ProviderAdapter):
    def build_stream_spec(self, request: TextGenerationRequest) -> RequestSpec:
        spec = self.build_generation_spec(request)
        return RequestSpec(
            method=spec.method,
            path=spec.path.replace(":generateContent", ":streamGenerateContent"),
            params={**dict(spec.params), "alt": "sse"},
            json_body=spec.json_body,
        )

    def stream_assembler(self, request: TextGenerationRequest) -> StreamAssembler:
        return _GeminiStreamAssembler(self.resolved, request)

    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"], params={"pageSize": _PAGE_SIZE})

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise KeyCallError(
                "model list response missing 'models' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = []
        for entry in payload["models"]:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            model_id = _strip_prefix(str(entry["name"]))
            methods = entry.get("supportedGenerationMethods")
            categories: set[ModelCategory] = set()
            source = "keycall_rule"
            if isinstance(methods, list):
                for method in methods:
                    category = _METHOD_CATEGORIES.get(str(method))
                    if category is not None:
                        categories.add(category)
                if categories:
                    source = "provider_metadata"
            # generateContent is the transport, not a modality claim: the TTS
            # and image variants use it to return audio/images and reject a
            # TEXT response. When the identifier names a distinctive
            # non-text modality, it wins over the generic mapping.
            rule_category = classify_model_id(model_id)
            if rule_category in _DISTINCTIVE_MODALITIES and (
                ModelCategory.TEXT_GENERATION in categories
            ):
                categories.discard(ModelCategory.TEXT_GENERATION)
                categories.add(rule_category)
                source = "provider_metadata+keycall_rule"
            elif any(
                family in model_id.lower()
                for family in self.resolved.capabilities.non_text_model_families
            ):
                categories.discard(ModelCategory.TEXT_GENERATION)
                categories.add(ModelCategory.UNKNOWN)
                source = "provider_metadata+keycall_rule"
            if not categories:
                categories = {rule_category}
            models.append(
                Model(
                    id=model_id,
                    provider=self.resolved.provider,
                    categories=frozenset(categories),
                    display_name=entry.get("displayName"),
                    context_limit=context_limit(entry),
                    classification_source=source,
                )
            )
        next_spec = None
        token = payload.get("nextPageToken")
        if token:
            op = self.resolved.operations["list_models"]
            next_spec = RequestSpec(
                method=op["method"],
                path=op["path"],
                params={"pageSize": _PAGE_SIZE, "pageToken": str(token)},
            )
        return models, next_spec

    def build_image_spec(self, request: Any) -> RequestSpec:
        # Gemini's image models take the ordinary generateContent path and
        # answer with an inlineData part instead of text (verified
        # 2026-08-10), so there is no separate image endpoint to call.
        op = self.resolved.operations["image_generation"]
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{model}", quote(_strip_prefix(request.model), safe="")),
            json_body={
                "contents": [{"role": "user", "parts": [{"text": request.prompt}]}]
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
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
        parts = candidate.get("content", {}).get("parts", []) if isinstance(candidate, dict) else []
        images = []
        commentary = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData")
            if isinstance(inline, dict) and inline.get("data"):
                images.append((str(inline["data"]), str(inline.get("mimeType", "image/png"))))
            elif part.get("text"):
                commentary.append(str(part["text"]))
        usage_raw = payload.get("usageMetadata") or {} if isinstance(payload, dict) else {}
        if not images and commentary:
            # The model answered in words instead of drawing, which is how a
            # refusal or a clarifying question arrives. Repeat what it said
            # rather than reporting a bare "no image".
            raise KeyCallError(
                "model returned no image, only text: " + " ".join(commentary)[:300],
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.IMAGE_GENERATION.value,
            )
        warnings: tuple[str, ...] = ()
        if commentary:
            # The model sometimes narrates alongside the picture; surface it
            # rather than dropping it, since a refusal arrives this way.
            warnings = (f"model also returned text: {' '.join(commentary)[:200]}",)
        return self.image_result(
            images,
            usage=Usage(
                input_tokens=usage_raw.get("promptTokenCount"),
                output_tokens=usage_raw.get("candidatesTokenCount"),
                total_tokens=usage_raw.get("totalTokenCount"),
            ),
            model=_strip_prefix(str(payload.get("modelVersion", model))),
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(payload.get("responseId")),
            warnings=warnings,
        )

    def build_speech_spec(self, request: Any) -> RequestSpec:
        # Same generateContent path as text and image generation, telling
        # the model to answer in audio instead of text (verified live
        # 2026-08-12). No separate speech endpoint exists.
        op = self.resolved.operations["speech_generation"]
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if request.voice:
            generation_config["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}
            }
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{model}", quote(_strip_prefix(request.model), safe="")),
            json_body={
                "contents": [{"role": "user", "parts": [{"text": request.text}]}],
                "generationConfig": generation_config,
            },
        )

    def parse_speech_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
        parts = candidate.get("content", {}).get("parts", []) if isinstance(candidate, dict) else []
        audio: tuple[str, str] | None = None
        commentary = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData")
            if audio is None and isinstance(inline, dict) and inline.get("data"):
                # Raw PCM, not a playable container (audio/L16;codec=pcm;
                # rate=24000 as of 2026-08-12) — a caller that wants a
                # standard file has to wrap it in a header themselves.
                # KeyCall passes the provider's bytes and its own stated
                # media type through unchanged rather than guessing a
                # format the provider didn't send.
                audio = (str(inline["data"]), str(inline.get("mimeType", "audio/L16")))
            elif part.get("text"):
                commentary.append(str(part["text"]))
        usage_raw = payload.get("usageMetadata") or {} if isinstance(payload, dict) else {}
        if audio is None:
            # The model answered in words instead of speaking, which is how
            # a refusal or a clarifying question arrives — same posture as
            # the image-generation path.
            raise KeyCallError(
                "model returned no audio, only text: " + " ".join(commentary)[:300]
                if commentary
                else "model returned no audio",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.SPEECH_GENERATION.value,
            )
        warnings: tuple[str, ...] = ()
        if commentary:
            warnings = (f"model also returned text: {' '.join(commentary)[:200]}",)
        data, media_type = audio
        return self.speech_result(
            base64_data=data,
            media_type=media_type,
            usage=Usage(
                input_tokens=usage_raw.get("promptTokenCount"),
                output_tokens=usage_raw.get("candidatesTokenCount"),
                total_tokens=usage_raw.get("totalTokenCount"),
            ),
            model=_strip_prefix(str(payload.get("modelVersion", model))),
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(payload.get("responseId")),
            warnings=warnings,
        )

    def build_video_start_spec(self, request: Any) -> RequestSpec:
        # Veo renders through :predictLongRunning — a job-starting verb,
        # not generateContent. The immediate answer is only an operation
        # name; everything else arrives by polling it (verified live
        # 2026-08-13).
        op = self.resolved.operations["video_generation"]
        parameters: dict[str, Any] = {}
        if request.duration_seconds is not None:
            parameters["durationSeconds"] = request.duration_seconds
        if request.aspect_ratio:
            parameters["aspectRatio"] = request.aspect_ratio
        body: dict[str, Any] = {"instances": [{"prompt": request.prompt}]}
        if parameters:
            body["parameters"] = parameters
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{model}", quote(_strip_prefix(request.model), safe="")),
            json_body=body,
        )

    def parse_video_start(self, payload: Any, *, model: str) -> VideoJob:
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name:
            raise KeyCallError(
                "provider did not return an operation name for the video job",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.VIDEO_GENERATION.value,
            )
        return VideoJob(provider=self.resolved.provider, model=model, job_id=name)

    def build_video_status_spec(self, job: VideoJob) -> RequestSpec:
        op = self.resolved.operations["video_status"]
        # The job id is the operation resource path the provider returned
        # (models/{model}/operations/{id}); slashes are structure, the
        # rest is escaped so a hostile id can't rewrite the URL.
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{operation_name}", quote(job.job_id, safe="/")),
        )

    def parse_video_status(self, payload: Any, *, job: VideoJob) -> VideoJob:
        data = payload if isinstance(payload, dict) else {}
        if not data.get("done"):
            # A running operation answers with no `done` field at all,
            # not `done: false` (observed live 2026-08-13).
            return job
        error = data.get("error")
        if isinstance(error, dict):
            # Job failure is an HTTP 200: the operation finished, the
            # render did not. High-demand refusals (code 14) and internal
            # errors (code 13) both arrive here, observed live 2026-08-13.
            message = str(error.get("message", "video generation failed"))[:300]
            return VideoJob(
                provider=job.provider,
                model=job.model,
                job_id=job.job_id,
                status="failed",
                provider_status=f"error code {error.get('code')}",
                error_message=message,
            )
        samples = (
            data.get("response", {}).get("generateVideoResponse", {}).get("generatedSamples")
        )
        sample = samples[0] if isinstance(samples, list) and samples else {}
        uri = sample.get("video", {}).get("uri") if isinstance(sample, dict) else None
        if not isinstance(uri, str) or not uri:
            raise KeyCallError(
                "video operation finished without an error or a video URI",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.VIDEO_GENERATION.value,
            )
        return VideoJob(
            provider=job.provider,
            model=job.model,
            job_id=job.job_id,
            status="succeeded",
            provider_status="done",
            video_url=uri,
        )

    def video_download_plan(self, job: VideoJob) -> DownloadPlan:
        # The finished file lives on the provider's own API host and needs
        # the API key header; the first request answers a same-origin 302
        # whose target also needs the header (both verified live
        # 2026-08-13). Files are kept about two days.
        host = urlsplit(self.resolved.base_url).hostname or ""
        return DownloadPlan(
            url=job.video_url or "",
            allowed_hosts=(host,),
            send_credential=True,
            allow_same_origin_redirect=True,
        )

    def build_embedding_spec(self, request: Any) -> RequestSpec:
        op = self.resolved.operations["embeddings"]
        model = request.model if request.model.startswith("models/") else f"models/{request.model}"
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{model}", quote(_strip_prefix(request.model), safe="")),
            json_body={
                "requests": [
                    {"model": model, "content": {"parts": [{"text": text}]}}
                    for text in request.inputs
                ]
            },
        )

    def parse_embedding_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
        expected: int,
    ) -> InvocationResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), list):
            raise KeyCallError(
                "embedding response missing 'embeddings' array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.EMBEDDING.value,
            )
        vectors = [
            tuple(float(v) for v in entry.get("values", ()))
            for entry in payload["embeddings"]
        ]
        # batchEmbedContents reports no token usage of its own.
        return self.embedding_result(
            vectors,
            usage=Usage(),
            model=model,
            round_trip_duration_ms=round_trip_duration_ms,
            expected=expected,
        )

    def realtime_plan(self, config: Any) -> tuple[str, Any]:
        if not self.resolved.capabilities.realtime or "realtime" not in self.resolved.operations:
            return super().realtime_plan(config)
        from ._realtime import GeminiRealtimeTranslator

        # BidiGenerateContent's path carries no model; the setup frame
        # names it. The API key rides the x-goog-api-key header on the
        # handshake (verified live 2026-08-14), so it never enters a URL.
        path = self.resolved.operations["realtime"]["path"]
        return path, GeminiRealtimeTranslator(config, provider=self.resolved.provider)

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        self.validate_generation_request(request)
        op = self.resolved.operations["text_generation"]
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                system_texts.extend(
                    part.text for part in message.content if isinstance(part, TextInput)
                )
                continue
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextInput):
                    parts.append({"text": part.text})
                elif isinstance(part, (ImageInput, AudioInput, FileInput)):
                    # Pictures, audio and documents all ride the same
                    # inlineData part; only the media type differs. Bytes
                    # only: Gemini's fileData URI is for its own Files API,
                    # not an arbitrary web URL, and the gate already refused
                    # a URL for this provider.
                    kind = {
                        ImageInput: "image",
                        AudioInput: "audio",
                        FileInput: "file",
                    }[type(part)]
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": media_type_for(
                                    part, kind=kind, provider="gemini"
                                ),
                                "data": base64.b64encode(part.data or b"").decode(),
                            }
                        }
                    )
                elif isinstance(part, ToolCall):
                    call: dict[str, Any] = {"name": part.name, "args": dict(part.arguments)}
                    wire: dict[str, Any] = {"functionCall": call}
                    if part.opaque:
                        echo = json.loads(part.opaque)
                        # Gemini 400s when thoughtSignature is missing from
                        # replayed functionCall parts (live-verified
                        # 2026-08-08); it must ride back verbatim.
                        if echo.get("thoughtSignature"):
                            wire["thoughtSignature"] = echo["thoughtSignature"]
                        if echo.get("id"):
                            call["id"] = echo["id"]
                    parts.append(wire)
                elif isinstance(part, ToolResult):
                    content = part.content
                    if isinstance(content, str):
                        # functionResponse requires an object.
                        try:
                            parsed = json.loads(content)
                        except ValueError:
                            parsed = None
                        content = parsed if isinstance(parsed, dict) else {"output": content}
                    parts.append(
                        {"functionResponse": {"name": part.name, "response": dict(content)}}
                    )
            contents.append(
                {
                    # Gemini's assistant role is "model".
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": parts,
                }
            )
        if not contents:
            raise KeyCallError(
                "gemini requires at least one non-system message",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        body: dict[str, Any] = {"contents": contents}
        if system_texts:
            body["systemInstruction"] = {"parts": [{"text": text} for text in system_texts]}
        generation_config: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.reasoning_effort is not None:
            # Gemini's control is thinkingLevel, an uppercase enum
            # (live-verified 2026-08-14: LOW and HIGH accepted, and
            # thought-token counts follow the level).
            generation_config["thinkingConfig"] = {
                "thinkingLevel": request.reasoning_effort.upper()
            }
        if request.response_schema is not None:
            # Live-verified 2026-08-06: standard lowercase JSON Schema type
            # names ("object", "string", ...) work directly, no dialect
            # conversion to Gemini's uppercase OpenAPI-subset spelling needed.
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = dict(request.response_schema)
        if generation_config:
            body["generationConfig"] = generation_config
        tools: list[dict[str, Any]] = []
        if request.tools:
            tools.append(
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": dict(tool.input_schema),
                        }
                        for tool in request.tools
                    ]
                }
            )
        if request.web_search:
            tools.append({"google_search": {}})
        if tools:
            body["tools"] = tools
        tool_config: dict[str, Any] = {}
        if request.tools and request.web_search:
            # Gemini rejects mixing functionDeclarations with built-in tools
            # unless this flag is set (live-verified 2026-08-08).
            tool_config["includeServerSideToolInvocations"] = True
        if request.tool_choice is not None:
            mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}[request.tool_choice]
            tool_config["functionCallingConfig"] = {"mode": mode}
        if tool_config:
            body["toolConfig"] = tool_config
        path = op["path"].format(model=quote(_strip_prefix(request.model), safe=""))
        return RequestSpec(method=op["method"], path=path, json_body=body)

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        if not isinstance(payload, dict):
            raise KeyCallError(
                "generation response was not a JSON object",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.TEXT_GENERATION.value,
            )
        parts: list[OutputPart] = []
        warnings: list[str] = []
        citations: list[Citation] = []
        finish_reason = None
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                finish_reason = candidate.get("finishReason")
                content = candidate.get("content")
                if isinstance(content, dict):
                    for part in content.get("parts", []):
                        if not isinstance(part, dict):
                            continue
                        if "text" in part:
                            parts.append(TextOutput(text=str(part["text"])))
                        elif "functionCall" in part and isinstance(part["functionCall"], dict):
                            call = part["functionCall"]
                            echo = {
                                key: value
                                for key, value in (
                                    ("thoughtSignature", part.get("thoughtSignature")),
                                    ("id", call.get("id")),
                                )
                                if value
                            }
                            parts.append(
                                ToolCall(
                                    id=str(call.get("id", "")),
                                    name=str(call.get("name", "")),
                                    arguments=self.parse_tool_arguments(call.get("args", {})),
                                    opaque=json.dumps(echo) if echo else None,
                                )
                            )
                        else:
                            kind = next(iter(part.keys()), "?")
                            parts.append(UnknownOutput(provider_kind=str(kind)))
                grounding = candidate.get("groundingMetadata")
                if isinstance(grounding, dict):
                    # web.uri is a vertexaisearch.cloud.google.com redirect,
                    # by Google's design; title names the original source.
                    for chunk in grounding.get("groundingChunks") or []:
                        web = chunk.get("web") if isinstance(chunk, dict) else None
                        if isinstance(web, dict) and web.get("uri"):
                            citations.append(
                                Citation(url=str(web["uri"]), title=web.get("title"))
                            )

        usage_raw = payload.get("usageMetadata")
        if isinstance(usage_raw, dict):
            usage = Usage(
                input_tokens=usage_raw.get("promptTokenCount"),
                output_tokens=usage_raw.get("candidatesTokenCount"),
                cached_input_tokens=usage_raw.get("cachedContentTokenCount"),
                reasoning_tokens=usage_raw.get("thoughtsTokenCount"),
                total_tokens=usage_raw.get("totalTokenCount"),
            )
        else:
            usage = Usage()
            warnings.append("provider reported no usage information")

        return InvocationResult(
            provider=self.resolved.provider,
            model=_strip_prefix(str(payload.get("modelVersion", model))),
            operation=Operation.TEXT_GENERATION,
            parts=tuple(parts),
            usage=usage,
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=safe_request_id(payload.get("responseId")),
            finish_reason=str(finish_reason) if finish_reason else None,
            citations=dedupe_citations(citations),
            warnings=tuple(warnings),
        )

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        message = ""
        status_name = ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = str(payload["error"].get("message", ""))
            status_name = str(payload["error"].get("status", ""))
        # Gemini reports an invalid key as 400 INVALID_ARGUMENT with an
        # API_KEY_INVALID detail, not as a 401.
        if status_code in (401, 403) or "API key" in message or "API_KEY" in message:
            code = (
                ErrorCode.PERMISSION_DENIED
                if status_name == "PERMISSION_DENIED"
                else ErrorCode.INVALID_API_KEY
            )
            return code, False, message or "invalid API key"
        if status_code == 402:
            # Same posture as the base adapter and Anthropic: valid key,
            # unfunded account, not a malformed response.
            return ErrorCode.PERMISSION_DENIED, False, message or "payment required"
        if status_code == 429 or status_name == "RESOURCE_EXHAUSTED":
            return ErrorCode.RATE_LIMITED, True, message or "rate or quota limited"
        if status_name == "NOT_FOUND" or status_code == 404:
            if "no longer available" in message:
                # Gemini keeps retired models in its list response and
                # withdraws them per account, ahead of the published
                # shutdown date: as of 2026-08-09 a new key is refused
                # gemini-2.5-* ("no longer available to new users") whose
                # documented shutdown is still months out, and the whole
                # 2.0-flash family (shut down 2026-06-01) is still listed.
                # The list alone can't tell a caller which models they can
                # actually invoke, so say what does.
                guidance = (
                    f"{message} (gemini lists models an account cannot invoke and "
                    "gives no lifecycle field to filter on, so this one came back "
                    "from its own model list; the 'latest' aliases, "
                    "gemini-flash-latest, gemini-flash-lite-latest and "
                    "gemini-pro-latest, track google's current models and survive "
                    "these retirements)"
                )
                return ErrorCode.MODEL_NOT_AVAILABLE, False, guidance
            return ErrorCode.MODEL_NOT_AVAILABLE, False, message or "model not found"
        if status_name in ("UNAVAILABLE", "DEADLINE_EXCEEDED") or status_code >= 500:
            return ErrorCode.PROVIDER_UNAVAILABLE, True, message or "provider unavailable"
        return (
            ErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
            message or f"unexpected status {status_code}",
        )
