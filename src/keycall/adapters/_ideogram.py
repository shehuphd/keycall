"""Ideogram adapter: pictures and video on a model-in-the-path wire.

Every endpoint is ``POST /v2/{content}/{action}/{model}``, so there is no
request field naming the model and no endpoint listing them. The catalog
carries the model list and, per model, a route for each operation it
serves: the path, the body encoding, and the inputs that route honours.
This adapter reads those rows; adding a model or an endpoint is a catalog
row. Results come back as expiring links, which the client downloads, and
some routes answer with a job to poll. Wire behaviour live-verified
2026-10-02.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._mask import mask_to_alpha_edit, mask_to_black_edit, read_mask
from .._sanitize import safe_request_id
from .._transport import DownloadPlan, RequestSpec
from .._types import (
    ImageGenerationRequest,
    ImageInput,
    ImageOperationRequest,
    InvocationResult,
    Model,
    TextBlockOutput,
    TextGenerationRequest,
    TextOutput,
    ToolJob,
    Usage,
    VideoGenerationRequest,
    VideoJob,
)
from ._base import (
    PendingImageDownloads,
    PendingImageJob,
    ProviderAdapter,
    image_media_type,
)

# The filename a multipart file part carries, by sniffed media type.
# Ideogram reads the bytes, so the name is a courtesy to its logs.
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}

# How a mask is redrawn for a route, by the route's own convention.
# KeyCall's convention (white marks the area to change) is the "white"
# case and passes through after a read that validates it.
_MASK_EDIT = ("white", "black", "transparent")

Part = tuple[str, "str | None", bytes, "str | None"]


class IdeogramAdapter(ProviderAdapter):
    """Pictures and video only: no text generation. The model listing is
    the catalog's, and a model outside it is refused before any request,
    naming the models that serve the operation asked for."""

    # --- discovery ---

    def initial_list_request(self) -> RequestSpec:
        # No list endpoint exists. A dry-run generate is the free call
        # that proves the key: it validates and prices the request with
        # nothing generated or billed, and answers 401 to a bad key.
        op = self.resolved.operations["list_models"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            params={"dry_run": "true"},
            json_body={"prompt": "key check"},
        )

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        # The response is discarded: reaching a 2xx proves the credential
        # works, which is all this call can establish.
        if not isinstance(payload, dict):
            raise KeyCallError(
                "the key check did not answer with JSON",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation="list_models",
            )
        models = [
            Model(
                id=str(entry["id"]),
                provider=self.resolved.provider,
                categories=frozenset(
                    ModelCategory(category) for category in entry.get("categories", ())
                ),
                capabilities=frozenset(str(name) for name in entry.get("routes", {})),
                classification_source="keycall_catalog",
                warnings=("ideogram has no model-list endpoint; the list is maintained by KeyCall",),
            )
            for entry in self.resolved.catalog_models
        ]
        return models, None

    # --- text generation: refused ---

    def _refuse_text(self) -> KeyCallError:
        return KeyCallError(
            f"provider {self.resolved.provider!r} makes pictures and video and has no "
            "text-generation API; use generate_image(), edit_image(), or describe_image()",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.TEXT_GENERATION.value,
        )

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        raise self._refuse_text()

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        raise self._refuse_text()

    # --- routing ---

    def _models_serving(self, operation: str, param: str | None = None) -> list[str]:
        found = []
        for entry in self.resolved.catalog_models:
            route = entry.get("routes", {}).get(operation)
            if route is None:
                continue
            if param is None or param in route.get("params", ()):
                found.append(str(entry["id"]))
        return found

    def _route(self, model: str, operation: Operation) -> dict[str, Any]:
        """The catalog route for a model and operation. A model the
        catalog doesn't list, or one that doesn't serve the operation, is
        refused naming the models that do."""
        serving = self._models_serving(operation.value)
        for entry in self.resolved.catalog_models:
            if entry["id"] != model:
                continue
            route = entry.get("routes", {}).get(operation.value)
            if route is None:
                raise KeyCallError(
                    f"model {model!r} has no {operation.value.replace('_', ' ')} "
                    f"endpoint on {self.resolved.provider}; models that do: "
                    + ", ".join(serving),
                    code=ErrorCode.MODEL_NOT_SUITABLE,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            return dict(route)
        raise KeyCallError(
            f"{self.resolved.provider} has no model {model!r}; models serving "
            f"{operation.value.replace('_', ' ')}: " + ", ".join(serving),
            code=ErrorCode.MODEL_NOT_AVAILABLE,
            provider=self.resolved.provider,
            operation=operation.value,
        )

    def _require_route_params(
        self, route: Mapping[str, Any], model: str, operation: Operation, given: Sequence[str]
    ) -> None:
        """Refuse an input this model's route doesn't honour. Ideogram
        accepts and ignores an unknown field (a seed on remix/ideogram-4
        prices normally, live 2026-10-02), so without this gate the caller
        would get a picture that silently disregards what they asked for."""
        for name in given:
            if name in route.get("params", ()):
                continue
            others = self._models_serving(operation.value, name)
            raise KeyCallError(
                f"model {model!r} does not take {name} on "
                f"{operation.value.replace('_', ' ')}, and it changes the result, "
                f"so it isn't dropped. Models that take it: "
                + (", ".join(others) or "none on this provider"),
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=operation.value,
            )

    def _require_quality(
        self, route: Mapping[str, Any], model: str, operation: Operation, quality: str
    ) -> None:
        values = route.get("quality_values")
        if values and quality not in values:
            raise KeyCallError(
                f"model {model!r} takes quality " + ", ".join(values) + f", not {quality!r}",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )

    # --- request bodies ---

    def _file_part(self, field: str, image: ImageInput, operation: Operation) -> Part:
        if image.data is None:
            raise KeyCallError(
                f"provider {self.resolved.provider!r} takes pictures as bytes, not as a "
                "URL; download the picture and pass ImageInput(data=...)",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        media_type = image_media_type(image, provider=self.resolved.provider)
        extension = _EXTENSIONS.get(media_type, "bin")
        return (field, f"{field}.{extension}", image.data, media_type)

    def _option_parts(
        self,
        options: Mapping[str, Any] | None,
        *,
        taken: Sequence[str],
        operation: Operation,
    ) -> tuple[dict[str, Any], list[Part]]:
        """provider_options split into plain fields and file parts. A value
        that is an ImageInput, or a sequence of them, travels as files. A
        name KeyCall already set from one of its own parameters is refused:
        two spellings of one setting have no defined winner."""
        fields: dict[str, Any] = {}
        files: list[Part] = []
        for name, value in (options or {}).items():
            if name in taken:
                raise KeyCallError(
                    f"provider_options sets {name!r}, which KeyCall already sends from "
                    "its own parameter; set it in one place",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            if isinstance(value, ImageInput):
                files.append(self._file_part(name, value, operation))
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(item, ImageInput) for item in value)
            ):
                files.extend(self._file_part(name, item, operation) for item in value)
            else:
                fields[name] = value
        return fields, files

    def _spec(
        self,
        route: Mapping[str, Any],
        fields: Mapping[str, Any],
        files: Sequence[Part],
        *,
        operation: Operation,
    ) -> RequestSpec:
        if files and route["body"] != "multipart":
            raise KeyCallError(
                "this endpoint takes a JSON body and cannot carry a picture file",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        if route.get("async"):
            # The route answers at once with a job to poll, so a slow
            # render never holds the connection past the read timeout.
            fields = {**fields, "async": True}
        if route["body"] == "json":
            return RequestSpec(method="POST", path=route["path"], json_body=dict(fields))
        parts: list[Part] = []
        for name, value in fields.items():
            # A list value repeats the field, which is how a multipart
            # form spells an array (colorways' `colors`).
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(item, bool):
                    text = "true" if item else "false"
                elif isinstance(item, (dict, list)):
                    text = json.dumps(item)
                else:
                    text = str(item)
                parts.append((name, None, text.encode("utf-8"), None))
        parts.extend(files)
        return RequestSpec(method="POST", path=route["path"], multipart=tuple(parts))

    def _mask_part(
        self, route: Mapping[str, Any], mask: ImageInput, operation: Operation
    ) -> Part:
        if mask.data is None:
            raise KeyCallError(
                "the mask must be passed as bytes, ImageInput(data=...)",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        provider = self.resolved.provider
        convention = route.get("mask_edit", "white")
        if convention == "black":
            data = mask_to_black_edit(mask.data, provider=provider, operation=operation.value)
        elif convention == "transparent":
            data = mask_to_alpha_edit(mask.data, provider=provider, operation=operation.value)
        else:
            # Already this route's convention; the read still refuses a
            # mask that isn't a readable PNG before it is sent.
            read_mask(mask.data, provider=provider, operation=operation.value)
            data = mask.data
        field = str(route["mask_field"])
        return (field, f"{field}.png", data, "image/png")

    # --- image generation ---

    def build_image_spec(self, request: ImageGenerationRequest) -> RequestSpec:
        operation = Operation.IMAGE_GENERATION
        route = self._route(request.model, operation)
        given = [
            name for name in ("size", "quality", "seed") if getattr(request, name) is not None
        ]
        self.image_operation_support(operation, given)
        self._require_route_params(route, request.model, operation, given)
        fields: dict[str, Any] = {"prompt": request.prompt}
        if request.size is not None:
            name, value = self._size_field(route, request.model, operation, request.size)
            fields[name] = value
        if request.quality is not None:
            self._require_quality(route, request.model, operation, request.quality)
            fields["quality"] = request.quality
        if request.seed is not None:
            fields["seed"] = request.seed
        extra, files = self._option_parts(
            request.provider_options, taken=tuple(fields), operation=operation
        )
        fields.update(extra)
        return self._spec(route, fields, files, operation=operation)

    def parse_image_response(  # type: ignore[override]
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult | PendingImageJob | PendingImageDownloads:
        return self._parse_pictures(
            payload, headers=headers, model=model, operation=Operation.IMAGE_GENERATION
        )

    # --- picture operations ---

    def build_image_operation_spec(self, request: ImageOperationRequest) -> RequestSpec:
        operation = request.operation
        route = self._route(request.model, operation)
        given = [
            name
            for name in ("mask", "seed", "quality", "factor")
            if getattr(request, name) is not None
        ]
        if request.reference_images:
            given.append("reference_images")
        if request.size is not None and operation is not Operation.IMAGE_EXPAND:
            given.append("size")
        if request.mask is not None:
            for name in route.get("masked_excludes", ()):
                if name in given:
                    raise KeyCallError(
                        f"model {request.model!r} keeps the source picture's size on a "
                        f"masked edit, so it takes no {name} with a mask",
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        provider=self.resolved.provider,
                        operation=operation.value,
                    )
        # A prompt is an optional input only where the operation doesn't
        # require one (upscale, layerize), so it is gated only there.
        if request.prompt is not None and operation in (
            Operation.IMAGE_UPSCALE,
            Operation.IMAGE_LAYERIZE,
            Operation.IMAGE_EXPAND,
        ):
            given.append("prompt")
        # A mask is the operation's own input on an erase, not an option.
        gated = [
            name for name in given if not (name == "mask" and operation is Operation.OBJECT_ERASE)
        ]
        self.image_operation_support(operation, gated)
        self._require_route_params(route, request.model, operation, gated)

        fields: dict[str, Any] = {}
        files: list[Part] = []
        if request.prompt is not None:
            fields["prompt"] = request.prompt
        if request.quality is not None:
            self._require_quality(route, request.model, operation, request.quality)
            fields["quality"] = request.quality
        if request.seed is not None:
            fields["seed"] = request.seed
        if request.factor is not None:
            values = route.get("factor_values", {})
            wire = values.get(str(request.factor))
            if wire is None:
                raise KeyCallError(
                    f"model {request.model!r} upscales by "
                    + ", ".join(sorted(values, key=int))
                    + f", not {request.factor}",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            fields[str(route["factor_field"])] = wire
        if request.size is not None:
            name, value = self._size_field(route, request.model, operation, request.size)
            fields[name] = value

        files.append(self._file_part(str(route["image_field"]), request.image, operation))
        limit = route.get("max_reference_images")
        if request.mask is not None and route.get("mask_takes_reference_slot") and limit:
            limit -= 1
        if limit is not None and len(request.reference_images) > limit:
            masked = " with a mask" if request.mask is not None else ""
            raise KeyCallError(
                f"model {request.model!r} takes at most {limit} reference image(s){masked}; "
                f"{len(request.reference_images)} were given",
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        for reference in request.reference_images:
            files.append(self._file_part(str(route["reference_field"]), reference, operation))
        path = route["path"]
        if request.mask is not None:
            files.append(self._mask_part(route, request.mask, operation))
            # ideogram-3 edits through two endpoints: remix repaints the
            # whole picture, inpaint the masked area.
            path = route.get("masked_path", path)
        extra, option_files = self._option_parts(
            request.provider_options,
            taken=(*fields, *(part[0] for part in files)),
            operation=operation,
        )
        fields.update(extra)
        files.extend(option_files)
        return self._spec({**route, "path": path}, fields, files, operation=operation)

    def _size_field(
        self, route: Mapping[str, Any], model: str, operation: Operation, size: str
    ) -> tuple[str, str]:
        """The route's own field and spelling for a size. A route takes a
        ratio, a pixel size, or both; the form it can't take is refused
        naming the one it can."""
        spec = route.get("size") or {}
        is_ratio = ":" in size
        if is_ratio and "ratio_field" in spec:
            return str(spec["ratio_field"]), size.replace(":", str(spec.get("ratio_separator", ":")))
        if not is_ratio and "pixel_field" in spec:
            return str(spec["pixel_field"]), size
        wanted = (
            'a pixel size like "1280x768"' if "pixel_field" in spec else 'an aspect ratio like "16:9"'
        )
        raise KeyCallError(
            f"model {model!r} takes size as {wanted}, not {size!r}",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=operation.value,
        )

    def parse_image_operation(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        request: ImageOperationRequest,
    ) -> InvocationResult | PendingImageJob | PendingImageDownloads:
        request_id = safe_request_id(
            headers.get(self.resolved.provider_request_id_header or "")
        )
        if request.operation is Operation.IMAGE_DESCRIPTION:
            return self._description_result(
                payload,
                model=request.model,
                round_trip_duration_ms=round_trip_duration_ms,
                provider_request_id=request_id,
            )
        return self._parse_pictures(
            payload, headers=headers, model=request.model, operation=request.operation
        )

    def _description_result(
        self,
        payload: Any,
        *,
        model: str,
        round_trip_duration_ms: float,
        provider_request_id: str | None,
    ) -> InvocationResult:
        """describe answers in one of two forms: ideogram-3 with a list of
        plain descriptions, ideogram-4 with a structured prompt whose
        high_level_description is the plain-language summary. The summary
        is the text; the structured prompt is kept whole as JSON in a
        second part, since it is what ideogram-4 takes back as a prompt."""
        data = payload if isinstance(payload, dict) else {}
        parts: list[Any] = []
        descriptions = data.get("descriptions")
        if isinstance(descriptions, list):
            for entry in descriptions:
                if isinstance(entry, dict) and entry.get("text"):
                    parts.append(TextOutput(text=str(entry["text"])))
        structured = data.get("json_prompt")
        if isinstance(structured, dict):
            summary = structured.get("high_level_description")
            if isinstance(summary, str) and summary:
                parts.append(TextOutput(text=summary))
        if not parts:
            raise KeyCallError(
                "provider returned no description for the picture",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.IMAGE_DESCRIPTION.value,
            )
        return InvocationResult(
            provider=self.resolved.provider,
            model=model,
            operation=Operation.IMAGE_DESCRIPTION,
            parts=tuple(parts[:1]),
            usage=Usage(),
            round_trip_duration_ms=round_trip_duration_ms,
            provider_request_id=provider_request_id,
        )

    def _parse_pictures(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        model: str,
        operation: Operation,
    ) -> PendingImageJob | PendingImageDownloads:
        """A synchronous answer carries ``data``; a job-only route answers
        with a ``generation_id`` alone, to poll."""
        request_id = safe_request_id(
            headers.get(self.resolved.provider_request_id_header or "")
        )
        data = payload if isinstance(payload, dict) else {}
        entries = data.get("data")
        if isinstance(entries, list):
            return self._downloads(
                entries, model=model, operation=operation, provider_request_id=request_id
            )
        job_id = data.get("generation_id")
        if isinstance(job_id, str) and job_id:
            return PendingImageJob(
                job_id=job_id, model=model, operation=operation, provider_request_id=request_id
            )
        raise KeyCallError(
            "provider answered with neither pictures nor a job to poll",
            code=ErrorCode.INVALID_PROVIDER_RESPONSE,
            provider=self.resolved.provider,
            operation=operation.value,
        )

    def _plan(self, url: str) -> DownloadPlan:
        # A signed link on a catalog-pinned host, fetched without the key
        # (live 2026-10-02): the credential never travels to it, and a
        # response naming any other host is refused before a request.
        return DownloadPlan(
            url=url,
            allowed_hosts=self.resolved.image_download_hosts,
            send_credential=False,
            allow_same_origin_redirect=False,
        )

    def _downloads(
        self,
        entries: Sequence[Any],
        *,
        model: str,
        operation: Operation,
        provider_request_id: str | None,
        usage: Usage | None = None,
    ) -> PendingImageDownloads:
        plans: list[DownloadPlan] = []
        labels: list[str | None] = []
        blocks: list[Any] = []
        unsafe = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("is_image_safe") is False:
                unsafe += 1
                continue
            url = entry.get("url")
            if isinstance(url, str) and url:
                plans.append(self._plan(url))
                labels.append("design" if operation is Operation.IMAGE_LAYERIZE else None)
            base_url = entry.get("base_image_url")
            if isinstance(base_url, str) and base_url:
                plans.append(self._plan(base_url))
                labels.append("base")
            for block in entry.get("text_blocks") or ():
                if isinstance(block, dict) and "text" in block:
                    blocks.append(_text_block(block))
        if not plans:
            if unsafe:
                raise KeyCallError(
                    "the provider's safety review withheld every picture this "
                    "request produced (is_image_safe: false); rephrase the "
                    "prompt or change the source picture",
                    code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            raise KeyCallError(
                "provider returned no picture",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        warnings = (
            (f"the provider's safety review withheld {unsafe} of the pictures",)
            if unsafe
            else ()
        )
        return PendingImageDownloads(
            plans=tuple(plans),
            labels=tuple(labels),
            model=model,
            operation=operation,
            usage=usage or Usage(),
            provider_request_id=provider_request_id,
            extra_parts=tuple(blocks),
            warnings=warnings,
        )

    # --- jobs ---

    def _status_spec(self, job_id: str) -> RequestSpec:
        op = self.resolved.operations["generation_status"]
        return RequestSpec(
            method=op["method"],
            path=op["path"].replace("{generation_id}", quote(job_id, safe="")),
        )

    def build_image_job_status_spec(self, pending: PendingImageJob) -> RequestSpec:
        return self._status_spec(pending.job_id)

    def parse_image_job_status(
        self,
        payload: Any,
        *,
        headers: Mapping[str, str],
        round_trip_duration_ms: float,
        pending: PendingImageJob,
    ) -> InvocationResult | PendingImageJob | PendingImageDownloads:
        data = payload if isinstance(payload, dict) else {}
        status = str(data.get("status", ""))
        if status in ("", "pending"):
            return pending
        if status == "completed":
            cost = data.get("usage_cost_usd_micros")
            usage = (
                Usage(provider_units=(("cost_usd_micros", float(cost)),))
                if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                else None
            )
            return self._downloads(
                data.get("data") or (),
                model=pending.model,
                operation=pending.operation,
                provider_request_id=pending.provider_request_id,
                usage=usage,
            )
        reason = str(data.get("failure_reason") or "no reason given")[:300]
        raise KeyCallError(
            f"the provider's job ended as {status}: {reason}",
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            provider=self.resolved.provider,
            operation=pending.operation.value,
        )

    # --- provider tools ---

    def build_tool_start_spec(
        self, row: Mapping[str, Any], inputs: Mapping[str, Any]
    ) -> RequestSpec:
        """A tool's inputs checked against its catalog row before any
        request: a missing required input, an unknown name, or too many
        pictures for one input is refused naming what the tool takes."""
        operation = Operation.PROVIDER_TOOL
        name = str(row["name"])
        files_allowed: Mapping[str, int] = row.get("files") or {}
        fields_allowed = set(row.get("fields") or ())
        accepted = sorted({*files_allowed, *fields_allowed})
        if not isinstance(inputs, Mapping):
            raise TypeError("start_tool inputs must be a mapping of input name to value")
        missing = [field for field in row.get("required", ()) if inputs.get(field) is None]
        if missing:
            raise KeyCallError(
                f"tool {name!r} needs " + ", ".join(missing) + "; it takes: " + ", ".join(accepted),
                code=ErrorCode.UNSUPPORTED_OPERATION,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        fields: dict[str, Any] = {}
        files: list[Part] = []
        for key, value in inputs.items():
            if value is None:
                continue
            if key in files_allowed:
                pictures = list(value) if isinstance(value, (list, tuple)) else [value]
                if not all(isinstance(item, ImageInput) for item in pictures):
                    raise TypeError(f"tool input {key!r} takes ImageInput values")
                if len(pictures) > int(files_allowed[key]):
                    raise KeyCallError(
                        f"tool {name!r} takes at most {files_allowed[key]} picture(s) "
                        f"for {key!r}; {len(pictures)} were given",
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        provider=self.resolved.provider,
                        operation=operation.value,
                    )
                files.extend(self._file_part(key, item, operation) for item in pictures)
            elif key in fields_allowed:
                fields[key] = value
            else:
                raise KeyCallError(
                    f"tool {name!r} takes no input {key!r}; it takes: " + ", ".join(accepted),
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
        return self._spec(
            {"path": row["path"], "body": "multipart"}, fields, files, operation=operation
        )

    def parse_tool_start(self, payload: Any, *, tool: str) -> ToolJob:
        job_id = payload.get("generation_id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise KeyCallError(
                "provider did not return a generation_id for the tool run",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.PROVIDER_TOOL.value,
            )
        return ToolJob(provider=self.resolved.provider, tool=tool, job_id=job_id)

    def build_tool_status_spec(self, job: ToolJob) -> RequestSpec:
        return self._status_spec(job.job_id)

    def parse_tool_status(self, payload: Any, *, job: ToolJob) -> ToolJob:
        data = payload if isinstance(payload, dict) else {}
        status = str(data.get("status", ""))
        if status in ("", "pending"):
            return job
        if status == "completed":
            urls = tuple(
                str(entry["url"])
                for entry in data.get("data") or ()
                if isinstance(entry, dict)
                and entry.get("url")
                and entry.get("is_image_safe") is not False
            )
            if not urls:
                return ToolJob(
                    provider=job.provider,
                    tool=job.tool,
                    job_id=job.job_id,
                    status="failed",
                    provider_status=status,
                    error_message=(
                        "the tool finished but the provider's safety review "
                        "withheld every picture"
                    ),
                )
            return ToolJob(
                provider=job.provider,
                tool=job.tool,
                job_id=job.job_id,
                status="succeeded",
                provider_status=status,
                output_urls=urls,
            )
        return ToolJob(
            provider=job.provider,
            tool=job.tool,
            job_id=job.job_id,
            status="failed",
            provider_status=status,
            error_message=str(data.get("failure_reason") or "the tool run failed")[:300],
        )

    def tool_downloads(self, job: ToolJob) -> PendingImageDownloads:
        return PendingImageDownloads(
            plans=tuple(self._plan(url) for url in job.output_urls),
            model=job.tool,
            operation=Operation.PROVIDER_TOOL,
        )

    # --- video ---

    def _video_route(self, request: VideoGenerationRequest) -> tuple[dict[str, Any], str]:
        operation = Operation.VIDEO_GENERATION
        routes = self._route(request.model, operation)
        if request.reference_images:
            kind, needed = "references", "reference_images"
        elif request.image is not None:
            kind, needed = "image", "image"
        else:
            kind, needed = "text", "a prompt alone"
        route = routes.get(kind)
        if route is None:
            others = [
                str(entry["id"])
                for entry in self.resolved.catalog_models
                if kind in entry.get("routes", {}).get(operation.value, {})
            ]
            raise KeyCallError(
                f"model {request.model!r} does not make video from {needed}; "
                "models that do: " + (", ".join(others) or "none on this provider"),
                code=ErrorCode.MODEL_NOT_SUITABLE,
                provider=self.resolved.provider,
                operation=operation.value,
            )
        return dict(route), kind

    def build_video_start_spec(self, request: VideoGenerationRequest) -> RequestSpec:
        operation = Operation.VIDEO_GENERATION
        route, kind = self._video_route(request)
        fields: dict[str, Any] = {"prompt": request.prompt}
        if request.duration_seconds is not None:
            fields["duration"] = request.duration_seconds
        if request.aspect_ratio:
            if kind == "image" and not route.get("aspect_ratio"):
                raise KeyCallError(
                    f"model {request.model!r} takes no aspect_ratio with a first frame: "
                    "the video keeps the picture's own shape",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            # Ideogram spells a ratio 16x9; KeyCall's callers write 16:9,
            # as every other video provider does.
            fields["aspect_ratio"] = request.aspect_ratio.replace(":", "x")
        files: list[Part] = []
        if request.image is not None:
            files.append(self._file_part(str(route["image_field"]), request.image, operation))
        if request.last_frame is not None:
            field = route.get("last_frame_field")
            if field is None:
                raise KeyCallError(
                    f"model {request.model!r} takes no last_frame",
                    code=ErrorCode.UNSUPPORTED_OPERATION,
                    provider=self.resolved.provider,
                    operation=operation.value,
                )
            files.append(self._file_part(str(field), request.last_frame, operation))
        for reference in request.reference_images:
            files.append(self._file_part(str(route["reference_field"]), reference, operation))
        return self._spec(route, fields, files, operation=operation)

    def parse_video_start(self, payload: Any, *, model: str) -> VideoJob:
        job_id = payload.get("generation_id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise KeyCallError(
                "provider did not return a generation_id for the video job",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.VIDEO_GENERATION.value,
            )
        return VideoJob(provider=self.resolved.provider, model=model, job_id=job_id)

    def build_video_status_spec(self, job: VideoJob) -> RequestSpec:
        return self._status_spec(job.job_id)

    def parse_video_status(self, payload: Any, *, job: VideoJob) -> VideoJob:
        data = payload if isinstance(payload, dict) else {}
        status = str(data.get("status", ""))
        if status in ("", "pending"):
            return job
        if status == "completed":
            url = next(
                (
                    entry.get("url")
                    for entry in data.get("data") or ()
                    if isinstance(entry, dict) and entry.get("url")
                ),
                None,
            )
            if not isinstance(url, str) or not url:
                raise KeyCallError(
                    "video job completed without a video URL",
                    code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                    provider=self.resolved.provider,
                    operation=Operation.VIDEO_GENERATION.value,
                )
            return VideoJob(
                provider=job.provider,
                model=job.model,
                job_id=job.job_id,
                status="succeeded",
                provider_status=status,
                video_url=url,
            )
        return VideoJob(
            provider=job.provider,
            model=job.model,
            job_id=job.job_id,
            status="failed",
            provider_status=status,
            error_message=str(data.get("failure_reason") or "video generation failed")[:300],
        )

    def video_download_plan(self, job: VideoJob) -> DownloadPlan:
        return DownloadPlan(
            url=job.video_url or "",
            allowed_hosts=self.resolved.video_download_hosts,
            send_credential=False,
            allow_same_origin_redirect=False,
        )

    # --- errors ---

    def translate_error(self, status_code: int, payload: Any) -> tuple[ErrorCode, bool, str]:
        # Ideogram's errors arrive in three bodies: {"error": "..."},
        # {"message": "..."}, and a problem document {"detail": "...",
        # "title": ...}. The base translator reads only the first, so its
        # status mapping is kept and the message rebuilt from whichever
        # field is present.
        code, retryable, message = super().translate_error(status_code, payload)
        provider_message = ""
        if isinstance(payload, dict):
            for name in ("error", "message", "detail"):
                value = payload.get(name)
                if isinstance(value, str) and value:
                    message = provider_message = value
                    break
            reason = payload.get("reject_reason")
            if isinstance(reason, str) and reason:
                message = f"{message} ({reason})"
                if reason == "inflight_limit":
                    code, retryable = ErrorCode.RATE_LIMITED, True
        if status_code == 401:
            # A paused key gets the same 401 and "Access denied" sentence
            # as a wrong one: Ideogram pauses every key on the account when
            # its balance reaches zero (observed 2026-10-02, cleared by a
            # top-up 2026-10-07), so the message names both causes.
            message = (
                f"{message.rstrip('.')}. Ideogram answers this way for a wrong or revoked key and "
                "also for a valid key paused because the account balance reached zero; "
                "check the key's status and the balance on the Ideogram dashboard"
            )
        elif status_code == 404:
            # The model is a path segment, so a model the provider doesn't
            # serve is an unknown URL. The catalog gate normally refuses
            # first; this covers a model the catalog lists and the
            # provider has since withdrawn.
            message = "the provider has no endpoint for this model and operation"
        elif status_code == 400:
            # A refused field or value: the request is at fault, and the
            # provider's sentence names what to change.
            code = ErrorCode.UNSUPPORTED_OPERATION
        elif status_code == 422:
            code = ErrorCode.UNSUPPORTED_OPERATION
            message = provider_message or "the prompt did not pass the provider's safety review"
        return code, retryable, message


def _text_block(block: Mapping[str, Any]) -> TextBlockOutput:
    def whole(name: str) -> int:
        value = block.get(name)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    angle = block.get("angle")
    size = block.get("font_size")
    return TextBlockOutput(
        text=str(block.get("text", "")),
        x=whole("x"),
        y=whole("y"),
        width=whole("width"),
        height=whole("height"),
        angle=float(angle) if isinstance(angle, (int, float)) else 0.0,
        alignment=str(block["alignment"]) if block.get("alignment") else None,
        formatting=tuple(str(item) for item in block.get("formatting") or ()),
        font_name=str(block["font_name"]) if block.get("font_name") else None,
        font_size=int(size) if isinstance(size, (int, float)) else None,
        color=str(block["color"]) if block.get("color") else None,
    )
