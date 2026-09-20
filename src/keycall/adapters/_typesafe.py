"""TypeSafe adapter: the System One judgment wire.

One POST carries a state and every question about it at once; the answers
come back typed, with calibrated probabilities. Not an LLM vendor: text
generation and every other LLM operation refuse with a typed error. All
wire behavior live-verified 2026-09-20 against jev-1.13.0.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .._enums import ModelCategory, Operation
from .._errors import ErrorCode, KeyCallError
from .._sanitize import safe_request_id
from .._transport import RequestSpec
from .._types import (
    ChoiceAnswer,
    ChoiceQuestion,
    InvocationResult,
    JudgmentAnswer,
    JudgmentRequest,
    JudgmentResult,
    Model,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    TextGenerationRequest,
    Usage,
)
from ._base import ProviderAdapter

# Documented and live-enforced (400 "Too many choices", 2026-09-17): the
# per-Choice option cap. The rubric-length ceiling the docs mention (10)
# is not enforced here because it has never been observed on the wire.
_MAX_CHOICE_OPTIONS = 255


class TypeSafeAdapter(ProviderAdapter):
    """Judgments only. The model listing is live but not exhaustive: it
    returns rolling aliases (jev-latest, jev-preview), while explicit
    versioned ids (jev-1.13.0) are accepted on the wire without appearing
    there — so no gate here ever refuses a model for being unlisted."""

    def initial_list_request(self) -> RequestSpec:
        op = self.resolved.operations["list_models"]
        return RequestSpec(method=op["method"], path=op["path"])

    def parse_model_page(self, payload: Any) -> tuple[list[Model], RequestSpec | None]:
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise KeyCallError(
                "model list response carried no models array",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
            )
        models = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            released = None
            raw_date = entry.get("release_date")
            if isinstance(raw_date, str) and raw_date:
                try:
                    released = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    released = None
            models.append(
                Model(
                    id=str(entry["name"]),
                    provider=self.resolved.provider,
                    categories=frozenset({ModelCategory.DECISION}),
                    display_name=(
                        str(entry["description"]) if entry.get("description") else None
                    ),
                    released_at=released,
                    classification_source="keycall_rule",
                    warnings=(
                        (
                            "the listing carries rolling aliases only; explicit "
                            "versioned ids are accepted without appearing here"
                        ),
                    ),
                )
            )
        return models, None

    def _refuse_llm_operation(self) -> KeyCallError:
        return KeyCallError(
            f"provider {self.resolved.provider!r} is a judgment service "
            "with no text-generation API; use judge()",
            code=ErrorCode.UNSUPPORTED_OPERATION,
            provider=self.resolved.provider,
            operation=Operation.TEXT_GENERATION.value,
        )

    def build_generation_spec(self, request: TextGenerationRequest) -> RequestSpec:
        raise self._refuse_llm_operation()

    def parse_generation_response(
        self,
        payload: Any,
        *,
        headers: Any,
        round_trip_duration_ms: float,
        model: str,
    ) -> InvocationResult:
        raise self._refuse_llm_operation()

    # --- judgment ---

    def build_judgment_spec(self, request: JudgmentRequest) -> RequestSpec:
        questions: dict[str, dict[str, Any]] = {}
        for question_id, question in request.questions.items():
            if isinstance(question, NoulQuestion):
                questions[question_id] = {
                    "type": "noul",
                    "instructions": question.instructions,
                }
            elif isinstance(question, ChoiceQuestion):
                if len(question.options) > _MAX_CHOICE_OPTIONS:
                    raise KeyCallError(
                        f"question {question_id!r} has {len(question.options)} "
                        f"options; the provider takes at most {_MAX_CHOICE_OPTIONS} "
                        "per choice (enforced live 2026-09-17). Narrow the set "
                        "in two passes instead",
                        code=ErrorCode.UNSUPPORTED_OPERATION,
                        provider=self.resolved.provider,
                        operation=Operation.JUDGMENT.value,
                    )
                questions[question_id] = {
                    "type": "choice",
                    "instructions": question.instructions,
                    # A mapping of option name to description; a bare list
                    # is rejected with a 422 (live 2026-09-20), and empty
                    # descriptions are accepted.
                    "criteria": dict(question.options),
                }
            else:
                assert isinstance(question, ScoreQuestion)
                questions[question_id] = {
                    "type": "score",
                    "instructions": question.instructions,
                    "criteria": list(question.levels),
                }
        op = self.resolved.operations["judgment"]
        return RequestSpec(
            method=op["method"],
            path=op["path"],
            json_body={
                "state": request.state,
                "model": request.model,
                "questions": questions,
            },
        )

    def parse_judgment_response(
        self,
        payload: Any,
        *,
        headers: Any,
        round_trip_duration_ms: float,
    ) -> JudgmentResult:
        raw_answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(raw_answers, dict):
            raise KeyCallError(
                "judgment response carried no answers",
                code=ErrorCode.INVALID_PROVIDER_RESPONSE,
                provider=self.resolved.provider,
                operation=Operation.JUDGMENT.value,
            )
        answers: dict[str, JudgmentAnswer] = {}
        warnings: list[str] = []
        for question_id, entry in raw_answers.items():
            if not isinstance(entry, dict):
                warnings.append(f"answer {question_id!r} was not an object; dropped")
                continue
            kind = str(entry.get("type", ""))
            if kind == "noul":
                answers[question_id] = NoulAnswer(
                    probability=float(entry.get("noul", 0.0))
                )
            elif kind == "choice":
                raw_probabilities = entry.get("probabilities")
                answers[question_id] = ChoiceAnswer(
                    choice=str(entry.get("choice", "")),
                    probabilities={
                        str(option): float(value)
                        for option, value in (
                            raw_probabilities.items()
                            if isinstance(raw_probabilities, dict)
                            else ()
                        )
                    },
                    confidence=entry.get("confidence"),
                )
            elif kind == "score":
                # legend and probabilities arrive keyed by the stringified
                # 0-based level index; both are re-ordered by that index so
                # the tuples align with the rubric the caller sent.
                legend = entry.get("legend")
                legend = legend if isinstance(legend, dict) else {}
                raw_probabilities = entry.get("probabilities")
                probability_map = (
                    raw_probabilities if isinstance(raw_probabilities, dict) else {}
                )
                indices = sorted(
                    (key for key in legend if str(key).lstrip("-").isdigit()),
                    key=lambda key: int(key),
                )
                answers[question_id] = ScoreAnswer(
                    score=float(entry.get("score", 0.0)),
                    levels=tuple(str(legend[index]) for index in indices),
                    probabilities=tuple(
                        float(probability_map.get(index, 0.0)) for index in indices
                    ),
                    confidence=entry.get("confidence"),
                )
            else:
                # A type this build doesn't know (the gated bounding_box,
                # or something newer): surfaced, never silently invented.
                warnings.append(
                    f"answer {question_id!r} has unrecognized type {kind!r}; dropped"
                )
        raw_usage = payload.get("usage")
        usage = None
        if isinstance(raw_usage, dict):
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens"),
                output_tokens=raw_usage.get("output_tokens"),
            )
        processing = headers.get("x-envoy-upstream-service-time")
        request_id = None
        if self.resolved.provider_request_id_header:
            request_id = safe_request_id(
                headers.get(self.resolved.provider_request_id_header)
            )
        return JudgmentResult(
            model=str(payload.get("model", "")),
            answers=answers,
            usage=usage,
            provider_request_id=request_id,
            provider_processing_ms=(
                float(processing) if processing not in (None, "") else None
            ),
            round_trip_duration_ms=round_trip_duration_ms,
            warnings=tuple(warnings),
        )

    def translate_error(self, status_code: int, payload: Any) -> tuple[Any, bool, str]:
        # TypeSafe's errors ride a `detail` field in three forms: an
        # object ({error_type, message}) for usage errors, a pydantic list
        # ([{loc, msg}]) for 422 validation failures, and occasionally a
        # bare string. The base translator reads none of them, so its
        # status mapping is kept and the message rebuilt from the detail.
        code, retryable, message = super().translate_error(status_code, payload)
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            error_type = str(detail.get("error_type", ""))
            detail_message = str(detail.get("message", ""))
            message = (
                f"{error_type}: {detail_message}" if error_type else detail_message
            ) or message
            # "Unknown model" arrives as a 400 api_usage_error, not a 404,
            # so the base mapping would file it as a malformed response
            # and send the caller looking for a bug in their request.
            if status_code == 400 and detail_message.startswith("Unknown model"):
                code = ErrorCode.MODEL_NOT_AVAILABLE
        elif isinstance(detail, list):
            parts = []
            for entry in detail:
                if not isinstance(entry, dict):
                    continue
                location = ".".join(str(piece) for piece in entry.get("loc", ()))
                entry_message = str(entry.get("msg", ""))
                parts.append(f"{location}: {entry_message}" if location else entry_message)
            if parts:
                message = "; ".join(parts)
            if status_code == 422:
                # A validation refusal names the request field at fault:
                # the caller's ask is the problem, not the provider's
                # response, so INVALID_PROVIDER_RESPONSE would mislead.
                code = ErrorCode.UNSUPPORTED_OPERATION
        elif isinstance(detail, str) and detail:
            message = detail
        return code, retryable, message
