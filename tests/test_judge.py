"""Judgments: typed questions about a state, typed answers with
calibrated probabilities.

Live-probed 2026-09-20 against api.typesafe.ai (jev-1.13.0 behind the
jev-latest alias). One POST /v1/systemone carries the state and every
question at once; answers come back per question id — noul as a bare
0-1 probability, choice with per-option probabilities and a confidence,
score with an index-keyed legend and probability map that normalize back
into rubric order. The model listing returns rolling aliases only, so
nothing here refuses a model for being unlisted. TypeSafe is the one
provider with the endpoint; everyone else refuses toward it.
"""

import httpx
import pytest

from keycall import (
    AsyncKeyCall,
    ChoiceAnswer,
    ChoiceQuestion,
    ErrorCode,
    JudgmentRequest,
    KeyCall,
    KeyCallError,
    ModelCategory,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    classify_model_id,
)

CANARY = "sk-canary-judge-key"


def make_client(provider, handler):
    return KeyCall(
        provider=provider, api_key=CANARY, httpx_transport=httpx.MockTransport(handler)
    )


def refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request expected, got {request.method} {request.url}")


THREE_QUESTIONS = {
    "is_urgent": NoulQuestion(instructions="Does this convey urgency?"),
    "route": ChoiceQuestion(
        instructions="Which team should handle this?",
        options={
            "billing": "payment and payout problems",
            "technical": "bugs and outages",
            "retention": "customers threatening to leave",
        },
    ),
    "anger": ScoreQuestion(
        instructions="How angry is the customer?",
        levels=["calm", "frustrated", "angry", "furious"],
    ),
}


def judgment_response(**overrides):
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.95},
            "route": {
                "type": "choice",
                "choice": "retention",
                "confidence": 0.34,
                "probabilities": {"billing": 0.43, "retention": 0.57, "technical": 0.0},
            },
            "anger": {
                "type": "score",
                "score": 1.81,
                "confidence": 0.78,
                "legend": {"0": "calm", "1": "frustrated", "2": "angry", "3": "furious"},
                "probabilities": {"0": 0.0, "1": 0.2, "2": 0.79, "3": 0.01},
            },
        },
        "usage": {"input_tokens": 413, "output_tokens": 75},
    }
    body.update(overrides)
    return body


# --- gates and validation, all before the network ---


def test_public_exports_are_reachable_from_the_package():
    import keycall

    for name in (
        "NoulQuestion",
        "ChoiceQuestion",
        "ScoreQuestion",
        "NoulAnswer",
        "ChoiceAnswer",
        "ScoreAnswer",
        "JudgmentRequest",
        "JudgmentResult",
    ):
        assert name in keycall.__all__
        assert getattr(keycall, name) is not None


def test_jev_ids_classify_as_decision_models():
    assert classify_model_id("jev-latest") is ModelCategory.DECISION
    assert classify_model_id("jev-1.13.0") is ModelCategory.DECISION


def test_judgmentless_provider_refuses_and_names_the_supporting_one():
    client = make_client("openai", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.judge(model="jev-latest", state="x", questions=THREE_QUESTIONS)
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "typesafe" in info.value.message


def test_text_generation_on_typesafe_points_at_judge():
    from keycall import Message, TextInput

    client = make_client("typesafe", refuse_network)
    with pytest.raises(KeyCallError) as info:
        client.generate_text(
            model="jev-latest",
            messages=[Message(role="user", content=[TextInput(text="hi")])],
        )
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "judge" in info.value.message


def test_question_validation_rejects_malformed_asks():
    with pytest.raises(ValueError):
        NoulQuestion(instructions="  ")
    with pytest.raises(ValueError):
        ChoiceQuestion(instructions="pick", options={"only": ""})
    with pytest.raises(ValueError):
        ChoiceQuestion(instructions="pick", options={"a": "", " ": ""})
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="rate", levels=["only one"])
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="rate", levels=["low", " "])


def test_request_validation_rejects_malformed_asks():
    question = NoulQuestion(instructions="ok?")
    with pytest.raises(ValueError):
        JudgmentRequest(model="jev-latest", state="", questions={"q": question})
    with pytest.raises(ValueError):
        JudgmentRequest(model="jev-latest", state=None, questions={"q": question})
    with pytest.raises(ValueError):
        JudgmentRequest(model="jev-latest", state="x", questions={})
    with pytest.raises(ValueError):
        JudgmentRequest(model="jev-latest", state="x", questions={" ": question})
    with pytest.raises(TypeError):
        JudgmentRequest(model="jev-latest", state="x", questions={"q": "not typed"})
    with pytest.raises(ValueError):
        JudgmentRequest(model=" ", state="x", questions={"q": question})


def test_oversized_choice_refuses_before_any_call():
    client = make_client("typesafe", refuse_network)
    too_many = ChoiceQuestion(
        instructions="pick", options={f"option-{i}": "" for i in range(256)}
    )
    with pytest.raises(KeyCallError) as info:
        client.judge(model="jev-latest", state="x", questions={"pick": too_many})
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "255" in info.value.message


# --- the wire ---


def test_judgment_posts_state_and_every_question_at_once():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization", "")
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=judgment_response(),
            headers={
                "x-typesafe-request-id": "req_abc123",
                "x-envoy-upstream-service-time": "69",
            },
        )

    client = make_client("typesafe", handler)
    result = client.judge(
        model="jev-latest",
        state="Customer message: payouts failing for 3 days.",
        questions=THREE_QUESTIONS,
    )

    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["method"] == "POST"
    assert captured["auth"] == f"Bearer {CANARY}"

    body = captured["body"]
    assert body["model"] == "jev-latest"
    assert body["state"] == "Customer message: payouts failing for 3 days."
    assert set(body["questions"]) == {"is_urgent", "route", "anger"}
    assert body["questions"]["is_urgent"] == {
        "type": "noul",
        "instructions": "Does this convey urgency?",
    }
    # Choice criteria go as a mapping of option name to description (a
    # bare list is rejected with a 422 on the live wire).
    assert body["questions"]["route"]["type"] == "choice"
    assert body["questions"]["route"]["criteria"] == {
        "billing": "payment and payout problems",
        "technical": "bugs and outages",
        "retention": "customers threatening to leave",
    }
    # Score criteria go as the ordered level list.
    assert body["questions"]["anger"]["criteria"] == [
        "calm",
        "frustrated",
        "angry",
        "furious",
    ]

    # The resolved id, not the alias that was sent.
    assert result.model == "jev-1.13.0"
    assert result.usage is not None
    assert result.usage.input_tokens == 413 and result.usage.output_tokens == 75
    assert result.provider_request_id == "req_abc123"
    assert result.provider_processing_ms == 69.0
    assert result.round_trip_duration_ms is not None
    assert result.warnings == ()

    urgent = result.answers["is_urgent"]
    assert isinstance(urgent, NoulAnswer) and urgent.probability == 0.95

    route = result.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "retention"
    assert route.confidence == 0.34
    assert route.probabilities["billing"] == 0.43

    anger = result.answers["anger"]
    assert isinstance(anger, ScoreAnswer)
    assert anger.score == 1.81
    assert anger.confidence == 0.78
    # Legend and probabilities re-ordered by index into rubric order.
    assert anger.levels == ("calm", "frustrated", "angry", "furious")
    assert anger.probabilities == (0.0, 0.2, 0.79, 0.01)


def test_state_passes_through_as_json_when_structured():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=judgment_response())

    client = make_client("typesafe", handler)
    client.judge(
        model="jev-latest",
        state={"ticket": {"messages": [{"text": "help"}]}},
        questions={"q": NoulQuestion(instructions="Is `ticket.messages[0].text` a plea?")},
    )
    assert captured["body"]["state"] == {"ticket": {"messages": [{"text": "help"}]}}


def test_unrecognized_answer_type_is_surfaced_not_invented():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=judgment_response(
                answers={
                    "q": {"type": "noul", "noul": 0.5},
                    "boxes": {"type": "bounding_box", "boxes": []},
                }
            ),
        )

    client = make_client("typesafe", handler)
    result = client.judge(
        model="jev-latest",
        state="x",
        questions={"q": NoulQuestion(instructions="ok?")},
    )
    assert set(result.answers) == {"q"}
    assert result.warnings and "bounding_box" in result.warnings[0]


def test_response_without_answers_is_a_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev-1.13.0"})

    client = make_client("typesafe", handler)
    with pytest.raises(KeyCallError) as info:
        client.judge(
            model="jev-latest",
            state="x",
            questions={"q": NoulQuestion(instructions="ok?")},
        )
    assert info.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


# --- error translation (the measured shapes) ---


def test_bad_key_maps_to_invalid_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "detail": {
                    "error_type": "authentication_error",
                    "message": "Invalid API key.",
                }
            },
        )

    client = make_client("typesafe", handler)
    with pytest.raises(KeyCallError) as info:
        client.judge(
            model="jev-latest",
            state="x",
            questions={"q": NoulQuestion(instructions="ok?")},
        )
    assert info.value.code is ErrorCode.INVALID_API_KEY
    assert "authentication_error" in info.value.message


def test_unknown_model_maps_to_model_not_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "detail": {
                    "error_type": "api_usage_error",
                    "message": "Unknown model: jev",
                }
            },
        )

    client = make_client("typesafe", handler)
    with pytest.raises(KeyCallError) as info:
        client.judge(
            model="jev", state="x", questions={"q": NoulQuestion(instructions="ok?")}
        )
    assert info.value.code is ErrorCode.MODEL_NOT_AVAILABLE
    assert "Unknown model" in info.value.message


def test_validation_failure_surfaces_the_field_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "questions", "pick", "choice", "criteria"],
                        "msg": "Field required",
                    }
                ]
            },
        )

    client = make_client("typesafe", handler)
    with pytest.raises(KeyCallError) as info:
        client.judge(
            model="jev-latest",
            state="x",
            questions={"q": NoulQuestion(instructions="ok?")},
        )
    assert info.value.code is ErrorCode.UNSUPPORTED_OPERATION
    assert "questions.pick.choice.criteria" in info.value.message
    assert "Field required" in info.value.message


def test_gated_question_type_error_carries_the_error_type():
    # The bounding_box organisation gate: its message text drifted between
    # probes ("not enabled for your organization" on 2026-09-17, a generic
    # "Invalid request." on 2026-09-20), so status and error_type are the
    # assertion, never the sentence.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": {"error_type": "api_usage_error", "message": "Invalid request."}},
        )

    client = make_client("typesafe", handler)
    with pytest.raises(KeyCallError) as info:
        client.judge(
            model="jev-latest",
            state="x",
            questions={"q": NoulQuestion(instructions="ok?")},
        )
    assert info.value.status_code == 400
    assert "api_usage_error" in info.value.message


# --- model listing and verify ---


def models_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "jev-latest",
                        "description": "The latest Jev",
                        "release_date": "2026-09-10T18:38:01.391457+00:00",
                    },
                    {
                        "name": "jev-preview",
                        "description": "A preview of jev-latest",
                        "release_date": "2026-09-10T18:39:06.057655+00:00",
                    },
                ]
            },
        )
    return httpx.Response(
        200,
        json=judgment_response(
            answers={"check": {"type": "noul", "noul": 0.99}}
        ),
    )


def test_model_list_carries_decision_models_with_the_not_exhaustive_warning():
    client = make_client("typesafe", models_handler)
    discovery = client.list_models(categories={ModelCategory.DECISION})
    ids = [model.id for model in discovery.models]
    assert ids == ["jev-latest", "jev-preview"]
    first = discovery.models[0]
    assert first.categories == frozenset({ModelCategory.DECISION})
    assert first.released_at is not None
    assert any("aliases only" in warning for warning in first.warnings)


def test_verify_generate_walks_one_minimal_noul():
    from keycall._sources import Target
    from keycall._verify_core import run_verify

    client = make_client("typesafe", models_handler)
    result = run_verify(
        Target(provider="typesafe", key=CANARY),
        generate=True,
        client=client,
    )
    assert result.outcome == "judged"
    assert result.generate_ok
    assert result.decision_model_count == 2
    assert result.attempts[0].model_id == "jev-latest"
    assert result.attempts[0].ok


@pytest.mark.anyio
async def test_async_judgment_matches_the_sync_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=judgment_response())

    client = AsyncKeyCall(
        provider="typesafe",
        api_key=CANARY,
        httpx_transport=httpx.MockTransport(handler),
    )
    result = await client.judge(
        model="jev-latest",
        state="Customer message: payouts failing for 3 days.",
        questions=THREE_QUESTIONS,
    )
    assert result.model == "jev-1.13.0"
    assert isinstance(result.answers["is_urgent"], NoulAnswer)
    await client.close()
