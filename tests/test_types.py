import pytest

from keycall import (
    EmbeddingOutput,
    InvocationResult,
    Message,
    Operation,
    TextGenerationRequest,
    TextInput,
    TextOutput,
    Usage,
)


def make_message(text="Hello"):
    return Message(role="user", content=[TextInput(text=text)])


def test_message_normalizes_content_to_tuple():
    msg = Message(role="user", content=[TextInput(text="hi")])
    assert isinstance(msg.content, tuple)


def test_message_rejects_bad_role():
    with pytest.raises(ValueError):
        Message(role="robot", content=[TextInput(text="hi")])


def test_message_rejects_empty_content():
    with pytest.raises(ValueError):
        Message(role="user", content=[])


def test_message_rejects_untyped_parts():
    with pytest.raises(TypeError):
        Message(role="user", content=["plain string"])
    with pytest.raises(TypeError):
        Message(role="user", content=[{"type": "text", "text": "hi"}])


def test_request_accepts_list_and_normalizes_to_tuple():
    request = TextGenerationRequest(model="m", messages=[make_message()])
    assert isinstance(request.messages, tuple)


def test_request_rejects_dict_messages():
    with pytest.raises(TypeError):
        TextGenerationRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def test_request_rejects_empty_messages_and_model():
    with pytest.raises(ValueError):
        TextGenerationRequest(model="m", messages=[])
    with pytest.raises(ValueError):
        TextGenerationRequest(model="", messages=[make_message()])


def test_request_is_frozen_and_carries_no_credential_fields():
    request = TextGenerationRequest(model="m", messages=[make_message()])
    with pytest.raises(AttributeError):
        request.model = "other"
    assert not hasattr(request, "api_key")
    assert not hasattr(request, "provider")


def test_usage_defaults_are_none_not_zero():
    usage = Usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def make_result(parts):
    return InvocationResult(
        provider="openai",
        model="m",
        operation=Operation.TEXT_GENERATION,
        parts=tuple(parts),
        usage=Usage(),
        round_trip_duration_ms=12.5,
    )


def test_result_text_concatenates_text_parts():
    result = make_result([TextOutput(text="Hello, "), TextOutput(text="world")])
    assert result.text == "Hello, world"


def test_result_text_is_none_without_text_parts():
    result = make_result([EmbeddingOutput(values=(0.1, 0.2))])
    assert result.text is None


def test_input_parts_carry_kind_discriminators():
    assert TextInput(text="x").kind == "text"
    assert TextOutput(text="x").kind == "text"
