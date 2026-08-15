import pytest

from keycall import ModelCategory
from keycall._classify import classify_model_id

CASES = [
    ("gpt-4o", ModelCategory.TEXT_GENERATION),
    ("gpt-4o-mini", ModelCategory.TEXT_GENERATION),
    ("chatgpt-4o-latest", ModelCategory.TEXT_GENERATION),
    ("o3-mini", ModelCategory.TEXT_GENERATION),
    ("claude-sonnet-4-5", ModelCategory.TEXT_GENERATION),
    ("gemini-2.5-flash", ModelCategory.TEXT_GENERATION),
    ("deepseek-reasoner", ModelCategory.TEXT_GENERATION),
    ("sonar-pro", ModelCategory.TEXT_GENERATION),
    ("kimi-k2", ModelCategory.TEXT_GENERATION),
    ("text-embedding-3-large", ModelCategory.EMBEDDING),
    ("whisper-1", ModelCategory.TRANSCRIPTION),
    ("gpt-4o-transcribe", ModelCategory.TRANSCRIPTION),
    ("tts-1-hd", ModelCategory.SPEECH_GENERATION),
    ("gpt-4o-mini-tts", ModelCategory.SPEECH_GENERATION),
    ("gpt-4o-realtime-preview", ModelCategory.REALTIME),
    ("grok-voice-latest", ModelCategory.REALTIME),
    ("dall-e-3", ModelCategory.IMAGE_GENERATION),
    ("gpt-image-1", ModelCategory.IMAGE_GENERATION),
    ("imagen-3.0-generate-002", ModelCategory.IMAGE_GENERATION),
    ("sora-2", ModelCategory.VIDEO_GENERATION),
    ("veo-3.0-generate-preview", ModelCategory.VIDEO_GENERATION),
    # Conservative unknowns: never guessed into the text picker.
    ("omni-moderation-latest", ModelCategory.UNKNOWN),
    ("gpt-4o-audio-preview", ModelCategory.UNKNOWN),
    ("totally-new-model-family", ModelCategory.UNKNOWN),
    ("davinci-002", ModelCategory.UNKNOWN),
]


@pytest.mark.parametrize(("model_id", "expected"), CASES)
def test_classification_rules(model_id, expected):
    assert classify_model_id(model_id) is expected
