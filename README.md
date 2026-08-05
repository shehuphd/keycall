# KeyCall

One consistent interface for validating AI-provider API keys, listing and filtering the models available to them, and making normalized calls, so every product stops rebuilding the same model-picker filters and provider wrappers.

**Status: pre-alpha scaffold.** The public types, provider registry, and security boundaries are in place; the adapter and transport layer (live model listing and text generation) is under construction. Nothing here makes a network call yet.

## What v1 will do

```python
from keycall import KeyCall, Message, ModelCategory, TextInput

with KeyCall(provider="openai", api_key=secret) as client:
    discovery = client.list_models(categories={ModelCategory.TEXT_GENERATION})

    result = client.generate_text(
        model=discovery.models[0].id,
        messages=[Message(role="user", content=[TextInput(text="Hello.")])],
    )

print(result.text)
print(result.usage.total_tokens)
print(result.round_trip_duration_ms)
```

- **Explicit provider, always.** KeyCall never guesses which vendor issued a key and never sends a credential to more than the one provider you name.
- **No credential storage.** Keys live in memory for the client's lifetime, wrapped in a redacting type that keeps them out of reprs, logs, traces, exceptions, and pickles. Your app decides how to store them.
- **Model filtering built in.** Text-generation models by default; embeddings, image, audio, and other categories on request; unknown models never silently enter the default picker.
- **Typed errors.** Invalid key, rate limit, provider outage, timeout, and malformed response are distinguishable, never collapsed into "invalid key."

## Provider support (v1 targets)

Live-verified 2026-08-05 (one model-list call plus one bounded generation per provider):

| Provider | Protocol | Listing | Generation |
|---|---|---|---|
| OpenAI | openai | verified | verified |
| Anthropic | anthropic | verified | verified |
| Google Gemini | gemini | verified | verified |
| DeepSeek | openai-compatible | verified | verified |
| Perplexity | openai-compatible | verified | verified |
| Moonshot/Kimi | openai-compatible | verified | verified |
| Custom endpoint (explicit `base_url`) | openai-compatible | fixtures only | fixtures only |

Two provider quirks worth knowing, both handled:

**Gemini** keeps retired models in its list endpoint (`gemini-2.5-flash` returns
"no longer available to new users") with no lifecycle field to pre-filter on,
and meters quota per model and tier, so one model's 429 says nothing about the
next. Its `supportedGenerationMethods` is also a transport signal rather than a
modality claim: TTS variants advertise `generateContent` and then refuse a text
response, so KeyCall lets a distinctive identifier modality outrank it.

**Perplexity**'s `GET /v1/models` is scoped to the Agent API and returns
vendor-prefixed router models (`anthropic/...`, `perplexity/sonar`) that the
Sonar route rejects. Sonar's own models are not API-discoverable, so KeyCall
maintains them in its catalog and uses the list call purely as a credential
check.

Because of quirks like these, `keycall verify --generate` walks the filtered
models in provider order and prints the outcome of every attempt until one
succeeds, so drift stays visible rather than being masked by a silent retry.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Author

Built by [Mo Shehu](https://mohammedshehu.com).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
