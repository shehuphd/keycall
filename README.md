# KeyCall

One consistent interface for validating AI-provider API keys, listing and filtering the models available to them, and making normalized calls, so every product stops rebuilding the same model-picker filters and provider wrappers.

**Status: early release.** Key validation, model listing and filtering,
text generation, streaming, tool calling, native web search with normalized
citations, structured JSON output, and image, audio, and document input all
work and are live-verified against every provider that supports them. The API
is settled but may still shift before 1.0.

Docs: [USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md) for the full API and CLI reference · [ARCHITECTURE.md](https://github.com/shehuphd/keycall/blob/main/ARCHITECTURE.md) for the layer diagram and component contracts · [CHANGELOG.md](https://github.com/shehuphd/keycall/blob/main/CHANGELOG.md) for version history.

## Quick start

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
- **Streaming.** `stream_text()` yields typed events (text increments, citations, tool calls, finish) across all four wire protocols, and refuses to call a stream complete without the provider's own terminal signal.
- **Tool calling.** Define tools once and KeyCall normalizes all four call/result wire shapes, streamed or not, carrying the provider echo data some models require back verbatim. It never executes a tool.
- **Images, audio, and documents.** Pass bytes (or a URL where the provider fetches one) beside your text; KeyCall maps each provider's shape and detects the media type from the content. Support varies by provider and by form, so a refusal happens before the network and names who does accept that kind.
- **Web search with citations.** `web_search=True` turns on the provider's native search tool (OpenAI, Anthropic, Gemini; Perplexity always searches) and returns sources normalized to one `Citation` shape.
- **Structured output.** `response_schema=<JSON Schema>` is enforced provider-side on OpenAI, Anthropic, Gemini, Moonshot, and Perplexity; on providers without enforcement (DeepSeek, unverified custom targets) KeyCall falls back to guaranteed-valid-JSON mode and adds a result warning rather than claiming a guarantee it can't back. `result.text` is always the JSON string, regardless of which mechanism produced it.
- **Hardened transport.** TLS always verified, redirects refused, response sizes capped, SSRF and DNS-rebinding guards on custom endpoints, and generation is never silently retried.

## Provider support

Live-verified 2026-08-09. Every release re-runs a model list, a bounded generation,
a stream, a full tool round (streamed and not), and an image read against each
provider that supports them, and blocks publishing if any of it fails:

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

**Gemini** keeps retired models in its list endpoint with no lifecycle field to
pre-filter on, and withdraws them per account ahead of the published shutdown
date: on 2026-08-09 the first six text models it advertised to a new key were
all refused, `gemini-2.5-*` with "no longer available to new users" months
before its documented shutdown. KeyCall tries the provider's maintained
`-latest` aliases first, so verification lands on a model that works instead of
walking a list of withdrawn ones, and the error for a retired model names those
aliases. It also meters quota per model and tier, so one model's 429 says
nothing about the next. Its `supportedGenerationMethods` is a transport signal
rather than a modality claim: TTS variants advertise `generateContent` and then
refuse a text response, and so do the Interactions-only, computer-use, and
music families, so KeyCall lets the identifier outrank it and keeps those out
of the default text picker.

**Perplexity**'s `GET /v1/models` is scoped to the Agent API and returns
vendor-prefixed router models (`anthropic/...`, `perplexity/sonar`) that the
Sonar route rejects. Sonar's own models are not API-discoverable, so KeyCall
maintains them in its catalog and uses the list call purely as a credential
check. Note the version prefix: the unversioned `https://api.perplexity.ai/models`
returns 404 for every key, valid or not, so anything validating a key against
that path rejects good credentials. `/v1/models` answers 401 for a bad key and
200 for a good one, which is what makes it usable as a check (verified
2026-08-09).

### Structured output notes, per provider

- **OpenAI** requires `additionalProperties: false` on every object level of
  the schema for its strict `json_schema` mode, or the request 400s. This is
  an OpenAI requirement, not a KeyCall one — write schemas with it from the
  start.
- **Anthropic** implements structured output by forcing a single synthetic
  tool call; it cannot be combined with `web_search=True` in the same
  request (forcing one tool prevents the model calling a different one), and
  KeyCall rejects that combination before any network call.
- **Gemini**'s equivalent combination (`web_search=True` with
  `response_schema`) is not gated — no live-verified evidence either way
  that Gemini rejects it, so KeyCall passes it through rather than guessing.
- **DeepSeek** hard-requires the literal word "json" somewhere in the prompt
  for its `json_object` fallback mode, or it 400s. KeyCall detects this and
  injects a short system instruction automatically when needed, and always
  says so via a result warning.
- **Moonshot/Kimi** reasoning-capable models can spend the entire
  `max_output_tokens` budget on a visible reasoning trace and never emit a
  final answer if the budget is too small. KeyCall detects the resulting
  empty-content-with-reasoning-trace response and adds a warning rather than
  returning a silent empty result; give these models a larger budget than
  you'd expect a short answer to need.

Because of quirks like these, `keycall verify --generate` walks the filtered
models in provider order and prints the outcome of every attempt until one
succeeds, so drift stays visible rather than being masked by a silent retry.

## Local viewer

```bash
keycall view --source ./keys.toml
```

Opens a token-protected local web app over your loaded targets: a dashboard
with live key checks, a sortable model browser with category filters, a
playground for real generation calls (web search included), and a verify
report. Keys never leave the server process and never appear in the browser.

Or double-click / run a launcher from a fresh clone — it creates the venv,
installs KeyCall, finds your key file, and starts the viewer:
`launch.command` (macOS), `launch.sh` (Linux/macOS), `launch.bat` (Windows).

## Verifying keys from the command line

```bash
keycall verify --source ./keys.toml
```

```bash
keycall verify --source ./keys.toml --generate
```

`--generate` also makes one small bounded call per target. Sources can be TXT,
JSON, or TOML, an explicit `env:VAR_NAME` reference, or an interactive prompt.
See `keycall-test-keys.example.toml` for the format and
[USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md#the-verify-cli) for the full reference. Keys never appear
in output, and KeyCall never writes to or deletes your credential file.

## Installation

```bash
pip install keycall
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Author

Built by [Mo Shehu](https://mohammedshehu.com).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
