# KeyCall

One consistent interface for validating AI-provider API keys, listing and filtering the models available to them, and making normalized calls, so every product stops rebuilding the same model-picker filters and provider wrappers.

**Status: early release.** Key validation, model listing and filtering,
text generation, streaming, tool calling, native web search with normalized
citations, structured JSON output, embeddings, image generation, and image,
audio, and document input all work and are live-verified against every
provider that supports them. The API
is settled but may still shift before 1.0.

Docs: [USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md) for the full API and CLI reference · [ARCHITECTURE.md](https://github.com/shehuphd/keycall/blob/main/ARCHITECTURE.md) for the layer diagram and component contracts · [CHANGELOG.md](https://github.com/shehuphd/keycall/blob/main/CHANGELOG.md) for version history.

## Install

```bash
pip install keycall
```

## See it work in 30 seconds

No config file, no signup, nothing to write. Export a key you already have
and point KeyCall at the variable:

```bash
export OPENAI_API_KEY=...
```

```bash
keycall verify --provider openai --source env:OPENAI_API_KEY --generate
```

```
✓ OPENAI_API_KEY (openai): key accepted, 79 text model(s), list digest 6d356bc3f4c24389, selection rule v4
✓ OPENAI_API_KEY: generated with gpt-5.6-luna (filtered position 0, provider-list position 123, 830 ms, total tokens: 18)
```

That is one command telling you the key is valid, how many text models it can
reach, and that a generation came back — including which model answered and
where it sat in the provider's own list. Swap `openai` for
`anthropic`, `gemini`, `deepseek`, `perplexity`, or `moonshot` and it works the
same way.

Prefer to click around? Same key, one word different:

```bash
keycall view --provider openai --source env:OPENAI_API_KEY
```

That opens a local web app in your browser with your key already loaded: a
dashboard that checks it live, a browsable model list with category filters,
and a Playground where you can chat, show a model a picture, record a voice
message straight from the page, attach a PDF, offer it a tool, or have it draw
you something. Keys stay in the local server process and never reach the
browser.

Got several keys? Put them in a file and load them all at once — see
[`keycall-test-keys.example.toml`](https://github.com/shehuphd/keycall/blob/main/keycall-test-keys.example.toml)
for the format:

```bash
keycall view --source ./keys.toml
```

From a fresh clone with no Python set up at all, double-click a launcher
instead: `launch.command` (macOS), `launch.sh` (Linux/macOS), `launch.bat`
(Windows). Each creates the venv, installs KeyCall, finds your key file, and
opens the viewer.

## Use it from Python

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

## What you get

- **Explicit provider, always.** KeyCall never guesses which vendor issued a key and never sends a credential to more than the one provider you name.
- **No credential storage.** Keys live in memory for the client's lifetime, wrapped in a redacting type that keeps them out of reprs, logs, traces, exceptions, and pickles. Your app decides how to store them.
- **Model filtering built in.** Text-generation models by default; embeddings, image, audio, and other categories on request; unknown models never silently enter the default picker.
- **Typed errors.** Invalid key, rate limit, provider outage, timeout, and malformed response are distinguishable, never collapsed into "invalid key."
- **Streaming.** `stream_text()` yields typed events (text increments, citations, tool calls, finish) across all four wire protocols, and refuses to call a stream complete without the provider's own terminal signal.
- **Tool calling.** Define tools once and KeyCall normalizes all four call/result wire shapes, streamed or not, carrying the provider echo data some models require back verbatim. It never executes a tool.
- **Image generation.** `generate_image()` returns the picture as bytes with the media type the provider produced, on OpenAI and Gemini; the rest refuse before the network.
- **Embeddings.** `embed()` returns one vector per input, in input order, on OpenAI and Gemini; providers without an embeddings endpoint refuse before the network instead of 404ing.
- **Images, audio, and documents.** Pass bytes (or a URL where the provider fetches one) beside your text; KeyCall maps each provider's shape and detects the media type from the content. Support varies by provider and by form, so a refusal happens before the network and names who does accept that kind.
- **Web search with citations.** `web_search=True` turns on the provider's native search tool (OpenAI, Anthropic, Gemini; Perplexity always searches) and returns sources normalized to one `Citation` shape.
- **Structured output.** `response_schema=<JSON Schema>` is enforced provider-side on OpenAI, Anthropic, Gemini, Moonshot, and Perplexity; on providers without enforcement (DeepSeek, unverified custom targets) KeyCall falls back to guaranteed-valid-JSON mode and adds a result warning rather than claiming a guarantee it can't back. `result.text` is always the JSON string, regardless of which mechanism produced it.
- **Hardened transport.** TLS always verified, redirects refused, response sizes capped, SSRF and DNS-rebinding guards on custom endpoints, and generation is never silently retried.

## Provider support

Live-verified 2026-08-10. Every release re-runs a model list, a bounded
generation, a stream, a full tool round (streamed and not), an image, sound,
and document read, embeddings, image generation, an async round trip, and a
probe that each provider still reaches a working model well inside the attempt
budget, against every provider that supports them, and blocks publishing if any
of it fails:

| Provider | Protocol | Listing | Generation |
|---|---|---|---|
| OpenAI | openai | verified | verified |
| Anthropic | anthropic | verified | verified |
| Google Gemini | gemini | verified | verified |
| DeepSeek | openai-compatible | verified | verified |
| Perplexity | openai-compatible | verified | verified |
| Moonshot/Kimi | openai-compatible | verified | verified |
| Custom endpoint (explicit `base_url`) | openai-compatible | fixtures only | fixtures only |

**OpenAI** advertises `-latest` aliases its own account can't invoke, and
is retiring that family wholesale. On 2026-08-10 all four were dead:
`gpt-5-chat-latest` and `gpt-5.1-chat-latest` returned "Model not found",
and `gpt-5.2-chat-latest` and `gpt-5.3-chat-latest` were newly deprecated
hours after both had worked. The numbered models were healthy throughout.
This is the same failure as Gemini's retired models, on a provider people
assume is tidier, and it is why `verify` walks the candidates and reports
every attempt rather than trusting the first listed model.

Because of that, **candidate order follows the provider's own dates where it
publishes them.** OpenAI, Anthropic, and Moonshot date every model they
list, and a model a provider published recently is one it hasn't yet
retired, so KeyCall tries the newest first. Gemini and DeepSeek publish no
dates, and there maintained `-latest` aliases lead instead, which is right
for Gemini because it keeps those aimed at a live model. Sorting aliases
first everywhere was the earlier rule, and on OpenAI it put the four worst
candidates at the front of every walk. A release probe now checks each
provider still reaches a working model well inside the attempt budget, so
this kind of drift is caught before it reaches a key.

Two further provider quirks worth knowing, both handled:

**Gemini** keeps retired models in its list endpoint with no lifecycle field to
pre-filter on, and withdraws them per account ahead of the published shutdown
date: on 2026-08-09 the first six text models it advertised to a new key were
all refused, `gemini-2.5-*` with "no longer available to new users" months
before its documented shutdown. Gemini dates none of its models, so KeyCall
tries its maintained `-latest` aliases first there, and verification reaches
a model that works instead of walking a list of withdrawn ones; the error for
a retired model names those aliases. It also meters quota per model and tier, so one model's 429 says
nothing about the next. Its `supportedGenerationMethods` is a transport signal
rather than a modality claim: TTS variants advertise `generateContent` and then
refuse a text response, and so do the Interactions-only, computer-use, and
music families, so KeyCall lets the identifier outrank it and keeps those out
of the default text picker.

**Perplexity**'s `GET /v1/models` is scoped to the Agent API and returns
vendor-prefixed router models (`anthropic/...`, `perplexity/sonar`) that the
Sonar route rejects. Sonar's own models aren't API-discoverable, so KeyCall
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
  tool call; it can't be combined with `web_search=True` in the same
  request (forcing one tool prevents the model calling a different one), and
  KeyCall rejects that combination before any network call.
- **Gemini**'s equivalent combination (`web_search=True` with
  `response_schema`) isn't gated — no live-verified evidence either way
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

## The viewer and the verify CLI in full

The viewer is token-protected and binds `127.0.0.1`: a dashboard with live key
checks, a sortable model browser, a Playground (text, pictures in and out,
recordings, documents, tool calling, web search), and a verify report that
walks every key. An attachment the selected key can't send is turned off with
a line naming a key that can.

`keycall verify` takes the same sources as the viewer — TXT, JSON, or TOML
files, an `env:VAR_NAME` reference, or an interactive prompt — and `--generate`
adds one small bounded call per target. Keys never appear in output, and
KeyCall never writes to or deletes your credential file. Full reference in
[USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md#the-verify-cli).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Author

Built by [Mo Shehu](https://mohammedshehu.com).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
