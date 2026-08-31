# KeyCall

One consistent interface for validating AI-provider API keys, listing and filtering the models available to them, and making normalized calls, so every product stops rebuilding the same model-picker filters and provider wrappers.

Key validation, model listing and filtering, text generation, streaming, tool calling (with tool search, custom freeform tools, and OpenAI's apply_patch convention), native web search with normalized citations, hosted code execution, structured JSON output, reasoning-effort control, prompt caching, embeddings, image, speech, realtime voice, and video generation, streaming speech-to-text, and image, audio, and document input all work and are live-verified against every provider that supports them. The API is stable.

Docs: [USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md) for the full API and CLI reference · [ARCHITECTURE.md](https://github.com/shehuphd/keycall/blob/main/ARCHITECTURE.md) for the layer diagram and component contracts · [CHANGELOG.md](https://github.com/shehuphd/keycall/blob/main/CHANGELOG.md) for version history.

## Install

Before you start: check you have Python with `python3 --version`. If that fails, see [USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md#installation) for how to install it.

```bash
pip install keycall
```

## See it work in 30 seconds

No config file, no signup, nothing to export. Type the command, name your provider, paste your key:

```bash
keycall verify
```

```
Provider (openai, anthropic, gemini, deepseek, perplexity, moonshot, xai, assemblyai, deepgram): openai
API key:
✓ openai (openai): key accepted, 79 text model(s), list digest 6d356bc3f4c24389, selection rule v4
```

The provider name is case-insensitive and the key is hidden while you type it. Add `--generate` to also make one small billable call, which reports the model that answered, its position in the provider's own list, the elapsed time, and the tokens spent.

For scripts, skip the prompts by pointing at an environment variable:

```bash
keycall verify --provider openai --source env:OPENAI_API_KEY --generate
```

Prefer to click around? Same key, one word different:

```bash
keycall view --provider openai --source env:OPENAI_API_KEY
```

That opens a local web app in your browser with your key already loaded: a dashboard that checks it live, a browsable model list with category filters, and a Playground where you can chat, show a model a picture, record a voice message straight from the page, hold a live voice conversation, transcribe your speech live, attach a PDF, offer it a tool, or have it draw you something. Keys stay in the local server process and never reach the browser.

Got several keys? Put them in a file and load them all at once — see [`keycall-test-keys.example.toml`](https://github.com/shehuphd/keycall/blob/main/keycall-test-keys.example.toml) for the format:

```bash
keycall view --source ./keys.toml
```

From a fresh clone with no Python set up at all, double-click a launcher instead: `launch.command` (macOS), `launch.sh` (Linux/macOS), `launch.bat` (Windows). Each creates the venv, installs KeyCall, finds your key file, and opens the viewer.

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
- **Rolling-alias detection.** `alias_fact(provider, model_id)` says whether an id is a rolling alias under the provider's recorded naming convention — no credential needed — including whether the provider maintains it aimed at a live model (Gemini) or was seen retiring the family (OpenAI's `-chat-latest`). `Model.alias` carries the same fact at discovery; ids with no recorded convention return `None` rather than a guess.
- **Typed errors.** Invalid key, rate limit, provider outage, timeout, and malformed response are distinguishable, never collapsed into "invalid key."
- **Streaming.** `stream_text()` yields typed events (text increments, visible reasoning progress, citations, tool calls, finish) across all four wire protocols, and refuses to call a stream complete without the provider's own terminal signal.
- **Tool calling.** Define tools once and KeyCall normalizes all four call/result wire shapes, streamed or not, carrying the provider echo data some models require back verbatim. It never executes a tool.
- **Image generation.** `generate_image()` returns the picture as bytes with the media type the provider produced, on OpenAI and Gemini; the rest refuse before the network.
- **Speech generation.** `generate_speech()` speaks text aloud on OpenAI and Gemini, the only providers with a public API for it — Anthropic's voice mode runs on a third-party subcontractor behind a consumer app, not a callable endpoint. The result carries the media type the provider sent, including Gemini's raw PCM, never a container the provider didn't produce.
- **Video generation.** `start_video()` returns a job handle, `check_video()` polls it, `fetch_video()` downloads the finished MP4 — or `generate_video()` runs all three against a timeout you choose. Gemini (Veo) and xAI (Grok Imagine) support it; the rest refuse before the network. A timeout hands back the still-valid job, so a slow render is never lost.
- **Embeddings.** `embed()` returns one vector per input, in input order, on OpenAI and Gemini; providers without an embeddings endpoint refuse before the network instead of 404ing.
- **Images, audio, and documents.** Pass bytes (or a URL where the provider fetches one) beside your text; KeyCall maps each provider's shape and detects the media type from the content. Support varies by provider and by form, so a refusal happens before the network and names who does accept that kind.
- **Web search with citations.** `web_search=True` turns on the provider's native search tool (OpenAI, Anthropic, Gemini, xAI, Moonshot; Perplexity always searches) and returns sources normalized to one `Citation` shape (Moonshot reports none).
- **Reasoning effort control.** `reasoning_effort="low"` (or `"medium"` / `"high"`) maps to the provider's native thinking control on OpenAI, Anthropic, Gemini, Perplexity, and xAI — each verified live to move reasoning-token spend. Providers that accept the parameter without honoring it refuse instead of silently ignoring it. The spend itself is normalized into `usage.reasoning_tokens` wherever the provider reports a count (OpenAI, Gemini, DeepSeek, Moonshot, xAI).
- **Realtime voice sessions.** `realtime()` opens a live WebSocket conversation on OpenAI, xAI, or Gemini — text or microphone audio up, normalized audio/transcript/turn events down, sync and async. The credential rides the handshake headers and never enters a URL.
- **Streaming transcription.** `transcribe_stream()` opens a live speech-to-text session on AssemblyAI or Deepgram — raw PCM audio up, normalized interim/final transcripts with per-word millisecond timings down, and the provider's billable audio seconds on the session-ended event. Sync and async, same header-auth rule as realtime.
- **Hosted code execution.** `code_interpreter=True` lets the model write and run code on the provider's own sandbox (OpenAI, Anthropic, Gemini, xAI), with each run returned as a typed part — nothing executes on your machine.
- **Tool search.** `Tool(defer_loading=True)` keeps a large tool library out of the model's context until it searches for what it needs (OpenAI, Anthropic); discovered tools call and reply like ordinary ones.
- **File-editing tool convention.** `apply_patch=True` enables OpenAI's built-in file-editing tool: the model proposes create/update/delete operations as V4A diffs, arriving as ordinary `ToolCall`/`ToolResult` parts named `"apply_patch"` in the same replay loop as any other tool. OpenAI-only; other providers refuse before the network.
- **Custom (freeform) tools.** `Tool(input_schema=None)` declares a tool with no JSON Schema: the model's call arrives as a plain string instead of parsed arguments. OpenAI-only; other providers refuse before the network.
- **Prompt caching.** `TextInput(cacheable=True)` marks a stable prefix (a big system prompt, reference material) for caching. Anthropic is the one provider where caching doesn't happen at all without this marker; OpenAI already caches automatically and the marker opts into its optional explicit mode; every other provider ignores the flag and keeps caching automatically on its own. `Usage.cached_input_tokens` reports a cache hit uniformly everywhere, marked or not.
- **Structured output.** `response_schema=<JSON Schema>` is enforced provider-side on OpenAI, Anthropic, Gemini, Moonshot, and Perplexity; on providers without enforcement (DeepSeek, unverified custom targets) KeyCall falls back to guaranteed-valid-JSON mode and adds a result warning rather than claiming a guarantee it can't back. `result.text` is always the JSON string, regardless of which mechanism produced it.
- **Hardened transport.** TLS always verified, redirects refused, response sizes capped, SSRF and DNS-rebinding guards on custom endpoints that fail closed when a proxy would bypass them, and generation is never silently retried.

## Provider support

Live-verified 2026-08-29. Every release re-runs a model list, a bounded generation, a stream, a full tool round (streamed and not, including apply_patch, custom tools, and tool search), hosted code execution, an image, sound, and document read, embeddings, image generation, a video render, a prompt-caching round trip, an async round trip, a live streaming-transcription session, capability-drift probes against previously observed provider behavior, and a probe that each provider still reaches a working model well inside the attempt budget, against every provider that supports them, and blocks publishing if any of it fails:

| Provider | Protocol | Listing | Generation |
|---|---|---|---|
| OpenAI | openai | verified | verified |
| Anthropic | anthropic | verified | verified |
| Google Gemini | gemini | verified | verified |
| DeepSeek | openai-compatible | verified | verified |
| Perplexity | openai-compatible | verified | verified |
| Moonshot/Kimi | openai-compatible | verified | verified |
| xAI (Grok) | openai-compatible | verified | verified |
| AssemblyAI | stt | verified | streaming transcription verified |
| Deepgram | stt | verified | streaming transcription verified |
| Custom endpoint (explicit `base_url`) | openai-compatible | fixtures only | fixtures only |

AssemblyAI and Deepgram are speech-to-text providers: their generation column is `transcribe_stream()`, since they have no text-generation API, and their model lists are maintained catalog data behind a live credential check.

**OpenAI** advertises `-latest` aliases its own account can't invoke, and is retiring that family wholesale. On 2026-08-10 all four were dead: `gpt-5-chat-latest` and `gpt-5.1-chat-latest` returned "Model not found", and `gpt-5.2-chat-latest` and `gpt-5.3-chat-latest` were newly deprecated hours after both had worked. The numbered models were healthy throughout. This is the same failure as Gemini's retired models, on a provider people assume is tidier, and it is why `verify` walks the candidates and reports every attempt rather than trusting the first listed model.

Because of that, **candidate order follows the provider's own dates where it publishes them.** OpenAI, Anthropic, and Moonshot date every model they list, and a model a provider published recently is one it hasn't yet retired, so KeyCall tries the newest first. Gemini and DeepSeek publish no dates, and there maintained `-latest` aliases lead instead, which is right for Gemini because it keeps those aimed at a live model. Sorting aliases first everywhere was the earlier rule, and on OpenAI it put the four worst candidates at the front of every walk. A release probe now checks each provider still reaches a working model well inside the attempt budget, so this kind of drift is caught before it reaches a key.

Two further provider quirks worth knowing, both handled:

**Gemini** keeps retired models in its list endpoint with no lifecycle field to pre-filter on, and withdraws them per account ahead of the published shutdown date: on 2026-08-09 the first six text models it advertised to a new key were all refused, `gemini-2.5-*` with "no longer available to new users" months before its documented shutdown. Gemini dates none of its models, so KeyCall tries its maintained `-latest` aliases first there, and verification reaches a model that works instead of walking a list of withdrawn ones; the error for a retired model names those aliases. It also meters quota per model and tier, so one model's 429 says nothing about the next. Its `supportedGenerationMethods` is a transport signal rather than a modality claim: TTS variants advertise `generateContent` and then refuse a text response, and so do the Interactions-only, computer-use, and music families, so KeyCall lets the identifier outrank it and keeps those out of the default text picker.

**Perplexity**'s `GET /v1/models` is scoped to the Agent API and returns vendor-prefixed router models (`anthropic/...`, `perplexity/sonar`) that the Sonar route rejects. Sonar's own models aren't API-discoverable, so KeyCall maintains them in its catalog and uses the list call purely as a credential check. Note the version prefix: the unversioned `https://api.perplexity.ai/models` returns 404 for every key, valid or not, so anything validating a key against that path rejects good credentials. `/v1/models` answers 401 for a bad key and 200 for a good one, which is what makes it usable as a check (verified 2026-08-09).

### Structured output notes, per provider

- **OpenAI** requires `additionalProperties: false` on every object level of the schema for its strict `json_schema` mode, or the request 400s. This is an OpenAI requirement, not a KeyCall one — write schemas with it from the start.
- **Anthropic** implements structured output by forcing a single synthetic tool call; it can't be combined with `web_search=True` in the same request (forcing one tool prevents the model calling a different one), and KeyCall rejects that combination before any network call.
- **Gemini**'s equivalent combination (`web_search=True` with `response_schema`) isn't gated — no live-verified evidence either way that Gemini rejects it, so KeyCall passes it through rather than guessing.
- **DeepSeek** hard-requires the literal word "json" somewhere in the prompt for its `json_object` fallback mode, or it 400s. KeyCall detects this and injects a short system instruction automatically when needed, and always says so via a result warning.
- **Moonshot/Kimi** reasoning-capable models can spend the entire `max_output_tokens` budget on a visible reasoning trace and never emit a final answer if the budget is too small. KeyCall detects the resulting empty-content-with-reasoning-trace response and adds a warning rather than returning a silent empty result; give these models a larger budget than you'd expect a short answer to need.

Because of quirks like these, `keycall verify --generate` walks the filtered models in provider order and prints the outcome of every attempt until one succeeds, so drift stays visible rather than being masked by a silent retry.

## The viewer and the verify CLI in full

The viewer is token-protected and binds `127.0.0.1`. Opening the printed link trades its token for an httpOnly, `SameSite=Strict` session cookie and strips it from the address bar, so the secret never reaches page script or browser history, and state-changing requests are CSRF-checked. Inside: a dashboard with live key checks, a sortable model browser, a Playground (text, pictures in and out, recordings, live voice conversations, documents, tool calling, web search), a verify report that walks every key, and a Traces tab logging every request this run has made (timing and outcome only, never prompts or replies), searchable and sortable by column. An attachment the selected key can't send is turned off with a line naming a key that can.

`keycall verify` takes the same sources as the viewer — TXT, JSON, or TOML files, an `env:VAR_NAME` reference, or an interactive prompt — and `--generate` adds one small bounded call per target. Keys never appear in output, and KeyCall never writes to or deletes your credential file. Full reference in [USAGE.md](https://github.com/shehuphd/keycall/blob/main/USAGE.md#the-verify-cli).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Author

Built by [Mo Shehu](https://mohammedshehu.com).

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/shehuphd/keycall/blob/main/LICENSE).

