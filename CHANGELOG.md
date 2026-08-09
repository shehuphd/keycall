# Changelog

All notable changes to KeyCall are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Image input** on `generate_text()` and `stream_text()`: pass an
  `ImageInput` beside your text in a user message. Live-verified on OpenAI,
  Anthropic, Gemini, Perplexity, and Moonshot. Support splits by form
  rather than by provider alone, and the gate names the form that works:
  Gemini and Moonshot read bytes but refuse a remote URL, and DeepSeek's
  API is text only. KeyCall never fetches a URL itself. The media type is
  detected from the content (PNG, JPEG, GIF, WebP), because Anthropic and
  Gemini both reject a mismatched type and the bytes are better evidence
  than a caller's label; an unidentifiable image raises rather than being
  sent with a guess.

- **Moonshot's pinned sampling values are gated before the network.** Every
  kimi model accepts only `temperature=1.0` and `top_p=0.95` and returns a
  400 for anything else (verified 2026-08-09; reported for kimi-k3, and the
  probe found it applies to the whole family). The error names the value
  that works rather than only refusing, and the permitted values still pass
  through, so this does not become a blanket ban on sampling parameters.
- `ModelDiscovery.catalog_stale` is now set, and carries a matching
  warning. The field has existed since 0.2.0 and nothing ever assigned it,
  so callers reading it always saw "fresh" no matter how old the bundled
  data was. It turns true when the catalog's verification stamp is more
  than 90 days old.
- The viewer Playground drives tool calling end to end: define tools as
  JSON, pick a `tool_choice`, and when the model asks for a tool the calls
  render with their arguments and a box for each result. Sending the
  results continues the conversation. KeyCall still never runs a tool, so
  the browser owns the loop and replays the turns, echo data included.

### Fixed

- HTTP 402 is reported as `PERMISSION_DENIED` rather than falling through
  to `INVALID_PROVIDER_RESPONSE`. A 402 means the key is valid and the
  account is unfunded or on a billing hold, so calling it a malformed
  response sent callers hunting for a bug in their own request. The
  provider's message, which carries the billing link, is preserved.
  Anthropic already mapped it this way; the base and Gemini adapters now
  match.
- **OpenAI reasoning models rejected any replayed tool call.** When the
  model emits a `reasoning` item alongside a `function_call`, the call
  cannot be replayed without it: the next request fails with HTTP 400
  naming the missing item. KeyCall discarded reasoning items as
  server-side traces, so every tool round on a reasoning model broke at
  the second turn. The item now travels in `ToolCall.opaque` and is
  replayed ahead of the call it belongs to, once even when parallel calls
  share it. Reasoning items appear only when the model actually reasons,
  which is why this survived the earlier live rounds; found by driving the
  Playground against gpt-5.3-chat-latest.
- `ImageInput`, `AudioInput`, and `FileInput` now say plainly that text
  generation does not accept them yet, in the error and on the types
  themselves. They are exported and validated, which reasonably reads as a
  promise of support; the refusal previously sounded like a malformed
  message rather than an unimplemented feature. It still raises
  `UNSUPPORTED_OPERATION` before any network call.
- **Citations were deduplicated inconsistently**, so the same request could
  return different citation lists depending on how it was called: the
  compat family collapsed by URL, Gemini collapsed while streaming but not
  otherwise, and OpenAI and Anthropic never did. One rule now applies
  everywhere — a citation is dropped only when it repeats an earlier one
  exactly, in URL, title, and excerpt. Per-claim citations that share a URL
  but carry different `cited_text` survive, because that text is the
  attribution. Streamed `citation` events match the final result exactly.

### Changed

- **Provider capability evidence moved into the registry.** Which providers
  support tool calling, web search, and schema enforcement, which model
  families restrict sampling, and which Gemini families only pretend to do
  text now live in the bundled catalog under each provider's
  `capabilities` block, each stamped with the date it was last verified
  against the live API. Gates and the error messages that list the working
  alternatives read the same data, so they cannot drift apart, and adding a
  capability is a catalog edit rather than a code change.

- **Verification tries a provider's maintained `-latest` aliases before its
  dated model ids** (selection rule version 3). Gemini withdraws models per
  account ahead of their published shutdown dates while still listing them:
  on 2026-08-09 the first six text models it advertised to a new key were
  all refused, including `gemini-2.5-*` whose documented shutdown is months
  away. A walk in list order spent almost its whole budget on models Google
  had already shut down; it now verifies on the first attempt. Each attempt
  still reports both its filtered and its raw position, so promotion is
  visible rather than silent.
- Gemini model families that advertise `generateContent` and then refuse a
  text call no longer enter the default text picker: the Interactions-only
  models (`deep-research-*`, `antigravity-*`, `gemini-omni-*`), the
  computer-use preview, and Lyria (music generation). They remain listable
  under the unknown category. Verified by calling each one.
- A retired Gemini model's error now explains that the model came from
  Gemini's own list and names the `-latest` aliases that survive these
  retirements, instead of leaving the caller to wonder why a listed model
  is unusable.

### Notes

- Google announced `temperature`, `top_p`, and `top_k` as deprecated on its
  latest Gemini models on 2026-07-21. KeyCall does **not** gate them:
  `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`, and
  `gemini-flash-latest` all still accept both parameters (verified
  2026-08-09). The sampling gate covers what a provider rejects today, not
  what it has announced it will stop supporting.

## [0.7.0] — 2026-08-09

### Added

- **Streamed tool calls**: `stream_text()` accepts `tools` and
  `tool_choice`, and reports calls as three typed events —
  `ToolCallStarted` when the model names a tool,
  `ToolCallArgumentsDelta` as the argument JSON arrives in fragments, and
  `ToolCallComplete` once it parses. `result.tool_calls` after a stream
  matches what the same request returns non-streamed, so dispatch and
  replay code is shared between both paths. Live-verified across every
  supporting provider, including Gemini, which sends each call whole and
  so emits no fragments at all.

### Changed

- Streaming and tools are no longer mutually exclusive; the gate that
  raised `UNSUPPORTED_OPERATION` for the combination is gone. Every other
  tool-calling gate still applies to streams, before any network call.
- Custom OpenAI-compatible targets now get the unverified-tool-support
  warning on streamed results too, not only non-streamed ones.

### Fixed

- `round_trip_duration_ms` on a streamed result excluded the request and
  the wait for the first byte, timing only from the response headers to
  the last event. On providers that buffer before responding it reported a
  small fraction of the real duration — about 1% on Anthropic, which read
  like a cache hit. The clock now starts before the request goes out, as
  it always has on the non-streamed path.
- The viewer reported "unreported tokens" whenever a provider sent no
  total, even though it had per-direction counts in hand; it now shows
  those (`16 in / 4 out tokens`), and the verify walk names the field it
  actually has, matching the CLI's "total tokens" wording.
- README described streaming and tool calling as unimplemented, which had
  been stale since 0.5.0.

## [0.6.0] — 2026-08-09

### Added

- **Tool calling** (`tools=` and `tool_choice=` on `generate_text()` and
  `TextGenerationRequest`): caller-defined tools normalized across all four
  wire protocols, live-verified with a full call → result → answer round
  per provider. The model's requests surface as typed `ToolCall` parts
  (`result.tool_calls`; parallel calls are normal), results go back as
  `ToolResult` parts in a user message, and
  `result.to_assistant_message()` replays the model's turn including
  provider echo data some providers require verbatim (Gemini rejects a
  replay missing its thought signature). KeyCall never executes a tool and
  never runs the loop. `web_search` combines with tools on OpenAI,
  Anthropic, and Gemini (the required Gemini `toolConfig` flag is set
  automatically). Gated before any network call: Perplexity (no tool
  support, verified), Anthropic tools + `response_schema`, and streaming +
  tools. Custom OpenAI-compatible targets pass through with a result
  warning that support is unverified. Malformed tool-call arguments from a
  provider raise `invalid_provider_response` rather than being dropped.
- The live suite runs a full tool round per supporting provider, plus a
  capability-drift probe that fails if Perplexity ever starts accepting
  tools, so the gate cannot silently outlive its evidence.

## [0.5.1] — 2026-08-08

### Added

- The viewer Playground renders generations token-by-token: a new
  authenticated `POST /api/generate/stream` endpoint relays `stream_text()`
  events to the browser as SSE, ending in either a full result payload or
  an error event, so an interrupted provider stream surfaces as a visible
  error instead of a hung request. The frontend falls back to the
  non-streamed path only when streaming never started; once tokens have
  arrived it reports the interruption rather than spending a second
  generation.

## [0.5.0] — 2026-08-08

### Added

- **Streaming** (`stream_text()` on `KeyCall` and `AsyncKeyCall`): typed
  event iteration (`StreamStart`, `TextDelta`, `CitationFound`,
  `StreamFinish`, `UnknownStreamEvent`) over every supported provider's
  native SSE surface, live-verified per provider. `stream.result()` after
  exhaustion returns the same `InvocationResult` as a non-streamed call.
  `web_search` and `response_schema` combine with streaming wherever the
  provider supports them non-streamed. A stream that closes without the
  provider's terminal signal raises `NETWORK_ERROR` instead of passing off
  a partial response as complete; size caps apply per event and to the
  whole stream; streaming is never retried.
- The live suite includes one bounded streamed generation per target.

### Known limitations (in addition to earlier versions')

- Gemini rejects a `response_schema` containing `additionalProperties`
  (HTTP 400), while OpenAI's strict mode requires `additionalProperties:
  false`. One schema cannot satisfy both providers; KeyCall passes the
  caller's schema through unmodified to either.

## [0.4.1] — 2026-08-08

### Changed

- Code comments and docstrings state their constraints directly instead of
  citing internal design documents that are not part of the repository.

## [0.4.0] — 2026-08-08

### Added

- `VerifyResult` records a digest of the raw model-list snapshot
  (`model_list_digest`) and the selection procedure version
  (`selection_rule_version`); each `ModelAttempt` records the
  classification evidence that made the model a candidate
  (`classification_source`). A verify report is now reconstructable
  against the provider surface that produced it.
- ARCHITECTURE.md: layer diagram, credential-boundary contract, and
  per-component contracts.
- CI test matrix includes Python 3.14.
- Live verification modes: a `live`-marked pytest suite (deselected by
  default) running the verify walk against real providers; a manual-only
  `live-warn` CI job that reports without failing; and a `live-strict`
  release gate that blocks publishing until every target verifies,
  including when the `KEYCALL_LIVE_TARGETS` secret is absent. Credentials
  load only at test run time from `KEYCALL_LIVE_SOURCE`.

## [0.3.1] — 2026-08-08

### Fixed

- The README status line no longer hardcodes a version number; the 0.3.0
  PyPI page showed "early release (0.2.0)" because the line was not bumped
  with the release. The version now appears only in surfaces that update
  automatically (package metadata, this changelog, release tags).

## [0.3.0] — 2026-08-08

### Added

- **Structured output** (`response_schema=<JSON Schema>` on `generate_text()`
  and `TextGenerationRequest`): enforced provider-side on OpenAI (Responses
  API `text.format`), Anthropic (forced single-tool `tool_choice`), Gemini
  (`responseSchema`), and the compat-family providers confirmed to support
  strict `response_format: json_schema` (Moonshot, Perplexity). Providers
  without enforcement (DeepSeek, unverified custom targets) fall back to
  guaranteed-valid-JSON mode with a result warning rather than a claimed
  guarantee. `result.text` is the JSON string on every provider and
  mechanism, so callers parse identically regardless of provider.
- DeepSeek's undocumented hard requirement that the prompt contain the
  literal word "json" for its `json_object` mode (confirmed live: a 400
  otherwise) is detected and satisfied automatically with an injected system
  instruction; always surfaced via a result warning, never silent.
- A reasoning-capable compat model (Moonshot/Kimi) exhausting
  `max_output_tokens` on its reasoning trace before emitting any final
  content now produces a result warning naming the likely cause, instead of
  a silent empty `result.text`.

### Changed

- `keycall verify` attempts now record and report each candidate's
  zero-based position in the provider's raw model list alongside its
  filtered position (`ModelAttempt.raw_position`).
- Model-list pagination stopping at the 10-page limit while the provider
  still reports more pages now adds a truncation warning to the returned
  `ModelDiscovery` instead of returning silently.
- `Retry-After` response headers in HTTP-date form are now honored; only
  the seconds form was parsed before.
- The viewer's per-target model cache now expires after the same 300-second
  TTL as the library's availability cache instead of persisting until a
  manual refresh.

### Fixed

- Provider-supplied request identifiers are sanitized (control characters
  stripped, length bounded) before entering results or errors.
- Boundary sanitization redacts URL-encoded and base64-encoded forms of the
  active credential, and recognizes the `pplx-` key prefix.
- Transient DNS-guard resolution failures on custom targets are now
  eligible for the model-list retry budget.
- A proxy environment variable combined with a guarded custom target emits
  a runtime warning that proxied requests bypass the DNS-rebinding and
  private-address guard.
- The viewer registry closes already-opened clients when a later target in
  the same batch fails resolution, and reads target views under its lock.
- The viewer returns HTTP 400 for a malformed `Content-Length` header, a
  non-object JSON body, or an out-of-range `attempts` value (valid range
  1–32) instead of failing in the handler thread.

### Known limitations (in addition to 0.2.0's)

- Anthropic cannot combine `web_search=True` with `response_schema` in one
  call (forced tool_choice is mechanically incompatible with also invoking
  a second server-side tool); KeyCall raises before any network call.
- Gemini's equivalent combination is not gated — no live-verified evidence
  either way that Gemini itself rejects it, so it is passed through rather
  than guessed at.
- OpenAI's strict `json_schema` mode requires `additionalProperties: false`
  at every object level of the caller's schema, or the request 400s. This is
  an OpenAI requirement; KeyCall does not rewrite caller-supplied schemas to
  add it.

## [0.2.0] — 2026-08-05

### Added

- **Web search** (`web_search=True` on `generate_text()` / `TextGenerationRequest`):
  enables the provider's native server-side search tool — OpenAI (`web_search`,
  Responses API), Anthropic (`web_search_20250305`), Gemini (`google_search` on
  `generateContent`). Perplexity's Sonar always searches; the flag is accepted
  as a no-op there. Providers without a native search tool (DeepSeek, Moonshot,
  custom targets) raise `UNSUPPORTED_OPERATION` before any network call rather
  than silently ignoring the request. Live-verified against all four providers.
- **`Citation` type and `InvocationResult.citations`**: web-search sources
  normalized to one shape (`url`, `title`, `cited_text`) across OpenAI's text
  annotations, Anthropic's per-block citations, Gemini's grounding chunks, and
  Perplexity's `citations`/`search_results` (previously discarded). Gemini
  citation URLs are Google's own vertexaisearch redirect links by design.
- **Launch scripts** (`launch.command`, `launch.sh`, `launch.bat`): one-click
  viewer startup from a fresh clone on any OS — path-independent, resolves the
  interpreter explicitly, validates or rebuilds the venv, and locates the key
  file.
- **Guided empty and error states in the viewer**: launching with no source
  opens a load-your-key-file panel (`POST /api/source` reads the file
  server-side; keys still never enter the browser), and an expired or missing
  token renders a clear explanation instead of a broken page.
- **Local web viewer** (`keycall view --source ./keys.toml`): dashboard of
  loaded targets with live key checks, model catalog browser with category
  filters, playground for real generation calls (including web search with
  rendered citations), and a verify report — all in the browser. Standard
  library only; static assets ship in the wheel. A per-run auth token is
  mandatory on every API request (printed once, never persisted), credentials
  stay server-side (the browser only ever sees target ids and names), and the
  page carries a `default-src 'self'` CSP.

### Changed

- `keycall verify`'s model walk extracted to a structured core
  (`VerifyResult`/`ModelAttempt`) shared between the CLI and the viewer.

## [0.1.0] — 2026-08-05

First release. Key validation, model discovery and filtering, and text
generation, live-verified against every supported provider.

### Added

**Clients**

- `KeyCall` and `AsyncKeyCall` with identical surfaces; only awaiting differs.
- Provider and credential bound once at construction as immutable client
  identity: no setters, no per-call override, no public `api_key` property.
- Context-manager support; `close()` releases the credential and HTTP client.
- Configurable `connect_timeout`, `read_timeout`, `max_response_bytes`,
  `trust_env`, `allow_insecure_localhost`, and `allow_private_network`.

**Model discovery**

- `list_models(categories=..., refresh=...)` returning a `ModelDiscovery`
  envelope with `models`, `fetched_at`, `from_cache`, `catalog_version`, and
  `warnings`.
- Eight-member `ModelCategory` taxonomy; text generation is the default filter.
- Conservative classification: explicit provider metadata first, then
  maintained identifier rules. Unclassifiable models resolve to `UNKNOWN` and
  never enter the default text picker.
- Process-local availability cache with a 5-minute TTL, keyed by an HMAC
  fingerprint of the credential rather than the credential itself.

**Text generation**

- `invoke(TextGenerationRequest)` as the low-level primitive and
  `generate_text(...)` as the convenience path.
- Normalized `InvocationResult` with typed output parts, `usage`,
  `round_trip_duration_ms`, `provider_request_id`, `finish_reason`, and
  `warnings`.
- `temperature` and `top_p`, validated at construction and omitted from the
  wire body when unset. Models with maintained evidence that they reject
  sampling parameters (OpenAI o-series and gpt-5; Anthropic Opus 4.7+,
  Opus 5+, Sonnet 5+) fail with `MODEL_NOT_SUITABLE` before any network call.
- Usage fields distinguish "provider reported zero" from "provider did not
  report", which stays `None` and is never fabricated.

**Providers**

- OpenAI (Responses API), Anthropic (Messages), Google Gemini
  (`generateContent`), DeepSeek, Perplexity (Sonar), Moonshot/Kimi.
- Explicit OpenAI-compatible custom targets via `protocol` plus `base_url`.
- Provider identity and wire protocol kept as separate concepts; named
  providers may override their protocol's default adapter.

**Errors**

- Single `KeyCallError` with a typed `ErrorCode` discriminator, plus
  `retryable`, `status_code`, `provider_request_id`, and `retry_after`.
- Twelve normalized codes spanning credential, model, provider, transport, and
  setup failures.

**Security**

- Credentials wrapped in a redacting type at the single public entry point;
  excluded from reprs, formatting, exceptions, logs, traces, copies, and
  pickles, with canary tests asserting absence.
- `Credential.reveal()` called from exactly one place, the transport layer's
  header builder.
- Provider-originated error text scrubbed for credential values, credential
  patterns, and control characters, then length-bounded, before reaching any
  result, exception, log, or trace.
- Redirects refused rather than followed while carrying a credential.
- Response bodies read incrementally against a 10 MB cap.
- SSRF guard rejecting literal private, loopback, link-local, and reserved IP
  targets unless explicitly opted in.
- DNS-rebinding guard for custom endpoints: resolves once, rejects if any
  resolved address is private, then connects to the validated address while
  preserving the original hostname for TLS SNI and `Host`.
- HTTPS required for custom endpoints; plain HTTP only for localhost behind an
  explicit flag.

**Reliability**

- Operation-aware retries: bounded retries with backoff and `Retry-After`
  support for model listing; none for generation, since no supported provider
  documents generation idempotency.
- Explicit connect and read timeouts on every request.

**CLI**

- `keycall verify` for live credential verification from TXT, JSON, or TOML
  files, an explicit `env:VAR_NAME` reference, or an interactive prompt.
- `--generate` makes one bounded call per target, walking filtered models in
  provider order and reporting every attempt (`--attempts`, default 8) so
  provider drift stays visible.
- `--strict-credentials` promotes credential-file warnings to errors.
- Credential files are never modified or deleted; keys never appear in output.

**Observability**

- Optional TraceAct integration emitting `keycall.list_models` and
  `keycall.text_generation` spans with safe fields only. Silent when TraceAct
  is absent or the host has not configured it; disabled with one warning on an
  incompatible version.

### Known limitations

- Streaming, tool calling, structured output, and non-text modalities are not
  implemented.
- Gemini's list endpoint advertises models an account cannot invoke and exposes
  no lifecycle field, so they cannot be pre-filtered.
- Perplexity's Sonar models are not API-discoverable and are maintained in the
  bundled catalog.
- The provider catalog ships inside the package and updates only on release.

[Unreleased]: https://github.com/shehuphd/keycall/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/shehuphd/keycall/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/shehuphd/keycall/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/shehuphd/keycall/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/shehuphd/keycall/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/shehuphd/keycall/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/shehuphd/keycall/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/shehuphd/keycall/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/shehuphd/keycall/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shehuphd/keycall/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shehuphd/keycall/releases/tag/v0.1.0
