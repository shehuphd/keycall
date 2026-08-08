# Changelog

All notable changes to KeyCall are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/shehuphd/keycall/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shehuphd/keycall/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shehuphd/keycall/releases/tag/v0.1.0
