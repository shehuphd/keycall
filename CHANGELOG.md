# Changelog

All notable changes to KeyCall are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
