# Manifest

Last updated: 2026-09-01 10:05:00 UTC

Every current source file, with what it does and what it touches. A map for orienting in the codebase, not a second copy of the docstrings.

## Package core (`src/keycall/`)

| File | What it does |
|---|---|
| `__init__.py` | Public surface: exports every public name, holds `__version__`. |
| `_client.py` | `KeyCall`/`AsyncKeyCall`: binds provider + credential + protocol at construction, drives discovery pagination and caching, category filtering, the server-tool round loop, and tracing spans. Network via `_transport.py` only. |
| `_registry.py` | Resolves a provider name to endpoints, auth scheme, operations, and dated capability evidence from the bundled catalog; validates custom base URLs. |
| `_catalog/catalog.json` | The dated per-provider evidence itself: endpoints, capabilities, sampling constraints, alias conventions, model lists for providers without a list endpoint. Versioned by `catalog_version`. |
| `_transport.py` | All HTTP and WebSocket execution: retries, response size cap, redirect refusal, header construction, download-plan enforcement. The only module that performs I/O. |
| `_cli.py` | The `keycall` command: `verify` and `view`, the no-command welcome, plain-language usage errors with one confident suggestion, pasted-key hiding, category-only color gated on a terminal and `NO_COLOR`. |
| `_verify_core.py` | The verify walk shared by the CLI and the viewer: candidate ordering, per-attempt reporting, outcome classification. |
| `_sources.py` | Credential-source loading: TXT/JSON/TOML files, `env:` references, the hidden interactive prompt; git-exposure and permission warnings. Reads key files, never writes them. |
| `_credential.py` | Internal redacting wrapper the raw key enters at client construction; refuses pickle/copy and never prints the key. |
| `_sanitize.py` | Credential scrubbing for every outbound string, request-id and display-name bounding. |
| `_classify.py` | Conservative model classification and `alias_fact()` rolling-alias facts, both from catalog evidence; unknowns stay UNKNOWN. |
| `_capabilities.py` | Typed capability lookups over the catalog's dated evidence. |
| `_cache.py` | Process-local TTL model-list cache keyed by provider + base URL + HMAC key fingerprint. Nothing persists to disk. |
| `_dnsguard.py` | Resolve-validate-pin DNS-rebinding guard for custom targets; fails closed when a proxy env var would bypass it. |
| `_realtime.py` | Sync/async realtime voice session sequencing over the transport's WebSocket wire. |
| `_transcription.py` | Sync/async streaming speech-to-text session sequencing over the same wire. |
| `_tracing.py` | Optional TraceAct spans with capture off and both redaction layers pinned on. |
| `_types.py` | Public frozen records: content parts, messages, requests, results, `Usage`, `AliasFact`, `Model`, `Voice`. |
| `_enums.py` | Public closed enums: model categories, wire protocols, operations. |
| `_errors.py` | `KeyCallError` with the typed `ErrorCode` discriminator, plus `VideoJobTimeout` carrying the still-valid job handle. |

## Adapters (`src/keycall/adapters/`)

| File | What it does |
|---|---|
| `__init__.py` | Adapter selection by protocol, with named overrides. |
| `_base.py` | The adapter contract: request building, response parsing, error translation. No I/O, never sees the credential. |
| `_openai.py` | OpenAI Responses API: text, streaming, tools, apply_patch, code interpreter, images, speech, embeddings. |
| `_anthropic.py` | Anthropic Messages API, including prompt-caching breakpoints and paginated listing. |
| `_gemini.py` | Google Gemini: text, streaming, embeddings, image and video generation, schema pre-flight gate. |
| `_openai_compat.py` | The shared chat-completions adapter (DeepSeek, Moonshot, xAI, Perplexity, custom targets): usage normalization including reasoning tokens, streaming assembly, tool calls. |
| `_moonshot.py` | Moonshot override: the `$web_search` builtin's echo-back handshake. |
| `_perplexity.py` | Perplexity override: catalog-maintained Sonar models, per-request cost units. |
| `_xai.py` | xAI override: `/v1/responses` routing for web search and reasoning effort, video generation. |
| `_realtime.py` | Realtime wire adapters (OpenAI, xAI, Gemini) mapping session events to normalized types. |
| `_stt.py` | AssemblyAI and Deepgram: credential-validating discovery, streaming transcription frames to normalized events. |
| `_elevenlabs.py` | ElevenLabs: live speech-model discovery plus catalog STT entries, voice listing, speech generation, and a streaming-transcription translator over its JSON-message wire. |

## Viewer (`src/keycall/viewer/`)

| File | What it does |
|---|---|
| `__init__.py` | `run()`: starts the server, prints the tokened URL, opens the browser, optional `--reload` restart loop. |
| `_server.py` | Localhost stdlib HTTP server: token handshake to an httpOnly cookie, CSRF checks, static files with `no-store`, WebSocket upgrade. |
| `_api.py` | Every `/api/*` route: key checks, model listing, playground generation, verify runs, settings, conversations, serialization. |
| `_registry.py` | Server-side target registry mapping integer ids to live clients; conversation store; read-timeout rebuilds. |
| `_traces.py` | In-memory request-outcome log for the Traces tab (timing and status only). |
| `_realtime_bridge.py` | Bridges the browser's voice WebSocket to a `realtime()` session. |
| `_transcription_bridge.py` | Bridges the browser's transcribe WebSocket to a `transcribe_stream()` session. |
| `_ws.py` | Minimal WebSocket frame codec for the bridges. |
| `auth.py` | Per-run token generation and constant-time comparison. |
| `static/index.html` | The single page: five tabs, dialogs, composer. |
| `static/app.js` | All frontend behavior: tabs and URL routing, playground tasks, gating, history, traces, voice/transcribe audio. |
| `static/markdown.js` | The reply renderer's small markdown subset. |
| `static/styles.css` | All styling, including the alias badge's instant hover tooltip. |

## Tests (`tests/`)

One file per surface, adversarial-first. `test_live.py` (deselected by default, `-m live`) holds the live smokes and capability-drift probes; `test_docs.py` is the docs-hygiene guard; `tests/js/markdown.test.mjs` covers the frontend renderer via `node --test`. The rest mock the wire per feature: adapters, client, CLI, streaming, tools, caching, realtime, transcription, viewer, sources, transport, types, tracing, hardening, alias facts, classification, credential, registry, embeddings, image/speech/video generation, structured output, web search, reasoning effort, async parity, and the ElevenLabs adapter with voice listing (`test_elevenlabs.py`).

## Everything else

| File | What it does |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, the `keycall` entry point, pytest config. |
| `keycall-test-keys.example.toml` | Placeholder-only example of the verify/viewer key-file format. |
| `.github/workflows/ci.yml` | Push/PR gate: tests, lint, JS tests; live smoke on manual dispatch only. |
| `.github/workflows/release.yml` | Tag-driven release: build, tests, live-strict verification, PyPI publish, GitHub release. |
| `README.md`, `USAGE.md`, `ARCHITECTURE.md`, `CHANGELOG.md` | The public doc set. |
