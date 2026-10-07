# Manifest

Last updated: 2026-10-02 15:09:08 UTC

Every current source file, with what it does and what it touches. A map for orienting in the codebase, not a second copy of the docstrings.

## Package core (`src/keycall/`)

| File | What it does |
|---|---|
| `__init__.py` | Public surface: exports every public name, holds `__version__`. |
| `_client.py` | `KeyCall`/`AsyncKeyCall`: binds provider + credential + protocol at construction, drives discovery pagination and caching, category filtering, the server-tool round loop, the picture flow shared by `generate_image()` and the named picture operations (one request, an optional job poll, then a download of the provider's result links), the provider-tool job pair (`list_tools()`, `start_tool()`/`check_tool()`/`fetch_tool()`/`run_tool()`), the video, batch, and transcription job flows (submit, poll, fetch, cancel where offered, and the timeout-bounded wrappers), the one-round-trip transcription path, the one-shot dictation path (`dictate()`), the judgment path (`judge()`), the retired-model pre-flight gate and listing withholding, `probe_services()` for service providers, and tracing spans. Network via `_transport.py` only. |
| `_registry.py` | Resolves a provider name to endpoints, auth scheme, operations, kind (model or service, with credential fields and service categories), and dated capability evidence from the bundled catalog; validates custom and required-at-construction base URLs; `retired_model_fact()` looks a model id up in a provider's retired-model records, alias spellings included; `providers_with_image_operation()` and `providers_with_video_input()` name the providers that honour a picture input, for refusals. |
| `_catalog/catalog.json` | The dated per-provider evidence itself: endpoints, capabilities (including `supports_seed`), sampling and tool-choice constraints, alias conventions, service-provider entries (kind, credential fields, category endpoints), model lists for providers without a list endpoint (transcription models carry per-wire facts), retired-model records per provider (id, aliases, date, replacement, evidence note), and per-provider picture facts (the picture operations each offers with the optional inputs it honours, video picture inputs, pixel-size rules, download hosts, and named provider tools), with Ideogram's per-model routes. Versioned by `catalog_version`. |
| `_transport.py` | All HTTP and WebSocket execution: retries, response size cap, redirect refusal, header construction (including the per-request HS256 mint for jwt_hs256 providers and the catalog-host override for multi-host service categories), multipart file upload (with file-less form-only and typed-part variants, the latter carrying dictation's JSON config part beside its audio part), raw binary request bodies, JSONL/text success-body passthrough, download-plan enforcement, and the fail-closed refusal when a proxy env var would route a custom target around the DNS-rebinding guard. The only module that performs I/O. |
| `_mask.py` | Edit-mask conversion with the standard library alone: reads an 8-bit PNG mask (white marks the area to change) and redraws it in a provider's own convention (black over the edit area for Ideogram, transparent for OpenAI); refuses an unreadable mask or one marking all or nothing. |
| `_size.py` | Picture-size parsing: tells an aspect ratio (`16:9`) from a pixel size (`1280x768`) and converts a ratio to pixels for a provider that takes pixels only. |
| `_cli.py` | The `keycall` command: `verify` and `view`, the no-command welcome, plain-language usage errors with one confident suggestion, pasted-key hiding, category-only color gated on a terminal and `NO_COLOR`. |
| `_verify_core.py` | The verify walk shared by the CLI and the viewer: candidate ordering, per-attempt reporting, outcome classification; a judgment provider's generate proof is one minimal Noul per decision candidate. |
| `_sources.py` | Credential-source loading: TXT/JSON/TOML files, `env:` references, the hidden interactive prompt; git-exposure and permission warnings. Malformed JSON and TOML report the parser's own position so an unreadable source can be placed. Reads key files, never writes them. |
| `_credential.py` | Internal redacting wrapper the raw secret fields enter at client construction (one `api_key` for a model provider, a named pair for a service provider); refuses pickle/copy and never prints any field. |
| `_sanitize.py` | Credential scrubbing for every outbound string, request-id and display-name bounding. |
| `_classify.py` | Conservative model classification and `alias_fact()` rolling-alias facts, both from catalog evidence; unknowns stay UNKNOWN. |
| `_capabilities.py` | Typed capability lookups over the catalog's dated evidence. |
| `_cache.py` | Process-local TTL model-list cache keyed by provider + base URL + HMAC key fingerprint. Nothing persists to disk. |
| `_dnsguard.py` | Resolve-validate-pin DNS-rebinding guard for custom targets. |
| `_realtime.py` | Sync/async realtime voice session sequencing over the transport's WebSocket wire. |
| `_live.py` | Sync/async live full-duplex voice session sequencing over the same wire (OpenAI's gpt-live), an additive sibling of the realtime session. |
| `_transcription.py` | Sync/async streaming speech-to-text session sequencing over the same wire. |
| `_tracing.py` | Optional TraceAct spans with capture off and both redaction layers pinned on. |
| `_types.py` | Public frozen records: content parts, messages, requests, results, `Usage`, `AliasFact`, `Model`, `Voice`, the batch records (`BatchRequest`, `BatchJob`, `BatchCounts`, `BatchResult`), the prerecorded-transcription records (`TranscriptionRequest`, `TranscriptionJob`, `TranscriptionResult`), the dictation records (`DictationRequest`, `DictationResult`, `DictationWord`), the judgment records (the typed question and answer classes plus `JudgmentRequest`/`JudgmentResult`), the picture records (`ImageOperationRequest`, `TextBlockOutput`, `ProviderTool`, `ToolJob`), the service-probe records (`ServiceReport`, `ServiceStatus`), the live-session records (`LiveConfig` and the `LiveEvent` family), and `WithheldModel` for a listing's filtered-out retired models. |
| `_enums.py` | Public closed enums: model categories, wire protocols, operations. |
| `_errors.py` | `KeyCallError` with the typed `ErrorCode` discriminator, plus `VideoJobTimeout`, `BatchJobTimeout`, `TranscriptionJobTimeout`, and `ToolJobTimeout`, each carrying the still-valid job handle. |

## Adapters (`src/keycall/adapters/`)

| File | What it does |
|---|---|
| `__init__.py` | Adapter selection by protocol, with named overrides. |
| `_base.py` | The adapter contract: request building, response parsing, error translation, the pre-flight generation checks (`validate_generation_request`, including the sampling-constraint, tool-choice-constraint, seed, and Gemini `additionalProperties` schema gates), the batch hook set (prelude/submit/status/results/cancel), the prerecorded-transcription hook set (sync build/parse plus job upload/submit/status/result), the picture hook set (operation build/parse, job status, result downloads, `provider_options` merging, and the gate that refuses an operation or input the provider doesn't honour, naming the providers that do), the provider-tool hook set, and the service-probe hook set (`ServiceProviderAdapter`: per-category specs and status parsing) with their refusal gates. No I/O, never sees the credential. |
| `_openai.py` | OpenAI Responses API: text, streaming, tools, apply_patch, code interpreter, image generation and edits (multipart `/images/edits` with a transparent-area mask, sizes held to each model family's pixel rules), speech, embeddings, realtime sessions; `FileBatchDialect`, the upload-a-JSONL batch flow shared with Moonshot; prerecorded transcription (multipart, whisper-1-only word timings); usage reads cache reads and explicit-breakpoint cache writes as parts of the input count. |
| `_anthropic.py` | Anthropic Messages API, including prompt-caching breakpoints and usage that sums the uncached, cache-write, and cache-read input parts into one input count (plain, streamed, and batch), native structured output via `output_config.format`, the server-tool round for web search, code interpreter, and tool search, paginated listing, and the inline batch dialect with mixed models and a host-pinned results download. |
| `_google_maps.py` | Google Maps Platform service adapter: one cheapest-request probe per category (geocoding on the v4beta surface, places ids-only, directions duration-only), google.rpc error translation including the 400-means-bad-key mapping. |
| `_livekit.py` | LiveKit service adapter: the RoomService ListRooms probe over Twirp on the caller's project host, with the two 401 bodies translated apart (bad signature vs missing roomList grant). |
| `_gemini.py` | Google Gemini: text, streaming, embeddings, image generation and edits (pictures sent inline, aspect ratio and seed in the generation config), speech and video generation, realtime sessions, the inline batch dialect (model in the URL, results on the operation object). One usage helper counts thoughts in the output and tool-use prompt tokens in the input on every parser. A bare refusal repeats the provider's own finishReason rather than reporting a missing image. |
| `_openai_compat.py` | The shared chat-completions adapter (DeepSeek, Moonshot, xAI, Perplexity, custom targets): usage normalization including reasoning tokens (added into the output where a subclass flags them as reported beside it) and Perplexity's per-request cost units, streaming assembly, tool calls. |
| `_moonshot.py` | Moonshot override: the `$web_search` builtin's echo-back handshake; batch rides the shared file dialect against chat completions. |
| `_perplexity.py` | Perplexity override: catalog-maintained Sonar models, the `max_output_tokens >= 16` floor gate, and the 400 "Invalid model"/"deprecated" mapping to `MODEL_NOT_AVAILABLE`. |
| `_xai.py` | xAI override: flags chat-completions reasoning as reported beside the output, `/v1/responses` routing for web search, reasoning effort, and code_interpreter, image generation and edits (one picture or several by URL or data URL), video generation, realtime sessions with Grok Voice appended from the catalog, the container batch dialect with counter-derived status and paginated results. |
| `_realtime.py` | Realtime wire adapters (OpenAI, xAI, Gemini) mapping session events to normalized types. |
| `_live.py` | The gpt-live full-duplex wire translator (OpenAI), mapping probe-confirmed session frames to normalized `LiveEvent`s. |
| `_stt.py` | AssemblyAI and Deepgram: credential-validating discovery, streaming transcription frames to normalized events (including each provider's own speaker-label dialect under `diarize=True`), Deepgram's one-round-trip file transcription, AssemblyAI's job-shaped one (upload, submit, poll), and AssemblyAI's one-shot dictation (verbatim transcript plus cleaned rewrite). |
| `_elevenlabs.py` | ElevenLabs: live speech-model discovery plus catalog STT entries, voice listing, speech generation, file transcription (multipart, or a source_url form), and a streaming-transcription translator over its JSON-message wire; a diarized session refuses, since its realtime wire never fills the speaker field. |
| `_ideogram.py` | Ideogram: a free dry-run key check, the catalog model list, per-model routes for generation, edits (with masks redrawn black-for-edit), enlarge, extend, background removal and replacement, erase, describe, and layerize, the job poll for its hosted models, result downloads from its signed links, text blocks, the six design tools, hosted video models from text, frames, or references, and error translation for its three error-body forms. Text generation refuses toward the picture methods. |
| `_typesafe.py` | TypeSafe: live alias listing as decision models, the judgment wire (typed questions in, typed answers with probabilities out, score legends normalized back into rubric order), the 255-option choice pre-flight, and error translation for the provider's three detail forms. |

## Viewer (`src/keycall/viewer/`)

| File | What it does |
|---|---|
| `__init__.py` | `run()`: starts the server, prints the tokened URL, opens the browser, optional `--reload` restart loop. |
| `_server.py` | Localhost stdlib HTTP server: token handshake to an httpOnly cookie, CSRF checks, static files with `no-store`, WebSocket upgrade. |
| `_api.py` | Every `/api/*` route: key checks, model listing, playground generation, picture operations, provider tools, model lists filtered by picture operation, file transcription, dictation, judgments, verify runs, settings, conversations, serialization; the per-provider capability map covers loaded custom targets too. |
| `_registry.py` | Server-side target registry mapping integer ids to live clients; conversation store (with each conversation's task setup); read-timeout rebuilds. |
| `_traces.py` | In-memory request-outcome log for the Traces tab (timing and status only). |
| `_realtime_bridge.py` | Bridges the browser's voice WebSocket to a `realtime()` session. |
| `_transcription_bridge.py` | Bridges the browser's transcribe WebSocket to a `transcribe_stream()` session. |
| `_ws.py` | Minimal WebSocket frame codec for the bridges. |
| `auth.py` | Per-run token generation and constant-time comparison. |
| `static/index.html` | The single page: five tabs, dialogs, composer. |
| `static/app.js` | All frontend behavior: tabs and URL routing, playground tasks, gating, history (foldable), traces, voice/transcribe audio, file transcription, dictation, judgments (with the draft kept per viewer in `localStorage`), the picture tasks (Change a picture, Run a design tool, video frames) with their per-key gates. |
| `static/markdown.js` | The reply renderer's small markdown subset. |
| `static/styles.css` | All styling, including the alias badge's instant hover tooltip. |

## Tests (`tests/`)

One file per surface, adversarial-first. `test_live.py` (deselected by default, `-m live`) holds the live smokes and capability-drift probes; `test_docs.py` is the docs-hygiene guard; `tests/js/markdown.test.mjs` covers the frontend renderer via `node --test`. The rest mock the wire per feature: adapters, client, CLI, streaming, tools, caching, realtime, live full-duplex sessions (`test_live_sessions.py`), transcription, viewer, sources, transport, types, tracing, hardening, alias facts, classification, credential, registry, embeddings, image/speech/video generation, picture operations across providers (`test_image_operations.py`), Ideogram end to end (`test_ideogram.py`), mask conversion (`test_mask.py`), batch generation (`test_batch.py`), prerecorded transcription (`test_transcribe.py`), one-shot dictation (`test_dictate.py`), judgments (`test_judge.py`), structured output, web search, hosted code execution (`test_code_interpreter.py`), reasoning effort, async parity, the retired-model gate, listing filter, and catalog invariants (`test_retired_models.py`), the sampling and seed gates (`test_hardening.py`), the ElevenLabs adapter with voice listing (`test_elevenlabs.py`), the service providers end to end (`test_service_providers.py`), and the docs-vs-code release gate (`test_shiplock.py`).

## Everything else

| File | What it does |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, the `keycall` entry point, pytest config. |
| `launch.sh` | Double-click launcher for Linux/macOS: finds a key file (argument, then conventional locations), then runs the viewer. |
| `launch.command` | macOS double-click launcher: delegates to `launch.sh` in its own directory. |
| `launch.bat` | Windows double-click launcher: the `launch.sh` equivalent for `cmd`. |
| `keycall-test-keys.example.toml`, `keycall-test-keys.example.txt` | Placeholder-only examples of the verify/viewer key-file format, one per accepted syntax, service targets included. |
| `.github/workflows/ci.yml` | Push/PR gate: tests, lint, JS tests; live smoke on manual dispatch only. |
| `.github/workflows/release.yml` | Tag-driven release: build, tests, live-strict verification, PyPI publish, GitHub release. |
| `.github/workflows/release-gate.yml` | Calls ShipLock's reusable gate: deterministic docs-vs-code checks plus the semantic audit, routed to the audit key's own provider. Manual dispatch until a hand-run passes. |
| `shiplock.toml` | Declares the doc surfaces, source globs, and version files the ShipLock gate checks. |
| `README.md`, `USAGE.md`, `ARCHITECTURE.md`, `CHANGELOG.md` | The public doc set. |
