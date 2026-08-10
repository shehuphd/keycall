# Changelog

All notable changes to KeyCall are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] — 2026-08-10

### Added

- **The Playground sends recordings and documents.** Both already worked
  from the library and neither had a control, so a caller could send a WAV
  or a PDF that a viewer user couldn't. Tick **Play it a sound** or
  **Attach a document** and it goes out as an `AudioInput` or
  `FileInput` on the user turn. The browser posts base64 and the server
  decodes it, the same route pictures take, so the Playground exercises the
  path a library caller uses rather than a viewer-only shortcut. A document
  keeps the filename it had on disk, because providers show that name to
  the model. Either attachment can carry a turn with no prompt.
- **You can record straight into the Playground** instead of finding a
  sound file first. The microphone sits in the message box, where the chat
  apps people already use put it, rather than buried under a settings
  toggle. Pressing it swaps the composer for a recording bar: a live
  waveform, a running clock, and two choices, keep or discard. Enter or the
  tick keeps it, Escape or the cross bins it, and a kept recording appears
  above the box with a player so a silent take is caught here rather than
  blamed on the model. With a recording attached and nothing typed, Enter
  sends, because there's no line to break and reaching for a modifier to
  send a voice message is a step nobody expects.

  The microphone is disabled on a key that can't send audio, naming a key
  that can, and hidden entirely in picture mode, on the same catalog
  reading the attachment toggles use. Blocked or missing microphones say
  which it was instead of failing quietly. A turn carrying only a recording
  draws that recording's own shape in the transcript rather than reading
  "(no message)", which said the opposite of what happened, and carries a
  play button: every recording in a session stays playable, because each
  turn holds its own copy rather than sharing the composer's. Sending
  detaches whatever went with the turn, so a recording or a picture can't
  ride along silently on the next one.

  Audio is captured as raw samples and encoded to WAV in the browser rather
  than handed to `MediaRecorder`, which looks like the obvious tool and is
  wrong twice over: Chrome produces `audio/webm;codecs=opus`, which Gemini
  doesn't accept and KeyCall's byte sniffer doesn't recognise, so the
  attachment would be refused before leaving the machine, and the container
  differs per browser anyway (mp4 on Safari, ogg on Firefox). WAV is one
  format every browser can produce, Gemini accepts, and the sniffer already
  identifies. It is downsampled to 16 kHz mono, which is what Gemini
  resamples to regardless, and which keeps a minute of speech near 2 MB
  instead of the ~11 MB that 48 kHz stereo would cost against an 8 MB body
  cap.
- **An attachment the selected key can't send is turned off, and says
  why.** Recordings reach only Gemini, and documents only OpenAI,
  Anthropic, and Gemini, so the control is dimmed with a line naming which
  of your loaded keys to pick instead. The list is read from the same
  catalog the adapters gate on, so a suggestion can never point at a
  provider that would refuse. Previously the only way to learn this was to
  attach a file and spend a round trip on the refusal.
- **A release probe measures how much of the attempt budget each provider
  actually needs**, failing if a working model isn't reached at least
  three attempts inside `DEFAULT_ATTEMPTS`. The walk has always assumed a
  healthy model appears early, nothing tested it, and it has now drifted
  twice (Gemini withdrawing six of its first eight advertised models,
  OpenAI killing all four aliases) with both found by accident. Asserting
  only that some model works would report the problem after users hit it;
  requiring a margin reports it while there's still room. A retired model
  refuses without charging, so the probe costs one generation per
  provider.
- `Model.released_at` carries the provider's own date for a model where it
  reports one, which is what the ordering rule reads.

### Changed

- **The Playground's two columns scroll independently.** They shared a
  height, so a long reply grew the page and pushed the settings out of
  reach, and a tall settings panel stretched the conversation to match.
  Each column is now bounded by the window and scrolls its own content;
  below 900px they stack and the page scrolls as before. The available
  height is measured rather than assumed, since a hardcoded header offset
  leaves either a dead gap or a second scrollbar.
- **The composer's button says Send, not Generate**, which was wrong the
  moment a turn could be a recording with nothing generated about it.
- **The Playground's model picker leads with the model the walk would try
  first, and drops one the provider has refused.** It listed models in raw
  provider order, so on Gemini it defaulted to `gemini-2.5-flash` — the
  model the walk already skips, withdrawn from new accounts — and a first
  generation could fail on a key that works perfectly. The picker now
  applies the same ordering rule `verify` uses, so a maintained alias leads
  on Gemini and the newest model leads on OpenAI. When a provider does
  refuse a model with `model_not_available` or `model_not_suitable`, that
  model drops to the bottom of the list for the session, is labelled
  "refused earlier", and can't be picked again; the selection falls back
  to one that hasn't been turned down. This is learned from what the
  provider answered rather than shipped as a list of retired models, which
  would be wrong for somebody the day it shipped.

- **Candidate order follows the provider's own dates where it publishes
  them** (`SELECTION_RULE_VERSION` is now `4`). OpenAI, Anthropic, and
  Moonshot date every model they list, and a recently published model is
  one the provider hasn't yet retired, so the walk tries the newest
  first. Gemini and DeepSeek publish no dates, and maintained `-latest`
  aliases still lead there.

  Sorting aliases first everywhere was right for Gemini and wrong for
  OpenAI, which is retiring its `*-chat-latest` family wholesale: on
  2026-08-10 all four were dead, two unknown and two newly deprecated
  hours after they had worked, while the numbered models stayed healthy.
  That rule spent four of eight attempts before reaching a model that
  could answer, and image input failed outright because the first survivor
  was too old to have vision. Under the new rule every provider reaches a
  working model at position 1 of 8.

### Fixed

- **`Model.context_limit` now reads every provider's spelling.** It had
  been populated on Gemini alone, from `inputTokenLimit`. Probing the
  endpoints directly showed three of the six report an input ceiling under
  three different names, so Anthropic's `max_input_tokens` and Moonshot's
  `context_length` are read into the same field. OpenAI and DeepSeek report
  nothing and Perplexity has no list endpoint, so it stays `None` there,
  meaning "this provider doesn't say" rather than zero. It is still never
  inferred from a bundled table: a caller budgets against this number, and
  an invented ceiling is worse than an absent one. The viewer's "Context
  window" column appears for the providers that fill it.
- `ImageInput`, `AudioInput`, and `FileInput` carried docstrings saying the
  type wasn't accepted yet and that every adapter refused it before any
  network call. That stopped being true when the media paths shipped. Each
  now states which providers take it and in which form.
- **A truncated reply now says so, in one vocabulary.** Each provider names
  a spent output budget differently — `incomplete:max_output_tokens` on
  OpenAI, `max_tokens` on Anthropic, `MAX_TOKENS` on Gemini, `length` on
  the Chat Completions family — so noticing that an answer was cut off
  meant knowing all four. Results now carry a warning saying what happened
  and what to change, and it explains the trap people hit: on a reasoning
  model the hidden reasoning is charged to the same budget, so a small
  `max_output_tokens` can be spent before any text appears.
- **The viewer renders markdown in replies.** Models answer in markdown
  whether or not you ask them to, so headings, bold, lists, links, and
  fenced code arrived as punctuation. Parsed into DOM nodes with no
  `innerHTML` anywhere, because model output is untrusted: `javascript:`
  and `data:` links are shown as plain text rather than made clickable,
  and any HTML in a reply stays literal text. The viewer also shows result
  warnings on the non-streaming path, where they had never been rendered.
- **The viewer's reply budget defaults to 2048, not 256**, and is labelled
  "Reply budget" rather than "Longest reply". The old default truncated
  most reasoning-model answers before the first word.
- **The DNS-rebinding guard could raise instead of deciding.** It passed
  the first element of every resolved `sockaddr` to the private-address
  check, which parses it as an IP. That element is only a string for
  AF_INET and AF_INET6; another address family would have raised inside the
  guard rather than returning a verdict. Non-string entries are now dropped
  before the check, and dropping them all leaves the existing "resolved to
  no addresses" refusal to fire, which is the safe direction.
- **The viewer could return a 500 when a model list expired mid-request.**
  `browse_models` re-read the cached listing after refreshing it, and that
  entry carries the same 5-minute TTL as the library cache, so expiry
  between the write and the read left it reading attributes off `None`. It
  now returns a `cache_expired` error asking for a retry.
- **The package's type annotations are checked again.** KeyCall ships
  `py.typed`, so its annotations are a promise to callers' type checkers,
  but nothing ran mypy in CI and the error count had drifted to 27 across
  several releases. All are fixed, none suppressed (two stale
  `type: ignore` comments were deleted), and `mypy src` now runs in both CI
  and the release workflow. The two defects above were found this way.

## [0.9.0] — 2026-08-10

### Added

- The viewer Playground sends images: tick **Image**, pick a file or paste
  a URL, and it goes out as an `ImageInput` on the user turn. The
  browser posts the file as base64 and the server decodes it, so the
  Playground exercises the same path a library caller takes rather than a
  viewer-only shortcut. An image can carry a turn with no prompt. The
  viewer's request-body cap moved from 64 KB to 8 MB to fit an encoded
  photo.

### Changed

- **`AsyncKeyCall` is now covered, not assumed.** It is half the public API
  and had two unit tests and no live coverage, so tool calling, streamed
  tool events, and image input had all shipped verified on the sync path
  only. There is now an async suite exercising each of them plus the
  pre-network gates, stream truncation, and round-trip timing, a signature
  check that fails if a parameter is added to one client and not the other,
  and a live async test in the release gate. No defects turned up: the
  async path was correct, it was untested.
- **The Playground makes pictures.** A "Task" selector switches between
  writing text and making a picture; picture mode offers only that key's
  image models, hides the controls that don't apply, and shows the result
  inline, capped so it can't push the page around, with a click to see it
  full size and a link to save it. The full-size view carries a close
  button rather than relying on Escape or a backdrop click, neither of
  which a first-time user can see. The composer clears once a turn is on
  screen, so a prompt can't be sent twice by accident. The viewer's content-security policy now
  allows `data:` images so the picture can render from bytes the page
  already holds, without writing it to disk or fetching anything remote.
- **The Playground is a two-column workspace.** Settings live in a left
  sidebar grouped as "Key and model" and "Extras"; the conversation owns
  the right, with turns as bubbles and a composer pinned at the bottom
  that also sends on Ctrl/Cmd+Enter and stays level with the bottom of the
  settings rather than floating mid-page. The single stacked column made the
  controls compete with the reply for attention, and the reply had nowhere
  to grow. Optional features are switches rather than checkboxes, so their
  state reads at a glance, and each reveals its settings inline.
- **The viewer is written for someone who has never seen it.** Each tab now
  opens with a heading and a plain-language explanation of what the screen
  does and what it costs. Controls are labelled by what they do rather than
  by the API field behind them: "Show the model a picture" instead of
  "Image", "When to use a tool" instead of "tool_choice", model categories
  as "Writes text" rather than `text_generation`. The picture panel says
  outright that it sends a picture to a model and can't make one, because
  "Image" read as image generation to a first-time user. Empty results are
  a designed state everywhere, saying what happened and what to try next,
  and the Generate and Run verify buttons report progress in their label
  instead of only greying out. Tab labels also line up with the header and
  content, which they hadn't since the nav and main used different
  gutters, and a hint that introduces a row of controls no longer sits
  crammed against them.

### Fixed

- **Reloading the viewer logged you out.** The access token arrives in the
  page URL and is stripped from the address bar so it doesn't linger in
  browser history, but it was only held in a page variable, so a plain
  refresh left the page with no token and it died with "Not authorized".
  The token now lives in `sessionStorage` for the tab: reloads work, the
  address bar stays clean, and closing the tab still discards it. A token
  the server no longer accepts (because KeyCall was restarted) is cleared
  rather than kept to fail again, and the message says to use the newest
  link the terminal printed.

## [0.8.0] — 2026-08-09

### Added

- **Image generation** (`generate_image()` on both clients), KeyCall's third
  operation. Supported on OpenAI, which answers on a dedicated
  `/images/generations` endpoint with base64 PNG, and on Gemini, whose
  image models answer on the ordinary content endpoint with an inline JPEG
  part; the two shapes normalize to the same `ImageOutput` parts. The
  request is deliberately a model and a prompt only, because OpenAI's size
  and count have no equivalent on Gemini. `media_type` is read from the
  response rather than assumed, so saved bytes get the right extension. A
  response carrying no image raises instead of returning an empty result,
  and when Gemini answers a refusal in words the error repeats what the
  model said. Anthropic, DeepSeek, Perplexity, and Moonshot generate no
  images and refuse before any network call.
- **Embeddings** (`embed()` on both clients), KeyCall's second operation.
  One `EmbeddingOutput` per input, in input order, so results zip against
  the strings that produced them; a provider returning a different count
  raises rather than handing back vectors that silently misalign with a
  caller's index. Supported on OpenAI (1536 dimensions) and Gemini (3072),
  both batching every input into one request. Anthropic publishes no
  embeddings endpoint and DeepSeek, Perplexity, and Moonshot return 404 or
  403 for one (verified 2026-08-09), so those refuse before any network
  call. This closes the asymmetry where `list_models` could surface
  embedding models that nothing in KeyCall could then invoke.
- **`Usage.provider_units` carries non-token billing.** Perplexity reports
  a `cost` object whose `request_cost` is charged per call rather than per
  token (verified 2026-08-09), which a token budget can't see. Numeric
  entries now surface as `(name, value)` pairs; descriptive fields like
  `search_context_size` aren't units and are left out. The field had
  existed since 0.2.0 with nothing ever assigning it, the same broken
  promise `catalog_stale` carried. Providers that report no cost leave it
  `None`, meaning not reported rather than zero.
- **Audio and document input.** `AudioInput` and `FileInput` join
  `ImageInput` as first-class content parts. Documents (PDF) work on OpenAI,
  Anthropic, and Gemini; audio is Gemini-only, because OpenAI's Responses
  API has no audio content part and Anthropic, Moonshot, and Perplexity
  each reject one (all verified 2026-08-09 by sending a WAV and a PDF).
  Media type is detected from the content for every kind, and a refusal
  names the providers that do accept it. This closes the last of the
  half-promised input types: nothing in the content taxonomy is now
  exported without either working or saying precisely why it can't.
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
  through, so this doesn't become a blanket ban on sampling parameters.
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

- `run_verify()` no longer raises when a target can't be resolved (an
  unknown provider name with no protocol and base URL). It reports that
  target as `unresolvable_target` and carries on, so one malformed entry
  in a key file stops aborting verification of every other target.
- A feature the provider has but the chosen model lacks now raises
  `MODEL_NOT_SUITABLE` instead of `INVALID_PROVIDER_RESPONSE`. OpenAI
  supports web search, but `gpt-3.5-turbo` refuses the tool, and calling
  that a malformed response pointed the caller at their own request rather
  than at the model. The error repeats the provider's wording and adds
  that another model will do it.
- The dashboard reports one model count per key, across every kind, and
  points at the Models tab for the breakdown.
- The model browser's Context column was blank on every row for five of
  six providers, because only Gemini's list endpoint reports a token
  limit. It now appears only when a provider fills it, explains itself on
  hover, and formats the number with separators. Category cells read
  "Writes text" rather than `text_generation`, matching the filter
  dropdown.
- **The viewer's static files are served `Cache-Control: no-store`.** They
  carried no cache header at all, so a browser cached the page and its
  script heuristically and an open tab kept running the previous version's
  JavaScript against the new server. That is what made a control appear
  dead or a status hang after an upgrade, and a plain reload didn't clear
  it. This was the cause behind more than one "the UI isn't updating"
  report.
- The viewer's model browser could hang on "Asking the provider…" forever
  if the page threw while loading. Any failure now lands in a visible
  panel with a way forward, and empty states are left-aligned with the
  rest of the page instead of centred.
- HTTP 402 is reported as `PERMISSION_DENIED` rather than falling through
  to `INVALID_PROVIDER_RESPONSE`. A 402 means the key is valid and the
  account is unfunded or on a billing hold, so calling it a malformed
  response sent callers hunting for a bug in their own request. The
  provider's message, which carries the billing link, is preserved.
  Anthropic already mapped it this way; the base and Gemini adapters now
  match.
- **OpenAI reasoning models rejected any replayed tool call.** When the
  model emits a `reasoning` item alongside a `function_call`, the call
  can't be replayed without it: the next request fails with HTTP 400
  naming the missing item. KeyCall discarded reasoning items as
  server-side traces, so every tool round on a reasoning model broke at
  the second turn. The item now travels in `ToolCall.opaque` and is
  replayed ahead of the call it belongs to, once even when parallel calls
  share it. Reasoning items appear only when the model actually reasons,
  which is why this survived the earlier live rounds; found by driving the
  Playground against gpt-5.3-chat-latest.
- `ImageInput`, `AudioInput`, and `FileInput` now say plainly that text
  generation doesn't accept them yet, in the error and on the types
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
  alternatives read the same data, so they can't drift apart, and adding a
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

- OpenAI, like Gemini, lists models an account can't invoke:
  `gpt-5-chat-latest` and `gpt-5.1-chat-latest` were both advertised and
  both returned "Model not found" on 2026-08-10, while `gpt-5.2` and
  `gpt-5.3` worked. The verify walk handles it by design, and the README
  now records it against OpenAI rather than only Gemini.

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
  small fraction of the elapsed duration — about 1% on Anthropic, which read
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
  tools, so the gate can't silently outlive its evidence.

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
  false`. One schema can't satisfy both providers; KeyCall passes the
  caller's schema through unmodified to either.

## [0.4.1] — 2026-08-08

### Changed

- Code comments and docstrings state their constraints directly instead of
  citing internal design documents that aren't part of the repository.

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
  default) running the verify walk against live providers; a manual-only
  `live-warn` CI job that reports without failing; and a `live-strict`
  release gate that blocks publishing until every target verifies,
  including when the `KEYCALL_LIVE_TARGETS` secret is absent. Credentials
  load only at test run time from `KEYCALL_LIVE_SOURCE`.

## [0.3.1] — 2026-08-08

### Fixed

- The README status line no longer hardcodes a version number; the 0.3.0
  PyPI page showed "early release (0.2.0)" because the line wasn't bumped
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

- Anthropic can't combine `web_search=True` with `response_schema` in one
  call (forced tool_choice is mechanically incompatible with also invoking
  a second server-side tool); KeyCall raises before any network call.
- Gemini's equivalent combination isn't gated — no live-verified evidence
  either way that Gemini itself rejects it, so it's passed through rather
  than guessed at.
- OpenAI's strict `json_schema` mode requires `additionalProperties: false`
  at every object level of the caller's schema, or the request 400s. This is
  an OpenAI requirement; KeyCall doesn't rewrite caller-supplied schemas to
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
  filters, playground for live generation calls (including web search with
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
- Usage fields distinguish "provider reported zero" from "provider didn't
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
  is absent or the host hasn't configured it; disabled with one warning on an
  incompatible version.

### Known limitations

- Streaming, tool calling, structured output, and non-text modalities aren't
  implemented.
- Gemini's list endpoint advertises models an account can't invoke and exposes
  no lifecycle field, so they can't be pre-filtered.
- Perplexity's Sonar models aren't API-discoverable and are maintained in the
  bundled catalog.
- The provider catalog ships inside the package and updates only on release.

[Unreleased]: https://github.com/shehuphd/keycall/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/shehuphd/keycall/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/shehuphd/keycall/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/shehuphd/keycall/compare/v0.7.0...v0.8.0
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
