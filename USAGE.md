# KeyCall Usage

Full reference for the Python API and the `keycall` CLI. For a quick
overview, see [README.md](README.md); for version history, [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
pip install keycall
```

Python 3.10+. Optional extras: `pip install "keycall[traceact]"` for tracing.

## Clients

Construct one client per provider and credential. Identity is fixed at
construction; switching provider, key, protocol, or base URL means
constructing a new client.

```python
from keycall import KeyCall

with KeyCall(provider="openai", api_key=secret) as client:
    ...
```

Async is identical except for `await`:

```python
from keycall import AsyncKeyCall

async with AsyncKeyCall(provider="anthropic", api_key=secret) as client:
    discovery = await client.list_models()
```

Supported provider names: `openai`, `anthropic`, `gemini`, `deepseek`,
`perplexity`, `moonshot`. Aliases: `claude`, `google`, `google-gemini`,
`pplx`, `kimi`.

### Custom OpenAI-compatible endpoints

An unknown provider name requires an explicit protocol and base URL:

```python
client = KeyCall(
    provider="university-lab",
    protocol="openai-compatible",
    api_key=secret,
    base_url="https://llm.example.edu/v1",
)
```

Rules for `base_url`: absolute HTTPS, no query string, fragment, or
userinfo. Plain HTTP is allowed only for localhost with
`allow_insecure_localhost=True`. Literal private/internal IP addresses
require `allow_private_network=True`. Hostnames are DNS-pinned per request:
KeyCall resolves once, refuses to proceed if any resolved address is
private, and connects to the validated address while TLS still verifies the
original hostname.

### Constructor options

| Parameter | Default | Purpose |
|---|---|---|
| `provider` | required | Provider name or custom label |
| `api_key` | required | The credential; wrapped in a redacting type immediately |
| `protocol` | from registry | Wire protocol; only needed for custom targets |
| `base_url` | from registry | Only for custom targets |
| `connect_timeout` | `10.0` | Seconds |
| `read_timeout` | `60.0` | Seconds |
| `max_response_bytes` | 10 MB | Response bodies are read incrementally against this cap |
| `trust_env` | `True` | Set `False` to ignore `HTTP_PROXY`/`HTTPS_PROXY` |
| `allow_insecure_localhost` | `False` | Permit `http://localhost` targets |
| `allow_private_network` | `False` | Permit literal private-IP targets |

## Listing and filtering models

```python
from keycall import ModelCategory

discovery = client.list_models()                                    # text models (default)
images = client.list_models(categories={ModelCategory.IMAGE_GENERATION})
both = client.list_models(
    categories={ModelCategory.TEXT_GENERATION, ModelCategory.IMAGE_GENERATION}
)
```

Categories: `TEXT_GENERATION`, `IMAGE_GENERATION`, `EMBEDDING`,
`TRANSCRIPTION`, `SPEECH_GENERATION`, `VIDEO_GENERATION`, `REALTIME`,
`UNKNOWN`. Models KeyCall cannot classify are `UNKNOWN` and appear only when
you request that category explicitly, never in the default text picker.

`ModelDiscovery` fields: `models`, `provider`, `categories`, `fetched_at`,
`from_cache`, `catalog_version`, `warnings`. Each `Model` carries `id`,
`provider`, `categories`, `display_name`, `context_limit`,
`classification_source`, and `warnings`.

Results are cached in-process for 5 minutes, keyed by an HMAC fingerprint of
the credential. Force a live call with `client.list_models(refresh=True)`,
always do this when verifying a newly entered key.

A successful listing proves the credential works for discovery. It does not
prove every listed model can be invoked; some providers advertise retired or
quota-walled models with no lifecycle field to filter on.

## Generating text

```python
from keycall import Message, TextInput

result = client.generate_text(
    model=discovery.models[0].id,
    messages=[
        Message(role="system", content=[TextInput(text="Be concise.")]),
        Message(role="user", content=[TextInput(text="Hello.")]),
    ],
    max_output_tokens=200,
    temperature=0.7,   # optional; omitted from the wire when unset
    top_p=0.9,         # optional
)

result.text                      # concatenated text output, or None
result.usage.input_tokens        # None means "provider didn't report", not zero
result.usage.total_tokens
result.round_trip_duration_ms
result.finish_reason
result.provider_request_id
```

`messages` accepts any sequence of `Message` objects, a plain list is fine.
Roles are `"system"`, `"user"`, `"assistant"`. Dicts and bare strings are
not accepted; there is one canonical representation.

The lower-level path accepts a typed request, useful when you build requests
in one place and execute them in another:

```python
from keycall import TextGenerationRequest

request = TextGenerationRequest(model="...", messages=[...], max_output_tokens=64)
result = client.invoke(request)
```

Some models reject sampling parameters outright (OpenAI o-series and gpt-5;
Anthropic Opus 4.7+, Opus 5+, Sonnet 5+). Passing `temperature` or `top_p`
for those raises `MODEL_NOT_SUITABLE` before any network call, omit the
parameters for those models.

## Streaming

`stream_text()` takes the same parameters as `generate_text()` and delivers
the response incrementally. Use it as a context manager, iterate the typed
events, then call `result()` for the same `InvocationResult` a non-streamed
call returns:

```python
with client.stream_text(model="gpt-4o-mini", messages=messages) as stream:
    for event in stream:
        if event.kind == "text_delta":
            print(event.text, end="", flush=True)
    result = stream.result()

print(result.usage.total_tokens, result.finish_reason)
```

`AsyncKeyCall.stream_text()` is the awaitable twin: `async with` and
`async for`.

Event types, discriminated by `kind`:

| Event | `kind` | Carries |
|---|---|---|
| `StreamStart` | `stream_start` | model id the provider confirmed |
| `TextDelta` | `text_delta` | a text increment; with `response_schema`, a fragment of the final JSON |
| `CitationFound` | `citation` | one web-search source, as it surfaces |
| `ToolCallStarted` | `tool_call_started` | id and name of a call beginning; arguments not known yet |
| `ToolCallArgumentsDelta` | `tool_call_arguments_delta` | a raw fragment of that call's argument JSON |
| `ToolCallComplete` | `tool_call_complete` | the assembled `ToolCall`, arguments parsed |
| `StreamFinish` | `stream_finish` | finish reason and usage |
| `UnknownStreamEvent` | `unknown` | bounded provider kind for content KeyCall doesn't recognize yet |

Behavior and guarantees:

- `web_search`, `response_schema`, and `tools` combine with streaming on
  every provider that supports them non-streamed; the same gates apply
  (Anthropic still refuses the web_search + response_schema combination).
- Streaming is never retried, before or after the first byte.
- The stream must end with the provider's own terminal signal. A connection
  that closes early raises `NETWORK_ERROR` from the iterator, and
  `result()` raises rather than returning a silent partial.
- Response-size caps apply to the total stream and to each individual
  event; the read timeout applies between chunks, so a stalled stream
  raises `TIMEOUT`.
- Leaving the `with` block closes the connection, including on early
  `break`. `result()` before the stream finishes raises rather than
  silently consuming the rest.
- Custom OpenAI-compatible targets stream with the `[DONE]` terminal
  convention; usage may be unreported there and surfaces as the standard
  missing-usage warning.

## Tool calling

Define tools, receive the model's call requests as typed parts, execute on
your side, and send results back. KeyCall normalizes all four wire shapes;
it never executes a tool and never runs the loop — that stays your code:

```python
from keycall import Message, TextInput, Tool, ToolResult

weather = Tool(
    name="get_weather",
    description="Get current weather for a city",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}},
                  "required": ["city"]},
)

messages = [Message(role="user", content=[TextInput(text="Weather in London?")])]
result = client.generate_text(model="...", messages=messages, tools=[weather])

while result.tool_calls:
    replies = []
    for call in result.tool_calls:            # several per turn is normal
        output = my_dispatch(call.name, call.arguments)
        replies.append(ToolResult(tool_call_id=call.id, name=call.name, content=output))
    messages += [result.to_assistant_message(),
                 Message(role="user", content=replies)]
    result = client.generate_text(model="...", messages=messages, tools=[weather])

print(result.text)
```

Rules and behavior:

- `to_assistant_message()` replays the model's turn, including provider
  echo data some providers require back verbatim (`ToolCall.opaque`, e.g.
  Gemini's thought signature — never modify or interpret it).
- `ToolResult.content` may be a string or a JSON-serializable mapping;
  adapters convert to each provider's required form.
- `tool_choice` accepts `"auto"`, `"required"`, or `"none"`. Forcing one
  named tool is not yet supported. Some provider/model pairs reject
  `"required"` (DeepSeek thinking models return 400); the provider's typed
  error is surfaced.
- `web_search` combines with tools on OpenAI, Anthropic, and Gemini (where
  KeyCall sets the required `toolConfig` flag automatically).
- Perplexity has no tool calling and raises `UNSUPPORTED_OPERATION` before
  any network call; the live suite carries a drift probe that fails if
  that ever changes. Custom OpenAI-compatible targets pass through with a
  result warning that support is unverified.
- Not combinable: Anthropic tools + `response_schema`, because schema
  enforcement is itself a forced tool call.

### Streaming tool calls

`stream_text()` takes `tools` and `tool_choice` too. Calls arrive as three
events: `tool_call_started` when the model names a tool,
`tool_call_arguments_delta` as its arguments stream in, and
`tool_call_complete` once they parse. Act on the complete event only:

```python
with client.stream_text(model="...", messages=messages, tools=[weather]) as stream:
    for event in stream:
        if event.kind == "text_delta":
            print(event.text, end="", flush=True)
        elif event.kind == "tool_call_started":
            print(f"\n[calling {event.name}]")
        elif event.kind == "tool_call_complete":
            pending.append(event.tool_call)
    result = stream.result()
```

- `result.tool_calls` after a stream matches what the same request returns
  non-streamed, so the loop above and the `generate_text()` loop can share
  their dispatch and replay code.
- Argument fragments are provider bytes, not KeyCall's: they split
  mid-token and only the concatenation is valid JSON. Show them as
  progress, never parse them. A provider that sends arguments whole
  (Gemini) emits no fragments at all, so treat their absence as normal.
- Malformed argument JSON raises `INVALID_PROVIDER_RESPONSE` from the
  iterator rather than yielding a call with silently dropped arguments.

## Image input

Pass an `ImageInput` alongside your text in a user message. Bytes are read
directly; no file is uploaded and KeyCall never fetches anything on your
behalf:

```python
from keycall import ImageInput, Message, TextInput

result = client.generate_text(
    model="gpt-5.3-chat-latest",
    messages=[Message(role="user", content=[
        TextInput(text="What is in this photo?"),
        ImageInput(data=photo_bytes),          # or ImageInput(url="https://…")
    ])],
)
```

Support splits by *form*, not only by provider, and the gate fires before
any network call:

| Provider | Image bytes | Image URL |
|---|---|---|
| OpenAI | yes | yes |
| Anthropic | yes | yes |
| Gemini | yes | no |
| Perplexity | yes | yes |
| Moonshot | yes | no |
| DeepSeek | no | no (API is text only) |

- **Bytes are the portable form.** Every image-capable provider accepts
  them, so `ImageInput(data=...)` works everywhere images work.
- **A URL is only sent, never fetched.** Providers that refuse remote URLs
  raise `UNSUPPORTED_OPERATION` telling you to pass bytes. KeyCall will not
  download it for you: an adapter making its own request could be pointed
  at anything by caller-supplied data.
- **The media type is detected from the content** (PNG, JPEG, GIF, WebP).
  A `media_type` you supply is used only for formats KeyCall doesn't
  recognize, because Anthropic and Gemini both reject a mismatched type and
  the bytes are the better evidence. An image KeyCall can't identify raises
  rather than being sent with a guess.
- Images belong in user messages. `AudioInput` and `FileInput` are still
  unimplemented and refuse before the network, as before.

## Web search

Providers with a native server-side search tool can ground a generation in
live web results:

```python
result = client.generate_text(
    model="gpt-4o-mini",
    messages=[Message(role="user", content=[TextInput(text="What's new in Python?")])],
    web_search=True,
)

for citation in result.citations:
    print(citation.url, citation.title, citation.cited_text)
```

| Provider | Mechanism | Citation URLs |
|---|---|---|
| OpenAI | `web_search` tool (Responses API) | direct source URLs |
| Anthropic | `web_search_20250305` tool | direct source URLs, with `cited_text` |
| Gemini | `google_search` tool | Google `vertexaisearch` redirect links (by Google's design; they resolve to the source when followed) |
| Perplexity | Sonar always searches — the flag is a no-op | direct source URLs, with snippets |

DeepSeek, Moonshot, and custom OpenAI-compatible targets have no native
search tool: `web_search=True` raises `UNSUPPORTED_OPERATION` before any
network call rather than silently ignoring the request.

`result.citations` is a tuple of `Citation(url, title, cited_text)`,
normalized across all four provider response shapes. Fields the provider
didn't supply are `None`.

What the citation list does and doesn't guarantee:

- **One URL can legitimately appear more than once.** Providers cite per
  claim, not per source. Anthropic gives each citation its own
  `cited_text`, so three citations of one page are three different
  excerpts. KeyCall drops only citations that repeat an earlier one
  *exactly* — same URL, title, and excerpt — because those carry nothing
  the first didn't. Building a sources list means collapsing by URL
  yourself, and only you know whether to keep the longest excerpt, the
  first, or all of them.
- **The count can't be bounded in the request.** No provider exposes a
  "return at most N citations" parameter; Anthropic's `max_uses` limits how
  many searches the model runs, not how many sources come back. Slicing
  `result.citations` is the only option, and the tokens are already spent
  by the time you do. If capping matters, cap `max_output_tokens`, which
  does bound the answer the citations attach to.
- **Search costs are not reported.** `Usage.provider_units` exists for
  non-token billing units but no adapter populates it today, so a
  per-search charge (as opposed to per-token) does not appear anywhere in
  `result.usage`. A token budget built on `Usage` will not see it. Check
  the provider's own billing for search-priced calls.
- **`context_limit` is populated only where the provider's list endpoint
  reports it**, which today means Gemini alone. OpenAI, Anthropic,
  DeepSeek, Perplexity, and Moonshot all return model lists with no token
  limit, so `Model.context_limit` is `None` there. It is left `None` rather
  than filled from a bundled table, because a wrong window is worse than a
  missing one: KeyCall never guesses a number a caller may budget against.

## Structured output

```python
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "version": {"type": "string"}},
    "required": ["name", "version"],
    "additionalProperties": False,   # required by OpenAI's strict mode — see below
}

result = client.generate_text(
    model="gpt-4o-mini",
    messages=[Message(role="user", content=[TextInput(text="Name and version, as JSON.")])],
    response_schema=schema,
)

import json
parsed = json.loads(result.text)   # result.text is the JSON string on every provider
```

| Provider | Mechanism | Enforced? |
|---|---|---|
| OpenAI | `text.format={"type":"json_schema",...,"strict":true}` (Responses API) | yes |
| Anthropic | forces a single synthetic tool call, reads its input back | yes |
| Gemini | `generationConfig.responseSchema` | yes |
| Moonshot | `response_format={"type":"json_schema",...}` | yes |
| Perplexity | `response_format={"type":"json_schema",...}` | yes |
| DeepSeek | falls back to `response_format={"type":"json_object"}` | no — valid JSON guaranteed, schema conformance is not |
| Custom OpenAI-compatible targets | same fallback as DeepSeek, since capability is unverified | no |

On a non-enforcing provider, `result.warnings` explains that the schema
wasn't enforced, validate client-side there rather than trusting the shape.

Three provider requirements to know before writing a schema:

- **OpenAI's strict mode requires `additionalProperties: false`** on every
  object level of the schema, or the request fails with a 400. Write it in
  from the start rather than discovering it from an error.
- **Gemini rejects any `additionalProperties` key** in the schema with a
  400 (live-verified 2026-08-08) — the direct opposite of OpenAI's
  requirement. One schema cannot satisfy both providers; strip or add the
  key per provider before the call.
- **DeepSeek requires the word "json" somewhere in the prompt** for its
  fallback mode, or it 400s. KeyCall detects this and inserts a short system
  instruction automatically when needed — you'll see it noted in
  `result.warnings`, not applied silently.

`response_schema` and `web_search` cannot be combined on Anthropic (forcing
the structured-output tool prevents the model calling a different one in the
same turn); combining them raises `UNSUPPORTED_OPERATION` before any network
call. The same combination on Gemini is untested and not gated — KeyCall
passes it through rather than guessing at behavior it hasn't verified.

## Error handling

Every failure raises `KeyCallError` with a typed `code`:

```python
from keycall import ErrorCode, KeyCallError

try:
    discovery = client.list_models(refresh=True)
except KeyCallError as error:
    if error.code is ErrorCode.INVALID_API_KEY:
        show_settings_error("That key was rejected.")
    elif error.retryable:
        schedule_retry(after=error.retry_after)
    else:
        log(error.code.value, error.message)
```

| Code | Meaning | Retryable |
|---|---|---|
| `INVALID_API_KEY` | Provider rejected the credential | no |
| `PERMISSION_DENIED` | Key valid but not allowed (includes billing) | no |
| `RATE_LIMITED` | Rate or quota limit; `retry_after` set when provided | yes |
| `PROVIDER_UNAVAILABLE` | 5xx or overload | yes |
| `NETWORK_ERROR` | Could not reach the provider | yes |
| `TIMEOUT` | No response within the timeout | yes |
| `INVALID_PROVIDER_RESPONSE` | Malformed body, redirect, or oversized response | no |
| `MODEL_NOT_AVAILABLE` | Model missing, retired, or rejected by name | no |
| `MODEL_NOT_SUITABLE` | Model can't serve this request (e.g. sampling params) | no |
| `UNSUPPORTED_PROVIDER` | Unknown name or invalid custom target | no |
| `UNSUPPORTED_OPERATION` | Request shape not supported in this version | no |
| `CATALOG_UPDATE_REQUIRED` | Bundled catalog too old for this client | no |

Error messages are sanitized: no credentials, no raw request bodies, no
unsanitized provider text.

Retry behavior: model listing gets a small bounded retry budget for transient
failures, honoring `Retry-After`. Generation is never retried by KeyCall,
since no supported provider documents generation idempotency, so retrying an
ambiguous failure risks a second charge. `retryable` tells *you* whether a
retry is reasonable at your layer.

## The verify CLI

Live credential verification, one model-list call per target:

```bash
keycall verify --source ./keys.toml
```

Add one bounded generation per target:

```bash
keycall verify --source ./keys.toml --generate
```

With `--generate`, KeyCall walks the filtered text models in provider order
and reports every attempt until one succeeds (default budget 8, adjustable
with `--attempts`). Skipped models are printed with reasons, so retired
models, modality mismatches, and per-model quota walls stay visible. Each
attempt reports the model's position in both the filtered list and the
provider's raw list, plus the classification evidence that made it a
candidate; the result carries a digest of the raw model-list snapshot and
the selection-rule version, so a failure is reconstructable against the
provider surface that produced it.

### Live verification in CI

The same walk runs as a pytest suite with three modes:

| Mode | Where | Behavior |
|---|---|---|
| `off` | Every ordinary `pytest` run | Live tests deselected; no credentials touched |
| `warn` | CI `live-warn` job, manual dispatch only | Runs live smoke, reports failures, never fails the workflow |
| `strict` | Release workflow, before publish | Any unverified target blocks the release, including a missing credentials secret |

Select live tests explicitly with `pytest -m live`; they load credentials
only at run time, from the target file named by `KEYCALL_LIVE_SOURCE`. In
CI the file is written from the `KEYCALL_LIVE_TARGETS` repository secret
(TOML target syntax, below). Rate-limit outcomes are reported as
verification-environment failures, distinct from adapter or credential
failures; in strict mode they still block the release because the
release remains unverified. Fork pull requests receive no secrets and
never run live jobs.

### Sources

TOML:

```toml
[[targets]]
provider = "openai"
key = "sk-..."
name = "my-openai-key"
```

TXT (one target per line, `#` for comments, quotes optional):

```text
provider=openai key=sk-... name=my-openai-key
protocol=openai-compatible provider=my-lab base_url=https://llm.example.edu/v1 key=...
```

JSON: `{"targets": [{"provider": "...", "key": "..."}]}`.

Environment variable (single target, provider required):

```bash
keycall verify --source env:MY_OPENAI_KEY --provider openai
```

Interactive (no `--source`): prompts for provider and a hidden key.

Fields: `provider` and `key` required; `protocol`, `base_url`, `name`
optional. Repeating a provider creates independent targets.

### Behavior and exit codes

Keys never appear in output. Credential files are never modified or deleted.
A broadly readable file or one inside a git repository produces a warning;
`--strict-credentials` turns those warnings into errors. Exit codes: `0` all
targets verified, `1` at least one failed, `2` usage or source error.

Use dedicated low-budget test keys, and `chmod 600` the file.

## The viewer

```bash
keycall view --source ./keys.toml
```

Starts a local, token-protected web app over the loaded targets and opens it
in your browser. Four tabs:

- **Dashboard** — every loaded target; click one for a live key check and its
  text-model count.
- **Models** — browse a target's full model list, filtered by category
  (text, image, embedding, and so on), with classification source and context
  limits. `Refresh` bypasses the cache.
- **Playground** — pick a target and model, write a prompt (optional system
  prompt), toggle web search, and run a real generation. Results show text,
  timing, token usage, finish reason, and rendered citation links.
- **Verify** — run the same walk as `keycall verify` (optionally with
  generation) across every target and read the per-model attempt report.

Options: `--host` (default `127.0.0.1`), `--port` (default: pick a free one),
`--no-open` to skip the browser launch. Sources are the same TXT/JSON/TOML/
`env:VAR` formats `verify` accepts.

Security properties, in brief: a fresh auth token is generated per run,
required on every API request, printed once to your terminal, and never
written to disk. Keys are held in the server process only — the browser sees
target ids and display names, never credentials. All responses carry a
`default-src 'self'` CSP, so the page can make no external requests. The
server binds localhost by default; treat `--host` values beyond that as
deliberately exposing live credentials to your network.

## Tracing (optional)

If the host application configures [TraceAct](https://github.com/traceact/traceact),
KeyCall emits `keycall.list_models` and `keycall.text_generation` spans with
safe fields only: provider, model IDs, counts, status, durations, token
totals. Prompts, responses, and credentials are never captured, and KeyCall
pins input capture off with the `api_keys`/`ai_prompts` redaction presets on
its spans regardless of global settings.

```python
import traceact

traceact.configure(project="my-app", sinks=[traceact.JsonlSink("traces.jsonl")])
# KeyCall spans now flow to your sink. Without configure(), KeyCall emits nothing.
```

KeyCall never calls `configure()` itself and works identically with TraceAct
absent.

## Security model in one paragraph

The raw key enters at exactly one boundary (the client constructor), is
wrapped in a redacting type immediately, and is revealed at exactly one call
site (the transport layer's header builder). It cannot be pickled, copied,
printed, or read back off the client. Everything provider-originated is
sanitized before it reaches you; redirects are refused; response sizes are
capped; custom endpoints face HTTPS, private-address, and DNS-rebinding
guards. KeyCall stores nothing: where your keys live is your application's
decision.
