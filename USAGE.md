# KeyCall Usage

Full reference for the Python API and the `keycall` CLI. For a quick overview, see [README.md](README.md); for version history, [CHANGELOG.md](CHANGELOG.md).

## Installation

Before you start: check `python3 --version` returns 3.10 or newer. macOS and Linux ship a system Python 3 already, though not always `pip` alongside it. If the check fails or `pip` is missing, install Python from [python.org/downloads](https://www.python.org/downloads/) (the Windows and macOS installers bundle `pip` automatically with the default options) or your platform's package manager: `apt install python3-pip` on Debian/Ubuntu, `dnf install python3-pip` on Fedora.

```bash
pip install keycall
```

Python 3.10+. Optional extras: `pip install "keycall[traceact]"` for tracing.

## Test KeyCall in under 60 seconds

No config file needed: simply point KeyCall at an environment variable for a key you already own.

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

Then open the same key in the local viewer and click around: a live dashboard, a browsable model list, and a Playground for chatting, showing a model a picture, recording a voice message in the page, holding a live voice conversation, transcribing your speech live, attaching a PDF, offering a tool, or generating an image.

```bash
keycall view --provider openai --source env:OPENAI_API_KEY
```

Swap `openai` for `anthropic`, `gemini`, `deepseek`, `perplexity`, `moonshot`, or `xai`; an `assemblyai` or `deepgram` key verifies too, with `--generate` left off, since a speech-to-text provider has no text models to generate with. To load several keys at once, put them in a file and use `--source ./keys.toml` instead; see [`keycall-test-keys.example.toml`](keycall-test-keys.example.toml) for the format. The rest of this document is the full reference.

## Clients

Construct one client per provider and credential. Identity is fixed at construction; switching provider, key, protocol, or base URL means constructing a new client.

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

Supported provider names: `openai`, `anthropic`, `gemini`, `deepseek`, `perplexity`, `moonshot`, `xai`, and the speech-to-text providers `assemblyai` and `deepgram`. Aliases: `claude`, `google`, `google-gemini`, `pplx`, `kimi`, `grok`, `x-ai`.

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

One verified example: Thinking Machines' Tinker exposes an OpenAI-compatible endpoint at `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`. Text generation, streaming, and tool calling all work through it (verified 2026-08-09); image input and schema enforcement don't. Its `GET /models` returns an empty list by design, because the models you address there are your own fine-tuned sampler checkpoints (`tinker://…/sampler_weights/000080`) rather than a shared catalog, so model discovery has nothing to return and you pass the checkpoint path as the model id. A key that lists successfully is still a validated key: `verify` reports `no_text_models` with `listed_ok` true rather than calling the credential bad.

Rules for `base_url`: absolute HTTPS, no query string, fragment, or userinfo. Plain HTTP is allowed only for localhost with `allow_insecure_localhost=True`. Literal private/internal IP addresses require `allow_private_network=True`. Hostnames are DNS-pinned per request: KeyCall resolves once, refuses to proceed if any resolved address is private, and connects to the validated address while TLS still verifies the original hostname.

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

Categories: `TEXT_GENERATION`, `IMAGE_GENERATION`, `EMBEDDING`, `TRANSCRIPTION`, `SPEECH_GENERATION`, `VIDEO_GENERATION`, `REALTIME`, `UNKNOWN`. Models KeyCall can't classify are `UNKNOWN` and appear only when you request that category explicitly, never in the default text picker.

`ModelDiscovery` fields: `models`, `provider`, `categories`, `fetched_at`, `from_cache`, `catalog_version`, `warnings`. Each `Model` carries `id`, `provider`, `categories`, `display_name`, `released_at`, `classification_source`, and `warnings`.

`released_at` is when the provider says the model appeared, and is `None` where the provider doesn't say. OpenAI and Moonshot report a unix timestamp, Anthropic an ISO date; Gemini and DeepSeek report nothing, and Perplexity has no list endpoint. It exists because `verify` orders its candidates by it, newest first, on the providers that publish it.

Results are cached in-process for 5 minutes, keyed by an HMAC fingerprint of the credential. Force a live call with `client.list_models(refresh=True)`, always do this when verifying a newly entered key.

A successful listing proves the credential works for discovery. It doesn't prove every listed model can be invoked; some providers advertise retired or quota-walled models with no lifecycle field to filter on.

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

`messages` accepts any sequence of `Message` objects, a plain list is fine. Roles are `"system"`, `"user"`, `"assistant"`. Dicts and bare strings are not accepted; there's one canonical representation.

The lower-level path accepts a typed request, useful when you build requests in one place and execute them in another:

```python
from keycall import TextGenerationRequest

request = TextGenerationRequest(model="...", messages=[...], max_output_tokens=64)
result = client.invoke(request)
```

Some models constrain sampling parameters, and KeyCall raises `MODEL_NOT_SUITABLE` before any network call rather than letting the provider 400. Two shapes:

- **No explicit value accepted**: OpenAI o-series and gpt-5, Anthropic Opus 4.7+, Opus 5+, and Sonnet 5+. Omit `temperature` and `top_p`.
- **One value accepted**: every Moonshot kimi model takes `temperature=1.0` and `top_p=0.95` and rejects anything else. Those values pass through; the error names the permitted one.

The evidence lives in the bundled catalog per provider, with the date each claim was last checked against the live API. A provider that merely announces a deprecation isn't gated: Gemini announced `temperature` and `top_p` as deprecated in July 2026 and still accepts both, so KeyCall passes them through.

## Streaming

`stream_text()` takes the same parameters as `generate_text()` and delivers the response incrementally. Use it as a context manager, iterate the typed events, then call `result()` for the same `InvocationResult` a non-streamed call returns:

```python
with client.stream_text(model="gpt-4o-mini", messages=messages) as stream:
    for event in stream:
        if event.kind == "text_delta":
            print(event.text, end="", flush=True)
    result = stream.result()

print(result.usage.total_tokens, result.finish_reason)
```

`AsyncKeyCall.stream_text()` is the awaitable twin: `async with` and `async for`.

Event types, discriminated by `kind`:

| Event | `kind` | Carries |
|---|---|---|
| `StreamStart` | `stream_start` | model id the provider confirmed |
| `TextDelta` | `text_delta` | a text increment; with `response_schema`, a fragment of the final JSON |
| `ReasoningDelta` | `reasoning_delta` | an increment of a visible reasoning trace, on providers that stream one (DeepSeek, Moonshot, xAI) — progress you can show while a reasoning model thinks, never part of `result.text` |
| `CitationFound` | `citation` | one web-search source, as it surfaces |
| `ToolCallStarted` | `tool_call_started` | id and name of a call beginning; arguments not known yet |
| `ToolCallArgumentsDelta` | `tool_call_arguments_delta` | a raw fragment of that call's argument JSON |
| `ToolCallComplete` | `tool_call_complete` | the assembled `ToolCall`, arguments parsed |
| `StreamFinish` | `stream_finish` | finish reason and usage |
| `UnknownStreamEvent` | `unknown` | bounded provider kind for content KeyCall doesn't recognize yet |

Behavior and guarantees:

- `web_search`, `response_schema`, and `tools` combine with streaming on every provider that supports them non-streamed; the same gates apply (Anthropic still refuses the web_search + response_schema combination).
- Streaming is never retried, before or after the first byte.
- The stream must end with the provider's own terminal signal. A connection that closes early raises `NETWORK_ERROR` from the iterator, and `result()` raises rather than returning a silent partial.
- Response-size caps apply to the total stream and to each individual event; the read timeout applies between chunks, so a stalled stream raises `TIMEOUT`.
- Leaving the `with` block closes the connection, including on early `break`. `result()` before the stream finishes raises rather than silently consuming the rest.
- Custom OpenAI-compatible targets stream with the `[DONE]` terminal convention; usage may be unreported there and surfaces as the standard missing-usage warning.

## Tool calling

Define tools, receive the model's call requests as typed parts, execute on your side, and send results back. KeyCall normalizes all four wire shapes; it never executes a tool and never runs the loop — that stays your code:

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

- `to_assistant_message()` replays the model's turn, including provider echo data some providers require back verbatim (`ToolCall.opaque`, e.g. Gemini's thought signature — never modify or interpret it).
- `ToolResult.content` may be a string or a JSON-serializable mapping; adapters convert to each provider's required form.
- `tool_choice` accepts `"auto"`, `"required"`, or `"none"`. Forcing one named tool isn't yet supported. Some provider/model pairs reject `"required"` (DeepSeek thinking models return 400); the provider's typed error is surfaced.
- `web_search` combines with tools on OpenAI, Anthropic, and Gemini (where KeyCall sets the required `toolConfig` flag automatically).
- Perplexity has no tool calling and raises `UNSUPPORTED_OPERATION` before any network call; the live suite carries a drift probe that fails if that ever changes. Custom OpenAI-compatible targets pass through with a result warning that support is unverified.
- Not combinable: Anthropic tools + `response_schema`, because schema enforcement is itself a forced tool call.

### Streaming tool calls

`stream_text()` takes `tools` and `tool_choice` too. Calls arrive as three events: `tool_call_started` when the model names a tool, `tool_call_arguments_delta` as its arguments stream in, and `tool_call_complete` once they parse. Act on the complete event only:

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

- `result.tool_calls` after a stream matches what the same request returns non-streamed, so the loop above and the `generate_text()` loop can share their dispatch and replay code.
- Argument fragments are provider bytes, not KeyCall's: they split mid-token and only the concatenation is valid JSON. Show them as progress, never parse them. A provider that sends arguments whole (Gemini) emits no fragments at all, so treat their absence as normal.
- Malformed argument JSON raises `INVALID_PROVIDER_RESPONSE` from the iterator rather than yielding a call with silently dropped arguments.

### apply_patch (OpenAI)

`apply_patch=True` enables OpenAI's file-editing convention: the model proposes create/update/delete operations instead of you defining a tool for it. Calls and replies are ordinary `ToolCall`/`ToolResult` parts with `name == "apply_patch"` — the same replay loop as any other tool, just with a fixed operation instead of arguments you defined:

```python
result = client.generate_text(model="...", messages=messages, apply_patch=True)

while result.tool_calls:
    replies = []
    for call in result.tool_calls:
        operation = call.arguments  # {"type": "create_file"|"update_file"|"delete_file",
                                     #  "path": "...", "diff": "..."}  # diff omitted for delete
        outcome = my_patch_executor(operation)  # your code applies it to disk
        replies.append(ToolResult(
            tool_call_id=call.id, name=call.name,
            content={"status": "completed" if outcome.ok else "failed", "output": outcome.message},
        ))
    messages += [result.to_assistant_message(),
                 Message(role="user", content=replies)]
    result = client.generate_text(model="...", messages=messages, apply_patch=True)
```

- `diff` is OpenAI's own V4A format (a bare `@@` marker, then `-`/`+` lines) — not unified diff. KeyCall passes it through verbatim; parsing and applying it is your executor's job.
- A plain string `content` on the reply defaults to `status: "completed"`; pass a mapping with an explicit `status` to report a failure. A failed status doesn't error the request — the model reads it and reports the failure in its own next reply.
- Reserved name: a caller-defined `Tool` also named `"apply_patch"` raises `UNSUPPORTED_OPERATION` while `apply_patch=True`, since that name is how KeyCall recognizes the tool's own parts on replay.
- OpenAI-only (live-verified 2026-08-22, `gpt-5.1`/`gpt-5.6`); other providers raise `UNSUPPORTED_OPERATION`. Per-model support is OpenAI's own typed `invalid_request_error`, surfaced rather than tracked in a model allowlist (`gpt-4o-mini` refuses it, for one).

### Code interpreter (OpenAI, Gemini, xAI, Anthropic)

`code_interpreter=True` enables the provider's hosted code-execution tool: the model writes and runs code server-side, and the run comes back as a `CodeExecutionOutput` part on `result.code_executions` — there's nothing for you to execute or reply to, unlike `apply_patch` and caller-defined tools:

```python
result = client.generate_text(
    model="...", messages=messages, code_interpreter=True,
)
for execution in result.code_executions:
    print(execution.language, execution.code, "->", execution.output)
print(result.text)  # the model's own written-out answer
```

- **`output` isn't always the answer.** OpenAI and xAI report the run's code but not its printed output at the call level (`output` comes back empty); the human-readable result only appears in `result.text`, on the model's following reply. Gemini and Anthropic do report output directly.
- **A generated file is only recovered on Gemini.** A code run that saves an image comes back as bytes inline only on Gemini, surfaced as an ordinary `ImageOutput` in `result.parts`. OpenAI, xAI, and Anthropic instead hand back an opaque file reference that needs a further authenticated download — KeyCall doesn't perform that download yet, so a run that only produces a file (no text answer) currently loses it on those three.
- **Anthropic maps this onto its own `bash_code_execution` server tool** and needs a beta header KeyCall sends automatically only when `code_interpreter=True`. Only that tool's calls are normalized; a chained `text_editor_code_execution` call Anthropic sometimes makes to author a file first surfaces as an `UnknownOutput` instead.
- **Non-streaming only on Anthropic.** `stream_text(..., code_interpreter=True)` still completes correctly on Anthropic with the right final text, but its code/output currently doesn't survive the stream — call `generate_text()` instead when you need Anthropic's code execution details. OpenAI, Gemini, and xAI stream it fully.
- Supported on OpenAI, Gemini, xAI, and Anthropic (all live-verified 2026-08-22); other providers raise `UNSUPPORTED_OPERATION`.

### Custom tools (OpenAI)

Pass `input_schema=None` on a `Tool` to declare a custom (freeform) tool instead of an ordinary JSON-Schema one: the model's call arrives as a plain string rather than parsed arguments, since there's no schema to parse against. The rest of the round trip is identical to an ordinary tool:

```python
write_poem = Tool(name="write_poem", description="Records a poem", input_schema=None)
result = client.generate_text(model="...", messages=messages, tools=[write_poem])

for call in result.tool_calls:
    text = call.arguments["input"]  # a plain string, not a JSON object
```

- The call's argument is always the single key `arguments["input"]`, a plain string — not the JSON dict an ordinary tool's arguments would parse into.
- Reply the same way as any other tool: `ToolResult(tool_call_id=call.id, name=call.name, content=your_output)`.
- OpenAI-only (live-verified 2026-08-22); other providers raise `UNSUPPORTED_OPERATION` for a `Tool` declared with `input_schema=None`.

### Tool search (OpenAI, Anthropic)

Pass `defer_loading=True` on a `Tool` to keep its definition out of the model's context until it searches for and finds it. This is a request-size optimization for a large tool library, not a behavior change — the discovered tool's call and reply are ordinary `ToolCall`/`ToolResult` parts, identical to a non-deferred tool's:

```python
weather = Tool(
    name="get_weather", description="Get the weather at a location",
    input_schema={"type": "object", "properties": {"location": {"type": "string"}},
                  "required": ["location"]},
    defer_loading=True,
)
result = client.generate_text(model="...", messages=messages, tools=[weather])
# result.tool_calls works the same way it would without defer_loading
```

- KeyCall sends the provider's tool-search tool automatically whenever any `Tool` in the request sets `defer_loading=True` — there's nothing else to enable.
- Pays off once a tool library is large enough that sending every definition on every request is wasteful; a handful of tools gets no benefit and standard tool calling is simpler.
- OpenAI and Anthropic only (live-verified 2026-08-22); other providers raise `UNSUPPORTED_OPERATION` for a `Tool` with `defer_loading=True`.

## Images, audio, and documents

Pass an `ImageInput`, `AudioInput`, or `FileInput` alongside your text in a user message. Bytes are read directly; no file is uploaded and KeyCall never fetches anything on your behalf:

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

Support splits by *form*, not only by provider, and the gate fires before any network call:

| Provider | Image bytes | Image URL |
|---|---|---|
| OpenAI | yes | yes |
| Anthropic | yes | yes |
| Gemini | yes | no |
| Perplexity | yes | yes |
| Moonshot | yes | no |
| DeepSeek | no | no (API is text only) |

- **Bytes are the portable form.** Every image-capable provider accepts them, so `ImageInput(data=...)` works everywhere images work.
- **A URL is only sent, never fetched.** Providers that refuse remote URLs raise `UNSUPPORTED_OPERATION` telling you to pass bytes. KeyCall won't download it for you: an adapter making its own request could be pointed at anything by caller-supplied data.
- **The media type is detected from the content** (PNG, JPEG, GIF, WebP). A `media_type` you supply is used only for formats KeyCall doesn't recognize, because Anthropic and Gemini both reject a mismatched type and the bytes are the better evidence. An image KeyCall can't identify raises rather than being sent with a guess.
- Media belongs in user messages.

`AudioInput` and `FileInput` work the same way, with narrower support:

| Provider | Image | Audio | Document (PDF) |
|---|---|---|---|
| OpenAI | yes | no | yes |
| Anthropic | yes | no | yes |
| Gemini | yes | **yes** | yes |
| Perplexity | yes | no | no |
| Moonshot | yes | no | no |
| DeepSeek | no | no | no |
| xAI | yes | no | no |

Audio is Gemini-only among the supported providers: OpenAI's Responses API takes no audio content part, and Anthropic, Moonshot, and Perplexity all reject one. Documents are sent as bytes everywhere that takes them, and `FileInput.filename` is passed through where the provider wants a name. Every refusal names the providers that do accept that kind, so a gate is a signpost rather than a dead end.

## Web search

Providers with a native server-side search tool can ground a generation in live web results:

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
| xAI | `web_search` tool, on xAI's Responses-shaped agentic route | direct source URLs |
| Moonshot | `$web_search` builtin function; KeyCall runs the echo round trip internally | none — Moonshot returns no citation structure |

On Moonshot the search runs server-side, but the wire protocol needs a second round: the model answers with a `$web_search` tool call, and the caller must echo it back before the answer arrives. KeyCall runs that loop itself (bounded, with usage summed across the rounds), so the call you write is the same one-liner as everywhere else — it just bills as two or more provider calls when the model searches.

DeepSeek and custom OpenAI-compatible targets have no native search tool: `web_search=True` raises `UNSUPPORTED_OPERATION` before any network call rather than silently ignoring the request.

### The model decides when to search

On every provider, `web_search=True` offers the tool; the model chooses whether to use it, per request. A question the model believes it can answer from memory often gets answered from memory, searched nowhere, and cited by nothing — with yesterday's knowledge presented confidently. Two phrasings of the same question:

```python
# The model may skip the search and answer from training data:
"Name one tech news headline from this week."

# Naming the need makes the search near-certain:
"Search the web for tech news from this week and name one headline."
```

Guidelines that hold across providers:

- Say "search the web" (or "look this up") when the answer must be current — the instruction, not the flag, is what the model weighs.
- Anchor the request in time ("as of today", "this week", "in August 2026"): recency the model can't have is the strongest search trigger.
- Check `result.citations` when the answer must be grounded: an empty tuple on a citation-reporting provider usually means no search happened (on Moonshot it is always empty — see below).
- Perplexity is the exception: Sonar always searches, flag or no flag.

## Reasoning effort

`reasoning_effort` tells a reasoning-capable model how hard to think, which is the main lever on both latency and reasoning-token spend:

```python
result = client.generate_text(
    model="grok-4.6",
    messages=[Message(role="user", content=[TextInput(text="Why is the sky blue?")])],
    reasoning_effort="low",
)
```

The value is passed in the provider's own vocabulary (commonly `"low"` / `"medium"` / `"high"`) and is never converted, so a value the provider itself rejects surfaces as that provider's own typed error. `"minimal"` is the one exception: it's live-verified on OpenAI's Responses API alone, so KeyCall refuses it with `UNSUPPORTED_OPERATION` before any network call on every other provider, rather than letting it reach a native control that doesn't define that level. Each supported mapping was verified live to bind: reasoning-token counts follow the requested level.

| Provider | Native control |
|---|---|
| OpenAI | `reasoning.effort` (Responses API) |
| Anthropic | `output_config.effort` |
| Gemini | `thinkingConfig.thinkingLevel` (KeyCall uppercases the value) |
| Perplexity | `reasoning_effort` |
| xAI | `reasoning.effort`, on the `/v1/responses` route |

On xAI, naming an effort switches the request to `/v1/responses` the same way `web_search=True` does: Grok's chat completions accepts a `reasoning_effort` field with HTTP 200 but measured reasoning-token counts don't follow it, while the responses route honors the level.

DeepSeek accepts the parameter and ignores it the same way (HTTP 200, unmoved token counts), and Moonshot's thinking model wasn't available to verify — so on DeepSeek, Moonshot, and custom targets, `reasoning_effort` raises `UNSUPPORTED_OPERATION` before any network call rather than shipping a knob that does nothing.

## Realtime sessions

`realtime()` opens a live WebSocket conversation with a voice model — one connection, many turns, audio and words streaming back as they are generated:

```python
with client.realtime(
    model="gpt-realtime",
    instructions="You are a concise assistant.",
) as session:
    session.send_text("Why is the sky blue?")
    for event in session.events(timeout=60):
        if event.kind == "audio_delta":
            speaker.play(event.data)          # raw 16-bit PCM
        elif event.kind == "transcript_delta":
            print(event.text, end="", flush=True)
        elif event.kind == "turn_complete":
            break
```

Turns go up three ways: `send_text(text)` is a whole typed turn, `send_audio(pcm)` streams caller microphone audio in chunks, and `end_audio_turn()` closes an audio turn and asks for the response (providers with server-side voice detection can also decide on their own). Events come back normalized:

| Event kind | Meaning |
|---|---|
| `session_started` | the provider accepted the session |
| `audio_delta` | a chunk of generated speech, decoded to raw PCM bytes |
| `transcript_delta` | the words being spoken (or the text answer, in a text-modality session) |
| `turn_complete` | the response finished; carries `usage` where the provider reports it |
| `interrupted` | the turn was cut off by a new one |
| `session_ended` | the connection closed; always the final event |

Provider notes, all verified live:

- **OpenAI** (`gpt-realtime`): the GA Realtime API; the only provider with a text output modality; usage reported per turn.
- **xAI** (`grok-voice-latest`): Grok Voice is voice-only — the words arrive as the transcript of the audio — and reports no usage.
- **Gemini** (`gemini-2.5-flash-native-audio-latest`): audio-only models; the API key rides a header on the WebSocket handshake, never the URL; usage includes thought tokens. Caller audio is 16 kHz 16-bit PCM (OpenAI and xAI take 24 kHz); generated audio is 24 kHz on all three.

Everything KeyCall doesn't model can be passed as `provider_config={...}`, merged verbatim into the provider's session-configuration message; using it reports a warning, since those keys won't port between providers. `AsyncKeyCall.realtime()` is the same surface with `async for` over `events()`. Providers without a realtime API (Anthropic, DeepSeek, Perplexity, Moonshot, custom targets) refuse with `UNSUPPORTED_OPERATION` before any connection.

## Streaming transcription

`transcribe_stream()` opens a live speech-to-text session with an STT provider — AssemblyAI or Deepgram, KeyCall's first non-LLM providers. Push raw 16-bit mono PCM in, read normalized transcript events out:

```python
client = KeyCall(provider="deepgram", api_key="...")   # or "assemblyai"

with client.transcribe_stream(model="nova-3", sample_rate=16000) as session:
    # feed audio from another thread (or interleave sends with reads)
    session.send_audio(pcm_chunk)          # raw 16-bit mono PCM, binary frames
    session.finish()                       # no more audio; finalize and close

    for event in session.events(timeout=30):
        if event.kind == "interim_transcript":
            display(event.text)            # provisional, will be superseded
        elif event.kind == "final_transcript":
            process(event.text, event.words)
        elif event.kind == "session_ended":
            bill_seconds = event.audio_duration_seconds
```

| Event kind | Meaning |
|---|---|
| `session_started` | the provider accepted the session (AssemblyAI; Deepgram sends no such frame — a successful connect is its accept signal) |
| `interim_transcript` | provisional text, superseded by later events; for live display only |
| `final_transcript` | finalized text with per-word timings; this text will not change |
| `session_ended` | the connection closed; always the final event, carrying the provider's billable-audio-seconds where its session summary arrived |
| `unknown` | a frame KeyCall doesn't recognize, bounded to its type name |

What a `final_transcript` carries, per the partial-support rule — a field only one provider reports is still normalized, and `None` means "this provider doesn't say":

- **`words`**: per-word `TranscriptWord(text, start_ms, end_ms, confidence)` on both providers. Timings are milliseconds from the session's start regardless of the provider's own unit (AssemblyAI counts ms, Deepgram counts seconds; KeyCall converts). Deepgram's words carry punctuation and casing (`punctuate=true` is always requested); AssemblyAI formats the transcript itself the same way.
- **`utterance_end`**: whether the speaker also finished the thought. Deepgram can finalize a stretch of text mid-utterance (`is_final` without `speech_final`) — more finals of the same utterance follow; AssemblyAI only finalizes whole turns, so it's always `True` there.
- **`confidence`**: overall on Deepgram; `None` on AssemblyAI, which scores per-word only (read `words` instead).
- **`channel`**: Deepgram's audio-channel index; `None` on AssemblyAI, which is single-channel. For dual-channel capture (a mic and system audio recorded separately), run one session per channel.

Both providers bill per second of audio, not per token — `session_ended.audio_duration_seconds` is the billable figure, from AssemblyAI's `Termination` frame or Deepgram's terminal `Metadata` frame. A session that drops before the summary arrives reports `None` there, with the close reason in `reason`.

- `model=None` takes the provider's default streaming model. On AssemblyAI that's its current flagship (`universal-3-5-pro`); on Deepgram it's a dated base model, so pass `model="nova-3"` explicitly there.
- `sample_rate` (default 16000) must match the PCM you send; both providers accept other rates.
- Sessions run long: AssemblyAI auto-closes after 3 hours. A dropped connection surfaces as `session_ended` with the close reason and no billing summary. Reconnection is yours: open a new session and resend audio from the point of the last `final_transcript` — everything after it was interim-only and is re-recognized from the resent audio. KeyCall doesn't buffer or replay audio itself.
- `list_models(categories={ModelCategory.TRANSCRIPTION})` lists each provider's streaming models (maintained by KeyCall — neither provider has a model-list API; the call still validates the credential against a live endpoint). `keycall verify`-style key checking works the same way as for LLM providers.
- `AsyncKeyCall.transcribe_stream()` is the same surface with `async with` / `async for`. LLM providers refuse `transcribe_stream` with `UNSUPPORTED_OPERATION` before any connection, and the STT providers refuse `generate_text` and every other LLM operation the same way.

On xAI the flag also changes which surface answers: plain generation uses chat completions, while a searched request goes to xAI's agentic route (`/v1/responses`, the same shape as OpenAI's Responses API). KeyCall switches route, body, and parser together — the call you write stays identical, and searched replies still stream, cite, and report usage the same way.

`result.citations` is a tuple of `Citation(url, title, cited_text)`, normalized across all four provider response shapes. Fields the provider didn't supply are `None`.

What the citation list does and doesn't guarantee:

- **Moonshot returns none.** Its `$web_search` builtin injects results into the model's context without any citation structure in the response (verified live 2026-08-14), so a searched Moonshot answer is grounded but unattributed: `result.citations` is always `()` there. Code that treats an empty citations tuple as "no search happened" must except Moonshot.
- **Campaign-tracking parameters are stripped.** OpenAI appends `?utm_source=openai` to every URL it cites, which attributes the click to OpenAI in the destination's analytics and would otherwise follow the link into whatever you render, log, or store; it offers no option to turn this off. Only the `utm_*` family goes, since those keys are ignored by the server receiving them and so can't change what a URL resolves to. Everything else is byte-identical to what the provider sent, including Gemini's `vertexaisearch.cloud.google.com` redirect, which is the citation by Google's design and which KeyCall doesn't pre-resolve.
- **One URL can legitimately appear more than once.** Providers cite per claim, not per source. Anthropic gives each citation its own `cited_text`, so three citations of one page are three different excerpts. KeyCall drops only citations that repeat an earlier one *exactly* — same URL, title, and excerpt — because those carry nothing the first didn't. Building a sources list means collapsing by URL yourself, and only you know whether to keep the longest excerpt, the first, or all of them.
- **The count can't be bounded in the request.** No provider exposes a "return at most N citations" parameter; Anthropic's `max_uses` limits how many searches the model runs, not how many sources come back. Slicing `result.citations` is the only option, and the tokens are already spent by the time you do. If capping matters, cap `max_output_tokens`, which does bound the answer the citations attach to.
- **Non-token charges appear in `Usage.provider_units`.** Perplexity bills a flat `request_cost` per call on top of tokens, so a budget counting only tokens misses it. Where a provider reports such figures, KeyCall passes them through as `(name, value)` pairs: `dict(result.usage.provider_units)["request_cost"]`. Providers that report nothing leave it `None`, which means "not reported", never zero. OpenAI, Anthropic, Gemini, DeepSeek, and Moonshot report no cost fields today, so their search or per-call pricing has to come from their own billing pages.
- **`Model.context_limit` is best-effort, and honest about it.** Three providers report the input ceiling under three different names, and KeyCall reads all three into one field: Gemini's `inputTokenLimit`, Anthropic's `max_input_tokens`, and Moonshot's `context_length`. OpenAI and DeepSeek report nothing and Perplexity has no list endpoint, so it is `None` there. `None` means "this provider doesn't say", never zero and never "lookup failed". It is deliberately never inferred from a bundled table or a sibling model: a caller budgets against this number, and an invented ceiling is worse than an absent one they can branch on.

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
| DeepSeek | falls back to `response_format={"type":"json_object"}` | no — valid JSON guaranteed, schema conformance isn't |
| Custom OpenAI-compatible targets | same fallback as DeepSeek, since capability is unverified | no |

On a non-enforcing provider, `result.warnings` explains that the schema wasn't enforced, validate client-side there rather than trusting the shape.

Three provider requirements to know before writing a schema:

- **OpenAI's strict mode requires `additionalProperties: false`** on every object level of the schema, or the request fails with a 400. Write it in from the start rather than discovering it from an error.
- **Gemini rejects any `additionalProperties` key** in the schema, at any nesting depth, with a 400 (live-verified 2026-08-08) — the direct opposite of OpenAI's requirement. One schema can't satisfy both providers; strip or add the key per provider before the call. KeyCall checks for the key before calling Gemini and raises `UNSUPPORTED_OPERATION` if it finds one, rather than letting the provider's raw 400 through — a schema generator that includes it by default (Pydantic's `model_json_schema()`, for one) will trip this on every nested object until it's stripped for Gemini specifically.
- **DeepSeek requires the word "json" somewhere in the prompt** for its fallback mode, or it 400s. KeyCall detects this and inserts a short system instruction automatically when needed — you'll see it noted in `result.warnings`, not applied silently.

`response_schema` and `web_search` can't be combined on Anthropic (forcing the structured-output tool prevents the model calling a different one in the same turn); combining them raises `UNSUPPORTED_OPERATION` before any network call. The same combination on Gemini is untested and not gated — KeyCall passes it through rather than guessing at behavior it hasn't verified.

## Embeddings

```python
result = client.embed(
    model="text-embedding-3-small",
    inputs=["first string", "second string"],
)

for text, part in zip(inputs, result.parts):
    part.values          # tuple[float, ...]
```

`result.parts` holds one `EmbeddingOutput` per input, **in the order the inputs were given**, so they zip together. A provider returning a different number of vectors raises `INVALID_PROVIDER_RESPONSE` rather than handing back a list that silently misaligns with your inputs.

| Provider | Embeddings | Example model | Dimensions |
|---|---|---|---|
| OpenAI | yes | `text-embedding-3-small` | 1536 |
| Gemini | yes | `gemini-embedding-001` | 3072 |
| Anthropic, DeepSeek, Perplexity, Moonshot | no | | |

Anthropic publishes no embeddings endpoint, and the other three return 404 or 403 for one (verified 2026-08-09). Calling `embed()` on them raises `UNSUPPORTED_OPERATION` before any network call, naming the providers that do support it. `AsyncKeyCall.embed()` is the awaitable twin.

Both providers batch: pass every string in one call rather than looping, which is one request instead of N. OpenAI reports token usage for the batch; Gemini's batch endpoint reports none, so `usage` is empty there rather than fabricated.

## Image generation

```python
result = client.generate_image(
    model="gpt-image-1",
    prompt="A flat illustration of a blue circle on a white background",
)

image = result.parts[0]
image.media_type          # "image/png", "image/jpeg", …
Path("out.png").write_bytes(base64.b64decode(image.base64_data))
```

| Provider | Image generation | Example model | Returns |
|---|---|---|---|
| OpenAI | yes | `gpt-image-1`, `gpt-image-2` | base64 PNG from `/images/generations` |
| Gemini | yes | `gemini-3.1-flash-image` | base64 JPEG, from the ordinary content endpoint |
| Anthropic, DeepSeek, Perplexity, Moonshot | no | | |

- **The request is a model and a prompt, nothing else.** OpenAI accepts a size and a count; Gemini's image models accept neither, and a parameter that silently does nothing on half the providers is worse than no parameter.
- **`media_type` reports what the provider actually produced**, so the bytes you write to disk carry the right extension. It is read from the response, never assumed.
- **A response with no image raises** rather than returning an empty result. Gemini answers refusals and clarifying questions as text instead of a picture, so the error repeats what the model said.
- Generation is slow: these calls take tens of seconds, and the default 60-second read timeout can be tight. Pass a larger `read_timeout` on the client for image work.
- Image *generation* is separate from image *input*. Sending a picture to a model is `ImageInput` on `generate_text()`, described above.

## Speech generation

```python
result = client.generate_speech(
    model="gpt-4o-mini-tts",
    text="Welcome to KeyCall.",
    voice="alloy",  # optional on this model — see the table below
)

clip = result.parts[0]
clip.media_type            # "audio/mpeg" on OpenAI
Path("out.mp3").write_bytes(base64.b64decode(clip.base64_data))
```

| Provider | Speech generation | Example model | Returns |
|---|---|---|---|
| OpenAI | yes | `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd` | raw audio file (`audio/mpeg` by default), read from `/audio/speech` |
| Gemini | yes | `gemini-2.5-flash-preview-tts` | raw PCM (`audio/L16;codec=pcm;rate=24000`), on the ordinary content endpoint |
| Anthropic, DeepSeek, Perplexity, Moonshot | no | | |

- **`voice` is optional, but not uniformly.** Gemini defaults one, and so does OpenAI's `gpt-4o-mini-tts`; OpenAI's `tts-1` and `tts-1-hd` require it and answer 400 without it (live-verified 2026-08-12 across all three). KeyCall never picks a voice for you — that would be a choice you never made — so a call to one of the older models needs `voice` passed explicitly, or the provider's own error names what's missing.
- **`media_type` is the provider's, not a guess, and it is not always a playable container.** OpenAI's response is a normal audio file. Gemini's is raw 16-bit PCM: to play or save it as a `.wav`, wrap it in a WAV header yourself (44 bytes, standard format — any audio library, or a dozen lines of `struct.pack`, does this) rather than treating the bytes as an MP3 or handing them to something that expects a container.
- **The response itself is not JSON on OpenAI** — the only operation in the package where that's true. Nothing about calling `generate_speech()` changes for this; it's mentioned here because if you ever inspect KeyCall's transport layer, this is the one route whose successful response is raw bytes rather than a parsed body.
- A text-only reply (Gemini asking a clarifying question, or refusing) raises rather than returning silence, and repeats what the model said — same posture as image generation's equivalent case.

## Video generation

Video renders as a job, not a round trip: every supporting provider answers a render request with a handle and expects polling, and render times observed live range from 10 seconds to over 11 minutes. KeyCall gives you the job directly, plus a one-call wrapper:

```python
result = client.generate_video(
    model="grok-imagine-video-1.5",
    prompt="A paper boat drifting across a puddle, morning light.",
    duration_seconds=6,        # optional; the provider defaults it otherwise
    aspect_ratio="16:9",       # optional
    timeout=300.0,             # required: your waiting budget, in seconds
)

clip = result.parts[0]
clip.media_type                # "video/mp4" on both providers
clip.url                       # the provider's own download URL, while it lives
Path("out.mp4").write_bytes(base64.b64decode(clip.base64_data))
```

Or drive the three phases yourself — the handle is plain data you can store and poll later, even from another process:

```python
job = client.start_video(model="veo-3.1-lite-generate-preview", prompt="...")
job = client.check_video(job)      # returns a new VideoJob; never mutates
if job.status == "succeeded":
    result = client.fetch_video(job)
```

| Provider | Video generation | Example model | Finished file |
|---|---|---|---|
| Gemini | yes | `veo-3.1-lite-generate-preview`, `veo-3.1-fast-generate-preview` | MP4 from Gemini's own host, kept about 2 days |
| xAI | yes | `grok-imagine-video-1.5` | MP4 from `vidgen.x.ai`, as a temporary unsigned URL |
| OpenAI, Anthropic, DeepSeek, Perplexity, Moonshot | no | | |

- **`timeout` on `generate_video()` has no default.** Only you know how long is too long for your caller. When the budget runs out KeyCall raises `VideoJobTimeout`, whose `.job` is the still-valid handle: the render keeps going provider-side, and `check_video(error.job)` picks up where the wait left off. You never lose a render you paid to start.
- **`job.status` is a closed set** — `running`, `succeeded`, `failed` — and `job.provider_status` carries the provider's own word verbatim (xAI's `expired` arrives as `failed` with `provider_status="expired"`). A failed render keeps the provider's message in `job.error_message`.
- **Job failures are outcomes, not HTTP errors.** Veo under load refuses renders with "high demand" messages inside a successful poll response; KeyCall reports these as failed jobs with the provider's wording, and never silently re-renders — a retry would be a second billable job.
- **Downloads are pinned.** The URL a job reports is only followed to hosts live-verified for that provider, the credential is only ever sent to the provider's own API host, and xAI's download URL works with no credential at all — treat that URL as a secret, since anyone holding it can fetch the file while it lives.

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
| `PERMISSION_DENIED` | Key valid but not entitled: permissions, or an unfunded account (HTTP 402) | no |
| `RATE_LIMITED` | Rate or quota limit; `retry_after` set when provided | yes |
| `PROVIDER_UNAVAILABLE` | 5xx or overload | yes |
| `NETWORK_ERROR` | Could not reach the provider | yes |
| `TIMEOUT` | No response within the timeout | yes |
| `INVALID_PROVIDER_RESPONSE` | Malformed body, redirect, or oversized response | no |
| `MODEL_NOT_AVAILABLE` | Model missing, retired, or rejected by name | no |
| `MODEL_NOT_SUITABLE` | Model can't serve this request: sampling parameters it pins or refuses, or a feature the provider has that this model lacks (web search on an older model) | no |
| `UNSUPPORTED_PROVIDER` | Unknown name or invalid custom target | no |
| `UNSUPPORTED_OPERATION` | Request shape not supported in this version | no |
| `CATALOG_UPDATE_REQUIRED` | Bundled catalog too old for this client | no |

Error messages are sanitized: no credentials, no raw request bodies, no unsanitized provider text.

Retry behavior: model listing gets a small bounded retry budget for transient failures, honoring `Retry-After`. Generation is never retried by KeyCall, since no supported provider documents generation idempotency, so retrying an ambiguous failure risks a second charge. `retryable` tells *you* whether a retry is reasonable at your layer.

## Falling back across discovered models

A model a provider's listing endpoint returns isn't a guarantee it will accept a request from your account: some providers advertise retired or entitlement-gated models with no lifecycle field to filter on ahead of time (see [Listing and filtering models](#listing-and-filtering-models) above). `MODEL_NOT_AVAILABLE` is how KeyCall reports that after the fact, and it's not retryable on the same model for that reason, so the useful response is to drop it and move to the next discovered candidate rather than retry it:

```python
from keycall import ErrorCode, KeyCallError

def generate_with_fallback(client, model_ids, prompt):
    for model_id in model_ids:
        try:
            return client.generate_text(model=model_id, prompt=prompt)
        except KeyCallError as error:
            if error.code is not ErrorCode.MODEL_NOT_AVAILABLE:
                raise
    raise RuntimeError("no candidate model was available")

discovery = client.list_models()
result = generate_with_fallback(client, [m.id for m in discovery.models], "Say hello.")
```

This is the same pattern the viewer's own Playground uses client-side: a model that comes back `MODEL_NOT_AVAILABLE` sinks to the bottom of its list for the rest of the session instead of being retried.

## The verify CLI

Live credential verification, one model-list call per target:

```bash
keycall verify --source ./keys.toml
```

Add one bounded generation per target:

```bash
keycall verify --source ./keys.toml --generate
```

With `--generate`, KeyCall walks the filtered text models in provider order and reports every attempt until one succeeds (default budget 8, adjustable with `--attempts`). Skipped models are printed with reasons, so retired models, modality mismatches, and per-model quota walls stay visible. Each attempt reports the model's position in both the filtered list and the provider's raw list, plus the classification evidence that made it a candidate; the result carries a digest of the raw model-list snapshot and the selection-rule version, so a failure is reconstructable against the provider surface that produced it.

### Live verification in CI

The same walk runs as a pytest suite with three modes:

| Mode | Where | Behavior |
|---|---|---|
| `off` | Every ordinary `pytest` run | Live tests deselected; no credentials touched |
| `warn` | CI `live-warn` job, manual dispatch only | Runs live smoke, reports failures, never fails the workflow |
| `strict` | Release workflow, before publish | Any unverified target blocks the release, including a missing credentials secret |

Select live tests explicitly with `pytest -m live`; they load credentials only at run time, from the target file named by `KEYCALL_LIVE_SOURCE`. In CI the file is written from the `KEYCALL_LIVE_TARGETS` repository secret (TOML target syntax, below). Rate-limit outcomes are reported as verification-environment failures, distinct from adapter or credential failures; in strict mode they still block the release because the release remains unverified. Fork pull requests receive no secrets and never run live jobs.

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

Fields: `provider` and `key` required; `protocol`, `base_url`, `name` optional. Repeating a provider creates independent targets.

### Behavior and exit codes

Keys never appear in output. Credential files are never modified or deleted. A broadly readable file or one inside a git repository produces a warning; `--strict-credentials` turns those warnings into errors. Exit codes: `0` all targets verified, `1` at least one failed, `2` usage or source error.

Use dedicated low-budget test keys, and `chmod 600` the file.

## The viewer

```bash
keycall view --source ./keys.toml
```

Starts a local, token-protected web app over the loaded targets and opens it in your browser.

The token is printed once and never written to disk. Opening the printed link is a handshake: the server sets an httpOnly, `SameSite=Strict` session cookie and redirects the token out of the address bar, so it never reaches page script or browser history, and a reload keeps working. Scripts and `curl` can authenticate with an `X-KeyCall-Token` header instead. Because a cookie rides along on requests other sites make, every POST must carry `Content-Type: application/json` and must not carry a foreign `Origin`. The cookie dies with the browser session, and restarting `keycall view` issues a new token that invalidates the old one.

Each tab has its own URL (`/models`, `/playground`, `/verify`, `/traces`; `/` is the Dashboard), so a reload or a bookmarked link opens straight onto it and back/forward walk the tabs you visited. Five tabs:

- **Dashboard** — every loaded target; click one for a live key check and its model count, or press **Test all keys** to run the same check on every row at once.
- **Models** — browse a target's full model list, filtered by category (text, image, embedding, and so on), with the classification source. `Refresh` bypasses the cache.
- **Playground** — pick a target and model, write a prompt (optional system prompt), toggle web search or tool calling, attach a picture, a recording (from a file, or the microphone button in the message box, which encodes to 16 kHz mono WAV in the browser), or a document, and send it to the provider. Results show text, timing, token usage, finish reason, and rendered citation links. The conversation carries across turns: each settled exchange is replayed with the next request, so follow-up questions keep their context, and switching the key or model mid-conversation hands the whole exchange to the new model. New chat clears the transcript and starts over. Attachments belong to the turn that sent them and are not replayed on later turns (each replay would be billed again); a short label stands in for the media. An attachment kind the selected key can't send is disabled with a line naming which of your keys to use instead, read from the same catalog the adapters gate on. Switch the task to **Make a picture** to call an image model instead, or **Make a video** for a video model (Gemini Veo, xAI Grok Imagine): a render runs far longer than a picture, from under a minute to over ten, and the reply bubble shows a running elapsed-time clock while it waits. A Reasoning effort select sends `reasoning_effort`, gated per key the same way as the other Extras. The reply budget field starts at a suggestion computed from what's selected: reasoning effort, web search, tool offering, and attachments all tend to spend more tokens than a bare reply, so the suggestion rises with them and drops back to a low floor when nothing token-intensive is on. Typing a value in by hand replaces the suggestion for the rest of the conversation; New chat resumes suggesting. A Timeout slider sets how many seconds the viewer waits on a provider before giving up (60 to 300, default 180); pictures routinely need more than a text reply. The task drives the Key select: picking a task narrows the list to keys whose own model list has at least one model for it, so a key that can't serve the task is never offered. Switch the task to **Have a voice conversation** for a live session on a realtime-capable key (OpenAI, xAI, Gemini): the viewer's server bridges a WebSocket in the browser to a `realtime()` session, so tapping the microphone once starts streaming caller audio and a second tap ends the session; the model's reply plays back as it arrives either way. Standing instructions above become the session's system prompt, set once when the session starts; ending the session or leaving voice mode closes the connection. Switch the task to **Transcribe speech** for live speech-to-text on an STT key (AssemblyAI, Deepgram): tap the microphone and talk, interim words appear as they are recognized and firm up into final bubbles (with the provider's confidence when it reports one), and tapping again finishes the session and shows the seconds of audio billed. A conversation is saved to the History pane as it grows — text tasks save each settled exchange, a transcription saves each finalized utterance — titled from the first prompt or the first recognized words, and clicking a saved conversation restores its transcript, key, and model.
- **Verify** — run the same walk as `keycall verify` (optionally with generation) across every target and read the per-model attempt report.
- **Traces** — every request this viewer run has made, newest first: which key and model, how long it took, and how it ended. When a button seems slow or silent, the answer is here — a reasoning model can think for most of a minute before its first visible token, and the trace shows that time. A search box filters rows as you type, and clicking a column header sorts by it, in either direction. Prompts and replies are never recorded, only timing and outcomes, and the log lives in the server process's memory for the run; Clear traces wipes it without a restart.

Options: `--host` (default `127.0.0.1`), `--port` (default: pick a free one), `--no-open` to skip the browser launch, `--reload` to restart the server whenever KeyCall's own source changes, keeping the same address and token so a hard reload in the browser picks up server-side edits (static files are re-read per request either way; this flag is for working on KeyCall itself). Sources are the same TXT/JSON/TOML/ `env:VAR` formats `verify` accepts.

Security properties, in brief: a fresh auth token is generated per run, required on every API request, printed once to your terminal, and never written to disk. Keys are held in the server process only — the browser sees target ids and display names, never credentials. All responses carry a `default-src 'self'` CSP, so the page can make no external requests. The server binds localhost by default; treat `--host` values beyond that as deliberately exposing live credentials to your network.

## Tracing (optional)

If the host application configures [TraceAct](https://github.com/traceact/traceact), KeyCall emits `keycall.list_models` and `keycall.text_generation` spans with safe fields only: provider, model IDs, counts, status, durations, token totals. Prompts, responses, and credentials are never captured, and KeyCall pins input capture off with the `api_keys`/`ai_prompts` redaction presets on its spans regardless of global settings.

```python
import traceact

traceact.configure(project="my-app", sinks=[traceact.JsonlSink("traces.jsonl")])
# KeyCall spans now flow to your sink. Without configure(), KeyCall emits nothing.
```

KeyCall never calls `configure()` itself and works identically with TraceAct absent.

## Security model

The raw key enters at exactly one boundary (the client constructor), is wrapped in a redacting type immediately, and is revealed at exactly one call site (the transport layer's header builder). It can't be pickled, copied, printed, or read back off the client. Everything provider-originated is sanitized before it reaches you; redirects are refused; response sizes are capped; custom endpoints face HTTPS, private-address, and DNS-rebinding guards. KeyCall stores nothing: where your keys live is your application's decision.

