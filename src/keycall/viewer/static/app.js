// KeyCall viewer frontend. Vanilla ES module, no dependencies (the CSP
// blocks external anything).
//
// The access token arrives once in the page URL's ?token=. We move it into
// sessionStorage and strip it from the address bar: keeping it in the URL
// would leave the secret in browser history and in anything the user
// bookmarks or copies, and keeping it only in a page variable meant a
// plain reload lost it and the app died with "Not authorized". Session
// storage is scoped to this tab and this origin, and it clears when the
// tab closes, so a reload works and the token still stays out of history.

const TOKEN_KEY = "keycall.viewer.token";

function readToken() {
  const params = new URLSearchParams(location.search);
  const fromUrl = params.get("token");
  if (fromUrl) {
    try {
      sessionStorage.setItem(TOKEN_KEY, fromUrl);
    } catch {
      // Private modes can refuse storage; the token still works for this
      // page load, it just won't survive a reload.
    }
    history.replaceState(null, "", location.pathname);
    return fromUrl;
  }
  try {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

function forgetToken() {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Nothing stored, nothing to clear.
  }
}

const TOKEN = readToken();

async function api(path, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}), "X-KeyCall-Token": TOKEN } };
  if (opts.body && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
    opts.headers["Content-Type"] = "application/json";
  }
  try {
    const res = await fetch(path, opts);
    return await res.json();
  } catch (err) {
    return { error: { code: "request_failed", message: String(err) } };
  }
}

const el = (id) => document.getElementById(id);
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

function td(text, className) {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

// Category names as a person would say them. The API values
// (text_generation, image_generation) are for code, not for a reader.
const CATEGORY_LABELS = {
  text_generation: "Writes text",
  image_generation: "Makes images",
  embedding: "Embeddings",
  transcription: "Transcribes audio",
  speech_generation: "Speaks text aloud",
  video_generation: "Makes video",
  realtime: "Realtime voice",
  unknown: "Unrecognised",
};

function categoryLabel(value) {
  return CATEGORY_LABELS[value] || value;
}

function emptyState(node, title, hint) {
  // An empty panel is a question the user has to answer by guessing. Say
  // what happened and what to do about it, every time.
  clear(node);
  const box = document.createElement("div");
  box.className = "empty";
  const heading = document.createElement("strong");
  heading.textContent = title;
  box.appendChild(heading);
  const line = document.createElement("p");
  line.textContent = hint;
  box.appendChild(line);
  node.appendChild(box);
}

function working(button, label) {
  // Disabling alone leaves the user unsure the click registered.
  if (!button.dataset.idleLabel) button.dataset.idleLabel = button.textContent;
  button.disabled = true;
  button.textContent = label;
}

function done(button) {
  button.disabled = false;
  if (button.dataset.idleLabel) button.textContent = button.dataset.idleLabel;
}

function pill(text, kind) {
  const span = document.createElement("span");
  span.className = `pill ${kind}`;
  span.textContent = text;
  return span;
}

let TARGETS = [];

// --- sortable tables --------------------------------------------------------

function attachSort(table) {
  table.querySelectorAll("thead th").forEach((th, index) => {
    th.classList.add("sortable");
    th.addEventListener("click", () => {
      const dir = th.dataset.dir === "asc" ? "desc" : "asc";
      table.querySelectorAll("thead th").forEach((h) => {
        delete h.dataset.dir;
        h.classList.remove("sort-asc", "sort-desc");
      });
      th.dataset.dir = dir;
      th.classList.add(dir === "asc" ? "sort-asc" : "sort-desc");
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.rows);
      const numeric = rows.every((r) => {
        const v = r.cells[index]?.textContent.trim();
        return v === "—" || v === "" || !Number.isNaN(Number(v));
      });
      rows.sort((a, b) => {
        const av = a.cells[index]?.textContent.trim() ?? "";
        const bv = b.cells[index]?.textContent.trim() ?? "";
        let cmp;
        if (numeric) {
          const an = av === "—" || av === "" ? -Infinity : Number(av);
          const bn = bv === "—" || bv === "" ? -Infinity : Number(bv);
          cmp = an - bn;
        } else {
          cmp = av.localeCompare(bv);
        }
        return dir === "asc" ? cmp : -cmp;
      });
      rows.forEach((r) => tbody.appendChild(r));
    });
  });
}

// --- tabs -------------------------------------------------------------------

document.querySelectorAll("#tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    el(btn.dataset.tab).classList.add("active");
  });
});

// --- boot -------------------------------------------------------------------

function showFatal(message) {
  const box = el("fatal");
  box.textContent = message;
  box.classList.remove("hidden");
  document.querySelector("main").classList.add("hidden");
  el("tabs").classList.add("hidden");
  el("source-panel").classList.add("hidden");
}

function toggleEmptyState(isEmpty) {
  el("source-panel").classList.toggle("hidden", !isEmpty);
  document.querySelector("main").classList.toggle("hidden", isEmpty);
  el("tabs").classList.toggle("hidden", isEmpty);
}

async function refreshTargets() {
  const data = await api("/api/targets");
  if (data.error) {
    showFatal(`${data.error.code}: ${data.error.message}`);
    return;
  }
  TARGETS = data.targets || [];
  const version = el("health").textContent.split(" ")[1] || "";
  el("health").textContent = `keycall ${version} · ${TARGETS.length} target(s)`;
  renderDashboard();
  fillTargetSelects();
  toggleEmptyState(TARGETS.length === 0);
  if (TARGETS.length) {
    await loadModels();          // fills the cache…
    await loadPlaygroundModels(); // …which this then reuses instantly
  }
}

el("source-load").addEventListener("click", async () => {
  const path = el("source-path").value.trim();
  const status = el("source-status");
  if (!path) {
    status.textContent = "enter a file path first";
    return;
  }
  status.textContent = "loading…";
  const data = await api("/api/source", { method: "POST", body: { path } });
  if (data.error) {
    status.textContent = `${data.error.code}: ${data.error.message}`;
    return;
  }
  status.textContent = "";
  (data.warnings || []).forEach((w) => console.warn("keycall:", w));
  await refreshTargets();
});

async function boot() {
  const health = await api("/api/health");
  if (health.error) {
    // Either this tab never had a token, or the server restarted and
    // issued a new one. A stale token can only mislead the next reload,
    // so drop it and say plainly where the working link comes from.
    forgetToken();
    showFatal(
      "This page needs the link from your terminal. KeyCall printed a web " +
      "address ending in ?token=… when it started — open that exact link " +
      "and this page will work. If you have restarted KeyCall since, use " +
      "the newest link it printed, because the earlier one stops working."
    );
    return;
  }
  el("health").textContent = `keycall ${health.version} · ${health.targets} target(s)`;

  attachSort(el("dashboard-table"));
  attachSort(el("models-table"));
  el("dash-text-header").title =
    "Models this key can use to write text, which is what the Playground calls.";
  el("dash-embed-header").title =
    "Models this key can use to turn text into vectors, via embed(). Zero means "
    + "the provider has no embeddings API, not that the key is limited.";
  // The Verify tab opens with no results, which is a state, not a blank.
  emptyState(
    el("verify-empty"),
    "No check has been run yet",
    "Press \u201cRun verify\u201d above to test every key you have loaded. Results appear here, one card per key."
  );
  await refreshTargets();
}

// --- dashboard --------------------------------------------------------------

function renderDashboard() {
  const tbody = el("dashboard-table").querySelector("tbody");
  clear(tbody);
  TARGETS.forEach((t) => {
    const row = document.createElement("tr");
    row.className = "clickable";
    row.appendChild(td(t.name));
    row.appendChild(td(t.provider));
    const statusCell = document.createElement("td");
    statusCell.appendChild(pill("not checked", "pending"));
    row.appendChild(statusCell);
    row.appendChild(td("—", "num"));
    row.appendChild(td("—", "num"));
    row.addEventListener("click", () => checkTarget(t.id, row));
    tbody.appendChild(row);
  });
}

async function checkTarget(id, row) {
  const statusCell = row.children[2];
  clear(statusCell);
  statusCell.appendChild(pill("checking…", "pending"));
  // Count both operations KeyCall can perform with a key, not just one:
  // a key with no text models but working embeddings is still useful, and
  // reporting only text made it look dead.
  const data = await api(`/api/models?target=${id}`);
  clear(statusCell);
  if (data.error) {
    statusCell.appendChild(pill(data.error.code, "err"));
    row.children[3].textContent = data.error.message;
    row.children[4].textContent = "";
    return;
  }
  statusCell.appendChild(pill("key valid", "ok"));
  const count = (category) =>
    data.models.filter((m) => m.categories.includes(category)).length;
  row.children[3].textContent = String(count("text_generation"));
  row.children[4].textContent = String(count("embedding"));
}

// --- shared target selects --------------------------------------------------

function fillTargetSelects() {
  ["models-target", "pg-target"].forEach((selId) => {
    const sel = el(selId);
    clear(sel);
    TARGETS.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.id;
      opt.textContent = `${t.name} (${t.provider})`;
      sel.appendChild(opt);
    });
  });
}

// --- models browser ---------------------------------------------------------

async function loadModels(refresh = false) {
  try {
    await loadModelsInner(refresh);
  } catch (err) {
    // A throw here used to leave "Asking the provider…" on screen with no
    // way forward. Any failure has to land somewhere the user can see.
    el("models-status").textContent = "";
    emptyState(
      el("models-empty"),
      "Could not load models",
      `Something went wrong in the page: ${err}. Press Refresh to try again, ` +
      "or reopen the link KeyCall printed in your terminal."
    );
  }
}

async function loadModelsInner(refresh) {
  const id = el("models-target").value;
  const category = el("models-category").value;
  if (id === "") return;
  el("models-status").textContent = "Asking the provider…";
  clear(el("models-empty"));
  const q = new URLSearchParams({ target: id });
  if (category) q.set("category", category);
  if (refresh) q.set("refresh", "1");
  const data = await api(`/api/models?${q}`);
  const tbody = el("models-table").querySelector("tbody");
  clear(tbody);
  if (data.error) {
    el("models-status").textContent = "";
    emptyState(
      el("models-empty"),
      "Could not load models",
      `${data.error.message} (${data.error.code}). Check the key on the Dashboard tab, then press Refresh.`
    );
    return;
  }
  populateCategoryOptions(data);
  if (!data.models.length) {
    el("models-status").textContent = "";
    const filtered = Boolean(el("models-category").value);
    emptyState(
      el("models-empty"),
      "No models to show",
      filtered
        ? "This key has no models of that kind. Set the dropdown back to \u201cAll categories\u201d to see everything it can use."
        : "This key returned no models at all. That can be normal — some providers only list models you have created yourself. Try another key, or press Refresh."
    );
    el("models-table").classList.add("hidden");
    return;
  }
  el("models-table").classList.remove("hidden");
  el("models-status").textContent =
    `${data.models.length} model${data.models.length === 1 ? "" : "s"}` +
    `${data.from_cache ? ", from a saved copy" : ""} · model list ${data.catalog_version}`;
  // Source only earns its column when it varies (e.g. Gemini mixes
  // provider_metadata with keycall_rule); a constant column is noise.
  const sources = new Set(data.models.map((m) => m.classification_source));
  el("models-table").classList.toggle("hide-source", sources.size <= 1);
  // Most providers report no context window at all; a column of dashes
  // tells the reader nothing, so it only appears when someone filled it.
  const anyContext = data.models.some((m) => m.context_limit);
  el("models-table").classList.toggle("hide-context", !anyContext);
  el("models-context-header").title = anyContext
    ? "How much text the model can consider at once, in tokens, as reported by the provider."
    : "";
  data.models.forEach((m) => {
    const row = document.createElement("tr");
    row.appendChild(td(m.id));
    row.appendChild(td(m.categories.map(categoryLabel).join(", ")));
    row.appendChild(td(m.classification_source));
    row.appendChild(
      td(m.context_limit ? Number(m.context_limit).toLocaleString() : "—", "num")
    );
    tbody.appendChild(row);
  });
}

let categoryOptionsFilled = false;
function populateCategoryOptions() {
  if (categoryOptionsFilled) return;
  const cats = [
    "text_generation", "image_generation", "embedding", "transcription",
    "speech_generation", "video_generation", "realtime", "unknown",
  ];
  const sel = el("models-category");
  cats.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = categoryLabel(c);
    sel.appendChild(opt);
  });
  categoryOptionsFilled = true;
}

el("models-target").addEventListener("change", () => loadModels());
el("models-category").addEventListener("change", () => loadModels());
el("models-refresh").addEventListener("click", () => loadModels(true));

// --- playground -------------------------------------------------------------

async function loadPlaygroundModels() {
  const id = el("pg-target").value;
  if (id === "") return;
  const sel = el("pg-model");
  clear(sel);
  const opt = document.createElement("option");
  opt.textContent = "loading models…";
  sel.appendChild(opt);
  const data = await api(`/api/models?target=${id}&category=text_generation`);
  clear(sel);
  if (data.error) {
    const o = document.createElement("option");
    o.textContent = `error: ${data.error.code}`;
    sel.appendChild(o);
    return;
  }
  data.models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.id;
    sel.appendChild(o);
  });
}

el("pg-target").addEventListener("change", loadPlaygroundModels);

// --- images -----------------------------------------------------------------

// Holds the picked file's bytes as base64. The server decodes it back into
// the same ImageInput any caller would construct, so the Playground
// exercises the real path rather than a viewer-only shortcut.
let PG_IMAGE = null;

el("pg-image-on").addEventListener("change", () => {
  el("pg-image-panel").hidden = !el("pg-image-on").checked;
});

el("pg-image-clear").addEventListener("click", () => {
  PG_IMAGE = null;
  el("pg-image-file").value = "";
  el("pg-image-url").value = "";
  el("pg-image-status").textContent = "";
});

el("pg-image-file").addEventListener("change", () => {
  const file = el("pg-image-file").files[0];
  if (!file) {
    PG_IMAGE = null;
    el("pg-image-status").textContent = "";
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    // readAsDataURL gives "data:<type>;base64,<payload>"; the server wants
    // the payload, and the media type is re-derived from the bytes anyway.
    const encoded = String(reader.result).split(",")[1] || "";
    PG_IMAGE = { data_base64: encoded, media_type: file.type || undefined };
    const size =
      file.size < 1024 ? `${file.size} bytes` : `${Math.round(file.size / 1024)} KB`;
    el("pg-image-status").textContent = `${file.name} · ${size}`;
  };
  reader.onerror = () => {
    PG_IMAGE = null;
    el("pg-image-status").textContent = "could not read that file";
  };
  reader.readAsDataURL(file);
});

function imagesFromInput() {
  if (!el("pg-image-on").checked) return undefined;
  const url = el("pg-image-url").value.trim();
  if (url) return [{ url }];
  return PG_IMAGE ? [PG_IMAGE] : undefined;
}

// --- tools ------------------------------------------------------------------

const TOOL_EXAMPLE = [
  {
    name: "get_weather",
    description: "Get the current weather for a city",
    input_schema: {
      type: "object",
      properties: { city: { type: "string", description: "City name" } },
      required: ["city"],
    },
  },
];

// The conversation so far, replayed on every continuation. KeyCall never
// runs the tool loop, so the browser owns the turns; tool_call parts carry
// their `opaque` echo data back untouched or providers that require it
// reject the next turn.
let PG_HISTORY = [];

el("pg-tools-on").addEventListener("change", () => {
  el("pg-tools-panel").hidden = !el("pg-tools-on").checked;
});

el("pg-tools-example").addEventListener("click", () => {
  el("pg-tools").value = JSON.stringify(TOOL_EXAMPLE, null, 2);
  el("pg-tools-on").checked = true;
  el("pg-tools-panel").hidden = false;
});

function toolsFromInput() {
  if (!el("pg-tools-on").checked) return { tools: undefined, choice: undefined };
  const raw = el("pg-tools").value.trim();
  if (!raw) return { tools: undefined, choice: undefined };
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    // Report the JSON error here rather than sending it and getting back a
    // generic bad_request from the server.
    throw new Error(`tool definitions are not valid JSON: ${err.message}`);
  }
  return { tools: parsed, choice: el("pg-tool-choice").value || undefined };
}

function renderToolCalls(result) {
  const panel = el("pg-tool-calls");
  clear(panel);
  const calls = result.tool_calls || [];
  if (!calls.length) return;

  const card = document.createElement("div");
  card.className = "card";
  const head = document.createElement("strong");
  head.textContent = `${calls.length} tool call(s) — answer to continue`;
  card.appendChild(head);

  calls.forEach((call) => {
    const row = document.createElement("div");
    row.className = "attempt";
    const label = document.createElement("div");
    label.textContent = `${call.name}(${JSON.stringify(call.arguments)})`;
    row.appendChild(label);
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "tool result (text or JSON)";
    input.dataset.callId = call.id;
    row.appendChild(input);
    card.appendChild(row);
  });

  const send = document.createElement("button");
  send.textContent = "Send results";
  send.addEventListener("click", () => {
    const results = [...card.querySelectorAll("input[data-call-id]")].map((input) => {
      const call = calls.find((c) => c.id === input.dataset.callId);
      return {
        kind: "tool_result",
        tool_call_id: call.id,
        name: call.name,
        content: input.value || "(no result)",
      };
    });
    // The model's turn, then ours, both replayed on the next request.
    PG_HISTORY.push({ role: "assistant", parts: calls });
    PG_HISTORY.push({ role: "user", parts: results });
    clear(panel);
    runGeneration({ continuation: true });
  });
  card.appendChild(send);
  panel.appendChild(card);
}

el("pg-run").addEventListener("click", () => runGeneration({ continuation: false }));

async function runGeneration({ continuation }) {
  const btn = el("pg-run");
  const out = el("pg-result");
  const model = el("pg-model").value;
  const prompt = el("pg-prompt").value.trim();
  const images = imagesFromInput();
  if (!model || (!prompt && !continuation && !images)) {
    out.textContent = "pick a model and enter a prompt";
    return;
  }
  if (!continuation) {
    PG_HISTORY = [];
    clear(el("pg-tool-calls"));
  }
  let tooling;
  try {
    tooling = toolsFromInput();
  } catch (err) {
    clear(out);
    renderGeneration(out, { error: { code: "bad_request", message: err.message } });
    return;
  }
  working(btn, "Generating…");
  clear(out);
  out.textContent = "Waiting for the model…";
  const body = {
    target: Number(el("pg-target").value),
    model,
    prompt,
    system: el("pg-system").value.trim() || undefined,
    max_output_tokens: Number(el("pg-maxtok").value) || undefined,
    web_search: el("pg-search").checked,
    tools: tooling.tools,
    tool_choice: tooling.choice,
    images,
    history: PG_HISTORY.length ? PG_HISTORY : undefined,
  };
  try {
    await streamGeneration(out, body);
  } catch (err) {
    if (err && err.sawDelta) {
      // Tokens already arrived and were spent; a second call would be a
      // second charge. Report the interruption instead of retrying.
      renderGeneration(out, {
        error: { code: "stream_interrupted", message: "the stream ended unexpectedly — the partial output above may be incomplete" },
      });
    } else {
      // Streaming never started: the plain request costs the same one
      // generation the stream would have.
      const data = await api("/api/generate", { method: "POST", body });
      renderGeneration(out, data);
      renderToolCalls(data);
    }
  }
  done(btn);
}

async function streamGeneration(out, body) {
  let res;
  try {
    res = await fetch("/api/generate/stream", {
      method: "POST",
      headers: { "X-KeyCall-Token": TOKEN, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw { sawDelta: false };
  }
  if (!res.ok || !res.body) throw { sawDelta: false };

  clear(out);
  const card = document.createElement("div");
  card.className = "card";
  const text = document.createElement("div");
  text.className = "result-text";
  text.textContent = "";
  card.appendChild(text);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = "streaming…";
  card.appendChild(meta);
  out.appendChild(card);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDelta = false;
  let settled = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!frame.startsWith("data:")) continue;
        const event = JSON.parse(frame.slice(5));
        if (event.error) {
          renderGeneration(out, event);
          settled = true;
        } else if (event.kind === "text_delta") {
          sawDelta = true;
          // Append a text node per delta: textContent += would re-copy the
          // whole accumulated string on every token.
          text.appendChild(document.createTextNode(event.text));
        } else if (event.kind === "tool_call_started") {
          // Named before the arguments finish parsing, so show the call is
          // coming without implying it can be acted on yet.
          meta.textContent = `calling ${event.name}…`;
        } else if (event.kind === "result") {
          renderGeneration(out, event);
          renderToolCalls(event);
          settled = true;
        }
      }
    }
  } catch (err) {
    throw { sawDelta };
  }
  if (!settled) {
    // The connection closed without a result or error event.
    throw { sawDelta };
  }
}

function usageLabel(usage) {
  // Providers report different fields: a missing total is not a missing
  // count, so fall back to the parts rather than claiming nothing arrived.
  if (!usage) return "usage unreported";
  if (usage.total_tokens != null) return `${usage.total_tokens} tokens`;
  const parts = [];
  if (usage.input_tokens != null) parts.push(`${usage.input_tokens} in`);
  if (usage.output_tokens != null) parts.push(`${usage.output_tokens} out`);
  return parts.length ? `${parts.join(" / ")} tokens` : "usage unreported";
}

function renderGeneration(out, data) {
  clear(out);
  if (data.error) {
    const box = document.createElement("div");
    box.className = "err-box";
    box.textContent = `${data.error.code}: ${data.error.message}`;
    out.appendChild(box);
    return;
  }
  const card = document.createElement("div");
  card.className = "card";
  const text = document.createElement("div");
  text.className = "result-text";
  text.textContent = data.text || "(no text output)";
  card.appendChild(text);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent =
    `${data.model} · ${Math.round(data.round_trip_duration_ms)} ms · ${usageLabel(data.usage)}` +
    (data.finish_reason ? ` · ${data.finish_reason}` : "");
  card.appendChild(meta);

  if (data.citations && data.citations.length) {
    const cites = document.createElement("div");
    cites.className = "citations";
    const label = document.createElement("strong");
    label.textContent = `${data.citations.length} citation(s): `;
    cites.appendChild(label);
    data.citations.forEach((c, i) => {
      const a = document.createElement("a");
      a.href = c.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = c.title || c.url;
      cites.appendChild(a);
      if (i < data.citations.length - 1) cites.appendChild(document.createTextNode(" · "));
    });
    card.appendChild(cites);
  }
  out.appendChild(card);
}

// --- verify -----------------------------------------------------------------

el("verify-run").addEventListener("click", async () => {
  const btn = el("verify-run");
  const generate = el("verify-generate").checked;
  const results = el("verify-results");
  working(btn, "Checking…");
  clear(results);
  clear(el("verify-empty"));
  el("verify-status").textContent = generate
    ? "Checking each key and sending one short message…"
    : "Checking each key…";

  for (const t of TARGETS) {
    const data = await api("/api/verify", {
      method: "POST",
      body: { target: t.id, generate },
    });
    results.appendChild(renderVerify(t, data));
  }
  el("verify-status").textContent =
    `Finished. Checked ${TARGETS.length} key${TARGETS.length === 1 ? "" : "s"}.`;
  done(btn);
});

function renderVerify(target, data) {
  const card = document.createElement("div");
  card.className = "card";
  const head = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = `${target.name} (${target.provider}) `;
  head.appendChild(name);

  if (!data.listed_ok) {
    head.appendChild(pill(data.list_error_code || "list failed", "err"));
    card.appendChild(head);
    const msg = document.createElement("div");
    msg.className = "meta";
    msg.textContent = data.list_error_message || "";
    card.appendChild(msg);
    return card;
  }

  const outcomeKind =
    data.outcome === "generated" || (!data.generate_requested && data.listed_ok) ? "ok"
    : data.outcome === "rate_limited_unverified" ? "warn" : "err";
  head.appendChild(pill(data.outcome, outcomeKind));
  const count = document.createElement("span");
  count.className = "meta";
  count.textContent = ` ${data.text_model_count} text model(s)`;
  head.appendChild(count);
  card.appendChild(head);

  (data.attempts || []).forEach((a) => {
    const line = document.createElement("div");
    line.className = a.ok ? "attempt" : "attempt fail";
    if (a.ok) {
      // Only the total reaches the attempt record, so name the field the
      // CLI names: a provider can report per-direction counts and no total.
      const tokens = a.total_tokens != null ? a.total_tokens : "unreported";
      line.textContent = `✓ ${a.model_id} (pos ${a.position}) — ${Math.round(a.round_trip_duration_ms)} ms, total tokens: ${tokens}`;
    } else {
      line.textContent = `✗ ${a.model_id} (pos ${a.position}) — ${a.error_code}: ${a.error_message}`;
    }
    card.appendChild(line);
  });
  return card;
}

boot();
