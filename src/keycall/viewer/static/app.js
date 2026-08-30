// KeyCall viewer frontend. Vanilla ES module, no dependencies (the CSP
// blocks external anything).
//
// There is no token in this file, on purpose.
//
// The terminal prints a link carrying one; the server trades it for an
// httpOnly cookie and redirects the token out of the address bar before
// this script ever runs. httpOnly means page script cannot read the cookie,
// so an injection in a model's reply has nothing to steal, and the secret
// stays out of browser history. The browser attaches the cookie to
// same-origin requests by itself, which is why nothing below sets an auth
// header.
//
// The cookie is SameSite=Strict and the server additionally requires
// Content-Type: application/json on every POST, because a cookie — unlike
// the custom header this replaced — is sent on requests other sites make.

import { renderMarkdown } from "/static/markdown.js";

// The composer hints and placeholders name the actual modifier key the
// keydown handlers accept (event.metaKey on Mac, event.ctrlKey elsewhere),
// not a hardcoded one — a Mac user reading "Ctrl+Enter" would reasonably
// try Ctrl first and wonder why nothing happened.
const MOD_KEY = /Mac|iPhone|iPod|iPad/.test(navigator.platform || navigator.userAgent)
  ? "Cmd"
  : "Ctrl";

async function api(path, options = {}) {
  const opts = {
    ...options,
    headers: { ...(options.headers || {}) },
    // Explicit rather than relying on the default: this is the line that
    // carries the session cookie, and it should be obvious.
    credentials: "same-origin",
  };
  if (opts.body && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
  }
  // Always JSON, so the server's CSRF gate passes for us and fails for a
  // cross-site "simple request" that cannot set this header.
  if (opts.body) opts.headers["Content-Type"] = "application/json";
  try {
    const res = await fetch(path, opts);
    return await res.json();
  } catch (err) {
    return { error: { code: "request_failed", message: String(err) } };
  }
}

const el = (id) => document.getElementById(id);
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

// The page's one in-app confirmation dialog, for every irreversible action
// — never window.confirm(), which can't be themed and reads as untrusted
// next to the rest of the page. `confirmLabel` names the action itself
// ("Clear conversations"), never "OK": the choice should read as what it
// does, not a generic acknowledgement. Focus starts on Cancel, never on
// the destructive button, so a stray Enter can't confirm by accident.
function confirmDialog({ title, message, confirmLabel }) {
  return new Promise((resolve) => {
    const backdrop = el("confirm-dialog");
    el("confirm-title").textContent = title;
    el("confirm-message").textContent = message;
    const accept = el("confirm-accept");
    const cancel = el("confirm-cancel");
    accept.textContent = confirmLabel;
    backdrop.hidden = false;
    cancel.focus();

    const finish = (result) => {
      backdrop.hidden = true;
      accept.removeEventListener("click", onAccept);
      cancel.removeEventListener("click", onCancel);
      backdrop.removeEventListener("click", onBackdrop);
      document.removeEventListener("keydown", onKeydown);
      resolve(result);
    };
    const onAccept = () => finish(true);
    const onCancel = () => finish(false);
    const onBackdrop = (event) => { if (event.target === backdrop) finish(false); };
    const onKeydown = (event) => { if (event.key === "Escape") finish(false); };

    accept.addEventListener("click", onAccept);
    cancel.addEventListener("click", onCancel);
    backdrop.addEventListener("click", onBackdrop);
    document.addEventListener("keydown", onKeydown);
  });
}

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
// {image: [...providers], audio: [...], file: [...]}, from the catalog.
let PROVIDERS_ACCEPTING = {};
let PROVIDER_CAPABILITIES = {};

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

// Each tab has its own URL (/models, /traces, …) so a reload or a pasted
// link opens straight onto it, and back/forward walk the tabs visited.
// The server hands out the same page shell for every one of these paths;
// "/" is the Dashboard.
const TAB_PATHS = {
  dashboard: "/",
  models: "/models",
  playground: "/playground",
  verify: "/verify",
  traces: "/traces",
};

function tabForPath(pathname) {
  const entry = Object.entries(TAB_PATHS).find(([, path]) => path === pathname);
  return entry ? entry[0] : "dashboard";
}

// The one place a tab comes on screen, however it was asked for (click,
// page load, back/forward), so per-tab side effects can't be skipped by
// arriving a different way.
function activateTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === name));
  // The Playground can only be measured once it is on screen; a hidden
  // tab has no position to measure from.
  if (name === "playground") sizePlayground();
  // Traces polls only while its tab is showing; anywhere else the timer
  // would refresh a table nobody can see.
  if (name === "traces") {
    loadTraces();
    startTracesTimer();
  } else {
    stopTracesTimer();
  }
}

document.querySelectorAll("#tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    activateTab(btn.dataset.tab);
    history.pushState(null, "", TAB_PATHS[btn.dataset.tab] || "/");
  });
});

window.addEventListener("popstate", () => activateTab(tabForPath(location.pathname)));

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
  PROVIDERS_ACCEPTING = data.providers_accepting || {};
  PROVIDER_CAPABILITIES = data.provider_capabilities || {};
  // Only overwrite the control when the server names a value: an older
  // server process without the field must not blank or reset it.
  if (Number.isInteger(data.read_timeout)) {
    el("pg-timeout").value = data.read_timeout;
    paintTimeoutLabel();
  }
  fillProviderOptions(data.providers || []);
  const version = el("health").textContent.split(" ")[1] || "";
  el("health").textContent = `keycall ${version} · ${TARGETS.length} target(s)`;
  renderDashboard();
  await fillTargetSelects();
  toggleEmptyState(TARGETS.length === 0);
  if (TARGETS.length) {
    await loadModels();          // fills the cache…
    await loadPlaygroundModels(); // …which this then reuses instantly
  }
}

el("source-toggle").addEventListener("click", (event) => {
  event.preventDefault();
  const panel = el("source-file");
  panel.hidden = !panel.hidden;
  el("source-toggle").textContent = panel.hidden
    ? "Load a key file instead"
    : "Hide the key file option";
});

// The same form appears twice: on the empty state, and on the dashboard so
// a one-off key can be tested without editing a file first, even when a key
// file is already loaded. Nothing is saved either way — the key lives in
// the running process and goes when it does. One implementation, addressed
// by id prefix.
function wireKeyForm(prefix) {
  const field = el(`${prefix}-value`);
  const status = el(`${prefix}-status`);
  const button = el(`${prefix}-add`);

  const submit = async () => {
    const key = field.value.trim();
    if (!key) {
      status.textContent = "paste a key first";
      return;
    }
    working(button, "Adding…");
    status.textContent = "";
    const data = await api("/api/key", {
      method: "POST",
      body: { provider: el(`${prefix}-provider`).value, key },
    });
    done(button);
    if (data.error) {
      status.textContent = `${data.error.code}: ${data.error.message}`;
      return;
    }
    // Clear the field the moment the server has it: the key is in the local
    // process now, and leaving it on screen is the one copy anyone can read.
    field.value = "";
    status.textContent = "";
    await refreshTargets();
  };

  button.addEventListener("click", submit);
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submit();
    }
  });
}

wireKeyForm("key");
wireKeyForm("dash-key");

el("dash-key-toggle").addEventListener("click", () => {
  const form = el("dash-key-form");
  form.hidden = !form.hidden;
  el("dash-key-toggle").textContent = form.hidden ? "Test another key" : "Cancel";
  if (!form.hidden) el("dash-key-value").focus();
});

// Used only when the server reports no provider list at all, which happens
// when the page is newer than the process serving it: static files are
// re-read per request, but Python modules load once, so an upgraded
// KeyCall that hasn't been restarted serves this script against an older
// API. Without a fallback the dropdown renders empty and the form cannot
// be used, with nothing on screen to explain why.
const FALLBACK_PROVIDERS = [
  "openai", "anthropic", "gemini", "deepseek", "perplexity", "moonshot", "xai",
];

/** Fill both provider dropdowns from the catalog the server reports. */
function fillProviderOptions(providers) {
  // An empty list means the server didn't report one, not that there are no
  // providers, so fall back rather than emptying a dropdown the form needs.
  const names = providers && providers.length ? providers : FALLBACK_PROVIDERS;
  // Display names for the ones whose catalog id isn't how people say it.
  const labels = {
    openai: "OpenAI",
    anthropic: "Anthropic",
    gemini: "Google Gemini",
    deepseek: "DeepSeek",
    perplexity: "Perplexity",
    moonshot: "Moonshot / Kimi",
    xai: "xAI / Grok",
  };
  ["key-provider", "dash-key-provider"].forEach((id) => {
    const sel = el(id);
    const previous = sel.value;
    clear(sel);
    names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      // An unlabelled provider still appears, under its catalog id, rather
      // than being dropped because this map wasn't updated.
      option.textContent = labels[name] || name;
      sel.appendChild(option);
    });
    if (previous) sel.value = previous;
  });
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

el("pg-composer-hint").textContent = `Enter starts a new line. ${MOD_KEY}+Enter to send.`;
el("pg-voice-composer-hint").textContent = `Enter starts a new line. ${MOD_KEY}+Enter to send.`;
el("pg-prompt").placeholder = `Ask anything. Press Send, or ${MOD_KEY}+Enter.`;

async function boot() {
  const health = await api("/api/health");
  if (health.error) {
    // Either this browser never adopted a token, or the server restarted
    // and issued a new one, leaving a cookie that no longer matches. The
    // cookie is httpOnly so this script cannot clear it; the next opened
    // link overwrites it, which is what the message asks for.
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
  el("dash-models-header").title =
    "Every model this key can reach, of every kind. The Models tab breaks them down.";
  transcriptEmpty();
  sizePlayground();
  loadConversationList();
  // The Verify tab opens with no results, which is a state, not a blank.
  emptyState(
    el("verify-empty"),
    "No check has been run yet",
    "Press \u201cRun verify\u201d above to test every key you have loaded. Results appear here, one card per key."
  );
  await refreshTargets();
  // The URL decides the opening tab, so /traces reloads onto Traces
  // instead of dropping back to the Dashboard.
  activateTab(tabForPath(location.pathname));
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
    row.addEventListener("click", () => checkTarget(t.id, row));
    tbody.appendChild(row);
  });
  // Disabled with a reason when there is nothing to test, per the "disable,
  // don't just render, a control with nothing to act on" rule.
  const checkAll = el("dash-check-all");
  checkAll.disabled = !TARGETS.length;
  checkAll.title = TARGETS.length
    ? "Check every key above in one go"
    : "Load a key first — there is nothing to test yet";
}

async function checkTarget(id, row) {
  const statusCell = row.children[2];
  clear(statusCell);
  statusCell.appendChild(pill("checking…", "pending"));
  // Every category, not just text: the count is what the key can reach.
  const data = await api(`/api/models?target=${id}`);
  clear(statusCell);
  if (data.error) {
    statusCell.appendChild(pill(data.error.code, "err"));
    row.children[3].textContent = data.error.message;
    return;
  }
  statusCell.appendChild(pill("key valid", "ok"));
  row.children[3].textContent = String(data.models.length);
}

// One click runs the same check every row's own click runs, all in flight
// together rather than one after another; each row's status pill reports
// its own outcome as it arrives. allSettled so one provider erroring in
// transit can't strand the button in its disabled state.
async function checkAllTargets() {
  const button = el("dash-check-all");
  const rows = [...el("dashboard-table").querySelectorAll("tbody tr")];
  if (!rows.length) return;
  button.disabled = true;
  try {
    await Promise.allSettled(TARGETS.map((t, i) => checkTarget(t.id, rows[i])));
  } finally {
    button.disabled = false;
  }
}

el("dash-check-all").addEventListener("click", checkAllTargets);

// --- shared target selects --------------------------------------------------

async function fillTargetSelects() {
  const sel = el("models-target");
  clear(sel);
  TARGETS.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = `${t.name} (${t.provider})`;
    sel.appendChild(opt);
  });
  await renderPlaygroundTargets();
  applyKeyGates();
}

// The model category the current task needs, or null for text (every
// provider on file serves text models). The name doubles as the provider
// capability flag, which the two share deliberately.
function modeCategory(mode) {
  return mode === "image" ? "image_generation"
    : mode === "video" ? "video_generation"
    : mode === "voice" ? "realtime"
    : mode === "transcribe" ? "transcription"
    : null;
}

// "targetId:category" -> whether that key's own model list has at least
// one model of the category. Session-scoped, same as the refused-model
// memory: what a key can reach changes with the account, not the page.
const PG_KEY_HAS_MODELS = new Map();

async function keyHasModels(id, category) {
  const cacheKey = `${id}:${category}`;
  if (PG_KEY_HAS_MODELS.has(cacheKey)) return PG_KEY_HAS_MODELS.get(cacheKey);
  const data = await api(`/api/models?target=${id}&category=${category}`);
  // An errored listing (bad credential, provider down) keeps the key
  // visible: silently dropping it would read as the key vanishing, and
  // the model picker names the error somewhere it can be acted on.
  const has = data.error ? true : data.models.length > 0;
  PG_KEY_HAS_MODELS.set(cacheKey, has);
  return has;
}

// Guards against a task switch landing while a slower check for the
// previous task is still in flight.
let PG_TARGET_RENDER = 0;

// Rebuilds the Key select for the current task: the task decides which
// models are needed, and only keys that can reach at least one such model
// are offered at all. Provider capability rules out whole providers
// without a network call; the survivors are then checked against their
// key's own model list, because a provider having a realtime API does not
// mean this key lists any realtime models. Keeps the current selection if
// it still qualifies; otherwise falls back to the first eligible key, or
// a disabled placeholder if none qualify.
async function renderPlaygroundTargets() {
  const sel = el("pg-target");
  const previous = sel.value;
  const category = modeCategory(currentMode());
  const token = ++PG_TARGET_RENDER;
  let eligible = TARGETS.filter((t) => {
    if (!category) return true;
    const caps = PROVIDER_CAPABILITIES[t.provider];
    return !caps || Boolean(caps[category]);
  });
  if (category && eligible.length) {
    sel.disabled = true;
    clear(sel);
    const busy = document.createElement("option");
    busy.value = "";
    busy.textContent = "checking your keys…";
    sel.appendChild(busy);
    const checks = await Promise.all(
      eligible.map((t) => keyHasModels(t.id, category))
    );
    if (token !== PG_TARGET_RENDER) return;
    eligible = eligible.filter((_, i) => checks[i]);
  }
  clear(sel);
  if (!eligible.length) {
    const none = document.createElement("option");
    none.value = "";
    none.textContent = TARGETS.length ? "no key has models for this" : "add a key first";
    sel.appendChild(none);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  eligible.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = `${t.name} (${t.provider})`;
    sel.appendChild(opt);
  });
  if (eligible.some((t) => String(t.id) === previous)) sel.value = previous;
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
  // Three providers report a context window and three don't, so the column
  // appears for the ones that fill it and stays out of the way otherwise.
  const anyContext = data.models.some((m) => m.context_limit);
  el("models-table").classList.toggle("hide-context", !anyContext);
  el("models-context-header").title = anyContext
    ? "How much text the model can take in at once, in tokens, as reported by the provider."
    : "";
  data.models.forEach((m) => {
    const row = document.createElement("tr");
    const idCell = td(m.id);
    // Badge only where the server sent an alias fact: the id matches the
    // provider's recorded rolling-alias convention. No fact, no badge —
    // the page never infers alias-ness from the id's shape itself.
    if (m.alias) {
      const badge = document.createElement("span");
      badge.className = "alias-badge";
      badge.textContent = "alias";
      // data-tip + a CSS :hover tooltip, not the title attribute: the
      // browser's own ~1s title delay can't be tuned, and hover popups
      // must appear with zero added delay. aria-label carries the same
      // text for screen readers.
      const tip =
        (m.alias.maintained === false
          ? "A rolling alias the provider was seen retiring. "
          : m.alias.maintained === true
            ? "A rolling alias the provider keeps aimed at a live model. "
            : "A rolling alias. ") + m.alias.note;
      badge.dataset.tip = tip;
      badge.setAttribute("aria-label", tip);
      idCell.appendChild(badge);
    }
    row.appendChild(idCell);
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
  if (id === "") {
    // No eligible key for this task: leaving the previous task's models
    // behind a dead Key select would offer choices nothing can run.
    const sel = el("pg-model");
    clear(sel);
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "—";
    sel.appendChild(none);
    sel.disabled = true;
    updateSendEnabled();
    return;
  }
  const category =
    currentMode() === "image" ? "image_generation"
    : currentMode() === "video" ? "video_generation"
    : currentMode() === "voice" ? "realtime"
    : currentMode() === "transcribe" ? "transcription"
    : "text_generation";
  const sel = el("pg-model");
  clear(sel);
  const opt = document.createElement("option");
  opt.textContent = "loading models…";
  sel.appendChild(opt);
  const data = await api(`/api/models?target=${id}&category=${category}`);
  clear(sel);
  if (data.error) {
    const o = document.createElement("option");
    o.textContent = `error: ${data.error.code}`;
    sel.appendChild(o);
    updateSendEnabled();
    return;
  }
  if (!data.models.length) {
    const none = document.createElement("option");
    // An explicit empty value, not left to default to the option's own
    // text: an unset <option> takes its value from its textContent, which
    // would otherwise let this placeholder's sentence itself pass every
    // `!model` guard as if it were a valid model id, and reach a provider.
    none.value = "";
    none.textContent =
      currentMode() === "image" ? "this key has no picture models"
      : currentMode() === "video" ? "this key has no video models"
      : currentMode() === "voice" ? "this key has no voice models"
      : currentMode() === "transcribe" ? "this key has no transcription models"
      : "this key has no text models";
    sel.appendChild(none);
    sel.disabled = true;
    updateSendEnabled();
    return;
  }
  sel.disabled = false;
  // A model the provider has already refused for this key sinks to the
  // bottom and says so, rather than sitting at the top waiting to fail
  // again. This is learned from what the provider actually answered, not
  // from a bundled list of retired models: retirement is per account and
  // moves weekly, so a shipped list would be wrong for somebody the day it
  // shipped.
  const live = data.models.filter((m) => !isRefused(id, m.id));
  const refused = data.models.filter((m) => isRefused(id, m.id));
  [...live, ...refused].forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id;
    const no = isRefused(id, m.id);
    o.textContent = no ? `${m.id} — refused earlier` : m.id;
    o.disabled = no;
    sel.appendChild(o);
  });
  if (category === "video_generation") {
    const cheap = cheapestModelId(live.map((m) => m.id));
    if (cheap) sel.value = cheap;
  }
  updateSendEnabled();
}

// No provider's model list carries a price or tier field, so a
// cheaper/lighter model can only be recognized by name. These are the
// tier words providers already use today, tried in priority order; a
// provider with none of them keeps the picker's default (first) choice.
// Mirrors tests/test_live.py's _CHEAP_TIER_HINTS so both pick the same
// model for the same key.
const CHEAP_TIER_HINTS = ["lite", "nano", "mini", "fast", "flash"];

function cheapestModelId(ids) {
  for (const hint of CHEAP_TIER_HINTS) {
    const match = ids.find((id) => id.toLowerCase().includes(hint));
    if (match) return match;
  }
  return null;
}

// "<target id>:<model id>" for every model this session has seen a provider
// turn down. Session-scoped on purpose: a quota that frees up or an account
// that gains access should not stay marked past a reload.
const PG_REFUSED = new Set();

// Target and model of the request in flight, so a failure is attributed to
// what was actually asked.
let PG_LAST_REQUEST = null;

function isRefused(targetId, modelId) {
  return PG_REFUSED.has(`${targetId}:${modelId}`);
}

// Errors that condemn the model rather than the key, the request, or the
// moment. A rate limit or a network blip says nothing about the model, so
// neither marks it.
const MODEL_SCOPED_ERRORS = new Set([
  "model_not_available",
  "model_not_suitable",
]);

function noteModelOutcome(targetId, modelId, code) {
  if (!MODEL_SCOPED_ERRORS.has(code)) return;
  PG_REFUSED.add(`${targetId}:${modelId}`);
  // Re-render so the refused model drops out of reach and the picker lands
  // on one that has not been turned down.
  loadPlaygroundModels();
}

el("pg-target").addEventListener("change", () => {
  loadPlaygroundModels();
  // What a key can do changes with the key, so re-gate before the user
  // reaches for a control the new one cannot honour.
  applyKeyGates();
  // A key swap within video mode can cross the Gemini/xAI duration-range
  // boundary; clamp onto the new range without resetting to its default.
  if (currentMode() === "video") syncVideoDuration(false);
});

// Bound the two Playground columns to what is actually left on screen, so
// each scrolls its own content instead of growing the page. Measured rather
// than assumed: the header and tab strip change height with the font and
// with a wrapped line, and a hardcoded offset leaves either a dead gap or a
// second scrollbar.
function sizePlayground() {
  const layout = document.querySelector(".pg-layout");
  if (!layout || !window.innerHeight) return;
  // Stacked single-column below this width, where two scroll regions would
  // compete for the same thumb; let the page scroll normally there.
  if (window.innerWidth <= 900) {
    layout.style.height = "";
    return;
  }
  const top = layout.getBoundingClientRect().top;
  const room = window.innerHeight - top - 16;
  layout.style.height = `${Math.max(360, room)}px`;
}

window.addEventListener("resize", sizePlayground);

// --- task mode --------------------------------------------------------------

// Image generation is its own operation, not a switch on text generation:
// different models, a different request, and a picture rather than words
// coming back. The mode drives which models are offered and which controls
// make sense.
function currentMode() {
  return el("pg-mode").value;
}

async function applyMode() {
  const image = currentMode() === "image";
  const video = currentMode() === "video";
  const voice = currentMode() === "voice";
  const transcribe = currentMode() === "transcribe";
  el("pg-extras").hidden = image || video || voice || transcribe;
  el("pg-maxtok-row").hidden = image || video || voice || transcribe;
  // Neither generate_image() nor generate_video() sends reasoning_effort
  // at all (their requests are model + prompt, nothing else), so the
  // control would silently do nothing if left up rather than refusing.
  el("pg-reasoning-row").hidden = image || video || voice || transcribe;
  // Transcription has no instructions either: the session takes audio in
  // and gives words back, with no prompt anywhere in it.
  el("pg-system-row").hidden = image || video || transcribe;
  // The cache marker only reaches generate_text/stream_text; voice runs
  // over its own realtime connection, a different protocol the marker
  // never touches, so the toggle would silently do nothing there.
  el("pg-cache-row").hidden = image || video || voice || transcribe;
  el("pg-image-mode-note").hidden = !image;
  el("pg-video-mode-note").hidden = !video;
  el("pg-voice-mode-note").hidden = !voice;
  el("pg-transcribe-mode-note").hidden = !transcribe;
  el("pg-video-duration-row").hidden = !video;
  if (!video) el("pg-video-duration-warning").hidden = true;
  // An image or video model takes a description and nothing else, so a
  // microphone in the composer would only offer something that cannot be
  // sent. Voice and transcribe sessions each have their own microphone
  // control, in their own panel.
  el("pg-mic").hidden = image || video || voice || transcribe;
  el("pg-composer").hidden = voice || transcribe;
  el("pg-composer-hint").hidden = voice || transcribe;
  el("pg-voice-panel").hidden = !voice;
  el("pg-transcribe-panel").hidden = !transcribe;
  // Leaving a session mode ends any session in progress rather than
  // leaving a WebSocket open behind a panel nothing points at any more.
  if (!voice) endVoiceSession();
  if (!transcribe) endTranscribeSession();
  if ((image || video || voice || transcribe) && REC) discardRecording();
  el("pg-prompt").placeholder = image
    ? `Describe the picture you want. Press Send, or ${MOD_KEY}+Enter.`
    : video
    ? `Describe the video you want. Press Send, or ${MOD_KEY}+Enter.`
    : `Ask anything. Press Send, or ${MOD_KEY}+Enter.`;
  // What a key qualifies for changes with the task, so the Key list is
  // rebuilt before the Model list is fetched for whichever key that leaves
  // selected.
  await renderPlaygroundTargets();
  loadPlaygroundModels();
  // renderPlaygroundTargets() can silently swap the selected key (setting
  // .value directly fires no change event), so the per-key gates have to
  // be re-run here too. Without this, a control gated for the previous
  // key's provider (e.g. OpenAI's "minimal" reasoning effort) stayed
  // enabled after the task switch moved the key to Gemini underneath it.
  applyKeyGates();
  // Fresh entry into video mode resets to the provider's default rather
  // than carrying over whatever the slider last held.
  if (video) syncVideoDuration(true);
}

el("pg-mode").addEventListener("change", applyMode);

// Gemini's Veo only accepts 4, 6, or 8 second clips; xAI's Grok Imagine
// takes any whole second from 1. Rather than one compromise range, the
// slider's min/max/step switch to match whichever provider the selected
// key belongs to. `reset` snaps to that provider's default (fresh entry
// into video mode); otherwise the current value is clamped and rounded
// onto the new step so a mid-task key swap doesn't silently jump.
function syncVideoDuration(reset) {
  const input = el("pg-video-duration");
  // Read the current value before touching min/max/step: reassigning those
  // attributes one at a time can leave the input in a momentarily invalid
  // state (e.g. old value 4 against a new min of 1 with the old step of 2
  // still in place), and the browser silently re-snaps .value right then,
  // not when the whole set finishes. Reading afterward would pick up that
  // intermediate snap instead of the value this function was asked to
  // carry forward.
  const previous = Number(input.value);
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  const gemini = target && target.provider === "gemini";
  const config = gemini
    ? { min: 4, max: 8, step: 2, def: 4 }
    : { min: 1, max: 15, step: 1, def: 2 };
  input.min = config.min;
  input.max = config.max;
  input.step = config.step;
  if (reset) {
    input.value = config.def;
  } else {
    const clamped = Math.min(config.max, Math.max(config.min, previous));
    input.value = config.min + Math.round((clamped - config.min) / config.step) * config.step;
  }
  paintVideoDurationLabel();
}

function paintVideoDurationLabel() {
  const input = el("pg-video-duration");
  el("pg-video-duration-value").textContent = `${input.value}s`;
  // Every provider charges per second, so a longer render past the
  // shorter end of either range is where the warning earns its keep.
  el("pg-video-duration-warning").hidden = Number(input.value) <= 4;
}

el("pg-video-duration").addEventListener("input", paintVideoDurationLabel);

// --- voice conversation -------------------------------------------------

// Caller audio rate each provider's realtime API expects (Gemini resamples
// server-side to 16 kHz regardless of what it's sent; OpenAI and xAI take
// 24 kHz). Generated audio is 24 kHz on all three, so playback needs no
// per-provider branch.
const REALTIME_INPUT_RATE = { openai: 24000, xai: 24000, gemini: 16000 };
const REALTIME_OUTPUT_RATE = 24000;

// Non-null for the life of one session: {ws, provider, pendingText, playCtx,
// playAnalyser, playCursor, mic*, recognition, talking, micStarting,
// liveBubble, liveBody, liveUserBubble, liveUserBody, gotFinalMessage}.
// playCursor is the Web Audio clock time the next audio_delta should start
// at, so consecutive chunks queue back to back instead of overlapping or
// leaving silence between them. `talking` means the microphone is actively
// streaming, which is the whole session for the common case: it starts
// false only for a session opened by typing (Send, with no mic yet) or
// during the short window before a mic permission prompt resolves.
let PG_VOICE = null;

function setVoiceStatus(text) {
  el("pg-voice-status").textContent = text;
}

// The mic button doubles as the session's on/off indicator: filled and
// glowing while a session is live, plain otherwise. There's no separate
// start/stop button for it to sit next to.
function setVoiceMicIndicator(active) {
  const btn = el("pg-voice-talk");
  btn.classList.toggle("live", active);
  btn.setAttribute("aria-pressed", String(active));
  btn.title = active ? "End the voice conversation" : "Start a voice conversation";
  btn.setAttribute("aria-label", btn.title);
}

// A short two-tone rise, generated rather than shipped as an audio file:
// the only signal it needs to carry is "the session is live now", which
// two tones already say, and it means one fewer asset in the page.
function playVoiceChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(660, ctx.currentTime);
    osc.frequency.setValueAtTime(880, ctx.currentTime + 0.09);
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.32);
    osc.onended = () => ctx.close();
  } catch {
    // No Web Audio, or the browser blocked audio before a user gesture:
    // the status line already says the session started, so silence here
    // costs nothing essential.
  }
}

// withMic: false for a session opened by typing (Send with no session
// yet), for someone who wants to hear the model without ever using the
// microphone; the mic button still ends that session on the next tap.
// pendingText: a line typed before any session existed, sent the moment
// the provider confirms the session so it isn't lost to the connect delay.
function startVoiceSession({ withMic = true, pendingText = null } = {}) {
  if (PG_VOICE) return;
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  const model = el("pg-model").value;
  if (!target || !model) {
    setVoiceStatus("pick a key and a model first");
    return;
  }
  clearTranscriptPlaceholder();
  const url = new URL("/api/realtime", location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("target", target.id);
  url.searchParams.set("model", model);
  // Standing instructions doubles as the session's system prompt: set once
  // here, same as the field's placeholder says.
  const instructions = el("pg-system").value.trim();
  if (instructions) url.searchParams.set("instructions", instructions);

  const ws = new WebSocket(url);
  PG_VOICE = {
    ws,
    provider: target.provider,
    pendingText,
    playCtx: null,
    playAnalyser: null,
    playCursor: 0,
    micStream: null,
    micCtx: null,
    micNode: null,
    micSource: null,
    micMute: null,
    micAnalyser: null,
    micStarting: false,
    recognition: null,
    talking: false,
    liveBubble: null,
    liveBody: null,
    liveUserBubble: null,
    liveUserBody: null,
    gotFinalMessage: false,
  };
  setVoiceMicIndicator(true);
  startVoiceWave();
  setVoiceStatus(`connecting to ${target.provider}…`);
  ws.onmessage = (event) => handleVoiceMessage(event.data);
  ws.onclose = () => {
    // Already torn down via the mic button; this is the socket finishing
    // its own close handshake after that, not a new event.
    if (!PG_VOICE) return;
    if (!PG_VOICE.gotFinalMessage) setVoiceStatus("connection closed");
    endVoiceSession();
  };
  if (withMic) startVoiceMic();
}

function handleVoiceMessage(raw) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return;
  }
  if (!PG_VOICE || !data || typeof data.type !== "string") return;
  switch (data.type) {
    case "session_started":
      setVoiceStatus(`connected to ${PG_VOICE.provider}`);
      playVoiceChime();
      if (PG_VOICE.pendingText) {
        const text = PG_VOICE.pendingText;
        PG_VOICE.pendingText = null;
        PG_VOICE.ws.send(JSON.stringify({ type: "text_turn", text }));
        addUserTurn(text, []);
      }
      break;
    case "transcript_delta":
      appendVoiceTranscript(data.text || "");
      break;
    case "audio_delta":
      scheduleVoiceAudio(data.pcm_base64 || "");
      break;
    case "turn_complete":
      finishVoiceTurn(data.usage);
      break;
    case "interrupted":
      // Not a failure: the model was mid-reply and the caller (or its own
      // voice detection) started a new turn, so it stopped talking to
      // listen. Named plainly so it doesn't read as something broken.
      setVoiceStatus("you started talking, so the reply stopped there, go ahead");
      break;
    case "session_ended":
      PG_VOICE.gotFinalMessage = true;
      setVoiceStatus(data.reason ? `session ended: ${data.reason}` : "session ended");
      break;
    case "error":
      PG_VOICE.gotFinalMessage = true;
      setVoiceStatus(`error: ${data.message}`);
      break;
    default:
      break;
  }
}

function appendVoiceTranscript(text) {
  if (!text) return;
  if (!PG_VOICE.liveBody) {
    const bubble = addBubble("model");
    const body = document.createElement("div");
    body.className = "result-text";
    bubble.appendChild(body);
    PG_VOICE.liveBubble = bubble;
    PG_VOICE.liveBody = body;
  }
  PG_VOICE.liveBody.appendChild(document.createTextNode(text));
  const transcript = el("pg-transcript");
  transcript.scrollTop = transcript.scrollHeight;
}

function finishVoiceTurn(usage) {
  if (PG_VOICE.liveBubble && usage) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = usageLabel(usage);
    PG_VOICE.liveBubble.appendChild(meta);
  }
  PG_VOICE.liveBubble = null;
  PG_VOICE.liveBody = null;
}

function ensurePlaybackContext() {
  if (!PG_VOICE.playCtx) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: REALTIME_OUTPUT_RATE,
    });
    // Every chunk routes through this one analyser rather than straight to
    // the destination, so the waveform has something to read while the
    // model is the one making sound.
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.connect(ctx.destination);
    PG_VOICE.playCtx = ctx;
    PG_VOICE.playAnalyser = analyser;
    PG_VOICE.playCursor = ctx.currentTime;
  }
  return PG_VOICE.playCtx;
}

// Decodes and schedules one chunk back to back with whatever is already
// queued, so a steady stream of deltas plays as one continuous voice
// instead of a click between every chunk.
function scheduleVoiceAudio(base64) {
  if (!base64) return;
  const binary = atob(base64);
  const frames = binary.length >> 1;
  if (!frames) return;
  const ctx = ensurePlaybackContext();
  const buffer = ctx.createBuffer(1, frames, REALTIME_OUTPUT_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) {
    let sample = binary.charCodeAt(i * 2) | (binary.charCodeAt(i * 2 + 1) << 8);
    if (sample >= 0x8000) sample -= 0x10000;
    channel[i] = sample / 32768;
  }
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(PG_VOICE.playAnalyser);
  const startAt = Math.max(ctx.currentTime, PG_VOICE.playCursor);
  source.start(startAt);
  PG_VOICE.playCursor = startAt + buffer.duration;
}

// --- the waveform: whichever of mic and playback is louder this frame -----
//
// Continuous mode means both can be live at once (the caller barges in
// while the model is still talking), so the wave reads the larger of the
// two levels each frame rather than switching sources on a boolean.

let VOICE_WAVE_FRAME = null;
let VOICE_LEVELS = [];

function startVoiceWave() {
  VOICE_LEVELS = [];
  el("pg-voice-wave").hidden = false;
  if (VOICE_WAVE_FRAME == null) VOICE_WAVE_FRAME = requestAnimationFrame(voiceWaveLoop);
}

function stopVoiceWave() {
  if (VOICE_WAVE_FRAME != null) cancelAnimationFrame(VOICE_WAVE_FRAME);
  VOICE_WAVE_FRAME = null;
  el("pg-voice-wave").hidden = true;
  VOICE_LEVELS = [];
}

function analyserLevel(analyser) {
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);
  let sum = 0;
  for (const sample of samples) {
    const centred = (sample - 128) / 128;
    sum += centred * centred;
  }
  return Math.sqrt(sum / samples.length);
}

function voiceWaveLoop() {
  if (!PG_VOICE) {
    VOICE_WAVE_FRAME = null;
    return;
  }
  const micLevel = PG_VOICE.micAnalyser ? analyserLevel(PG_VOICE.micAnalyser) : 0;
  const playLevel = PG_VOICE.playAnalyser ? analyserLevel(PG_VOICE.playAnalyser) : 0;
  drawVoiceWave(Math.max(micLevel, playLevel));
  VOICE_WAVE_FRAME = requestAnimationFrame(voiceWaveLoop);
}

function drawVoiceWave(level) {
  const canvas = el("pg-voice-wave");
  const width = canvas.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.floor(width * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(canvas.clientHeight * ratio);
  }
  const ctx = canvas.getContext("2d");
  VOICE_LEVELS.push(level);

  const barWidth = 3 * ratio;
  const spacing = 2 * ratio;
  const slots = Math.floor(canvas.width / (barWidth + spacing));
  if (VOICE_LEVELS.length > slots) VOICE_LEVELS.splice(0, VOICE_LEVELS.length - slots);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = getComputedStyle(canvas).getPropertyValue("color") || "#c69c6d";
  const mid = canvas.height / 2;
  VOICE_LEVELS.forEach((level, index) => {
    const height = Math.max(2 * ratio, Math.sqrt(level) * canvas.height * 0.9);
    const x = index * (barWidth + spacing);
    ctx.fillRect(x, mid - height / 2, barWidth, height);
  });
}

// --- live captions of the caller's own speech ------------------------------

// Local to the browser, independent of what the provider hears: a caption
// of the microphone, not a transcript of the turn the provider received.
// Unsupported browsers simply show no caption; the session itself doesn't
// depend on this at all.
function speechRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

// Runs for the whole session rather than per utterance: each recognizer
// result finalizes into its own bubble via `isFinal`, and the browser's own
// habit of stopping continuous recognition after a pause is covered by
// restarting it from `onend` as long as the mic is still meant to be live.
function startVoiceCaption() {
  const Ctor = speechRecognitionCtor();
  if (!Ctor) return;
  const recognition = new Ctor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = navigator.language || "en-US";
  recognition.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      updateVoiceCaption(result[0].transcript);
      if (result.isFinal) finishVoiceCaption();
    }
  };
  // A caption failing (no permission, no speech, browser policy) is not a
  // session failure; it just means no caption this turn.
  recognition.onerror = () => {};
  recognition.onend = () => {
    if (PG_VOICE && PG_VOICE.recognition === recognition && PG_VOICE.talking) {
      try {
        recognition.start();
      } catch {
        // Already restarting on its own, or the mic just stopped; either
        // way there's nothing more to do here.
      }
    }
  };
  try {
    recognition.start();
  } catch {
    return;
  }
  PG_VOICE.recognition = recognition;
}

function updateVoiceCaption(text) {
  if (!PG_VOICE || !text) return;
  if (!PG_VOICE.liveUserBody) {
    const bubble = addBubble("user");
    const body = document.createElement("div");
    body.className = "result-text";
    bubble.appendChild(body);
    PG_VOICE.liveUserBubble = bubble;
    PG_VOICE.liveUserBody = body;
  }
  // Replaced wholesale rather than appended: the recognizer keeps revising
  // its own interim guess for the phrase in progress, not only adding to it.
  PG_VOICE.liveUserBody.textContent = text;
  const transcript = el("pg-transcript");
  transcript.scrollTop = transcript.scrollHeight;
}

// One utterance is done: the next caption starts a fresh bubble instead of
// overwriting this one.
function finishVoiceCaption() {
  if (!PG_VOICE) return;
  PG_VOICE.liveUserBubble = null;
  PG_VOICE.liveUserBody = null;
}

function stopVoiceCaption() {
  if (!PG_VOICE) return;
  const recognition = PG_VOICE.recognition;
  PG_VOICE.recognition = null;
  if (recognition) {
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    try {
      recognition.stop();
    } catch {
      // Already stopped, or never started without error; nothing left to release.
    }
  }
  finishVoiceCaption();
}

// Starts the microphone for the current session: the whole tap-to-toggle
// flow (fresh session or a session that was opened by typing) routes
// through here, and it streams continuously until the session ends rather
// than per press.
async function startVoiceMic() {
  if (!PG_VOICE || PG_VOICE.talking || PG_VOICE.micStarting) return;
  PG_VOICE.micStarting = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setVoiceStatus(
      err && err.name === "NotAllowedError"
        ? "your browser blocked microphone access, allow it for this page and try again"
        : "could not use the microphone"
    );
    if (PG_VOICE) PG_VOICE.micStarting = false;
    return;
  }
  // The session ended while the permission prompt was up.
  if (!PG_VOICE) {
    stream.getTracks().forEach((track) => track.stop());
    return;
  }
  const context = new (window.AudioContext || window.webkitAudioContext)();
  const source = context.createMediaStreamSource(stream);
  const node = context.createScriptProcessor(4096, 1, 1);
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  const targetRate = REALTIME_INPUT_RATE[PG_VOICE.provider] || REALTIME_OUTPUT_RATE;
  node.onaudioprocess = (event) => {
    if (!PG_VOICE || !PG_VOICE.talking) return;
    const samples = downsample(
      new Float32Array(event.inputBuffer.getChannelData(0)), context.sampleRate, targetRate
    );
    if (!samples.length) return;
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = Math.round(clamped * 32767);
    }
    PG_VOICE.ws.send(JSON.stringify({ type: "audio_chunk", pcm_base64: base64OfBytes(pcm.buffer) }));
  };
  source.connect(node);
  source.connect(analyser);
  // Same silent-routing trick as the recorder: a ScriptProcessor only runs
  // while connected to a destination, and this keeps the microphone from
  // being played back through the speakers.
  const mute = context.createGain();
  mute.gain.value = 0;
  node.connect(mute);
  mute.connect(context.destination);

  PG_VOICE.micStream = stream;
  PG_VOICE.micCtx = context;
  PG_VOICE.micNode = node;
  PG_VOICE.micSource = source;
  PG_VOICE.micMute = mute;
  PG_VOICE.micAnalyser = analyser;
  PG_VOICE.talking = true;
  PG_VOICE.micStarting = false;
  setVoiceStatus("listening…");
  startVoiceCaption();
}

// Stops the microphone without ending the session. Deliberately doesn't
// send an end-of-turn signal: the provider's own voice-activity detection
// owns turn boundaries for the whole continuous stream, and this only runs
// as part of ending the session outright, not on every pause.
function stopVoiceMic() {
  if (!PG_VOICE || !PG_VOICE.talking) return;
  PG_VOICE.talking = false;
  stopVoiceCaption();
  const { micStream, micCtx, micNode, micSource, micMute, micAnalyser } = PG_VOICE;
  micNode.onaudioprocess = null;
  micSource.disconnect();
  micNode.disconnect();
  micMute.disconnect();
  micAnalyser.disconnect();
  micStream.getTracks().forEach((track) => track.stop());
  micCtx.close();
  PG_VOICE.micStream = null;
  PG_VOICE.micCtx = null;
  PG_VOICE.micNode = null;
  PG_VOICE.micSource = null;
  PG_VOICE.micMute = null;
  PG_VOICE.micAnalyser = null;
}

// The mic button's single click handler: activate or deactivate the whole
// session, one tap either way. If mic permission never came through (the
// session is open but never got to start talking), the next tap still ends
// it rather than retrying the microphone forever.
function toggleVoiceSession() {
  if (PG_VOICE) {
    endVoiceSession();
  } else {
    startVoiceSession();
  }
}

// Send always works, spoken session or not: typed text starts a session
// with no mic if none is open yet, so someone who can't speak can still
// hear the model reply.
function sendVoiceText() {
  const box = el("pg-voice-prompt");
  const text = box.value.trim();
  if (!text) return;
  if (!PG_VOICE) {
    startVoiceSession({ withMic: false, pendingText: text });
    box.value = "";
    updateVoiceSendEnabled();
    return;
  }
  if (PG_VOICE.ws.readyState !== WebSocket.OPEN) {
    PG_VOICE.pendingText = text;
    box.value = "";
    updateVoiceSendEnabled();
    return;
  }
  PG_VOICE.ws.send(JSON.stringify({ type: "text_turn", text }));
  addUserTurn(text, []);
  box.value = "";
  updateVoiceSendEnabled();
}

// Idempotent: safe from the mic button, from a mode switch away from
// voice, and from the socket's own close event, whichever gets here first.
function endVoiceSession() {
  if (!PG_VOICE) return;
  stopVoiceMic();
  const { ws, playCtx } = PG_VOICE;
  PG_VOICE = null;
  if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
  if (playCtx) playCtx.close();
  stopVoiceWave();
  setVoiceMicIndicator(false);
  setVoiceStatus("not connected");
}

el("pg-voice-talk").addEventListener("click", toggleVoiceSession);
el("pg-voice-send").addEventListener("click", sendVoiceText);
el("pg-voice-prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    sendVoiceText();
  }
});

// --- live transcription -----------------------------------------------------

// Every STT provider takes 16 kHz 16-bit mono, so the browser downsamples
// to one rate with no per-provider branch, unlike the realtime table above.
const STT_SAMPLE_RATE = 16000;

// Non-null for the life of one transcription session: {ws, mic*, running,
// micStarting, finishing, liveBubble, liveBody, gotFinalMessage}. The same
// tap-to-toggle flow as a voice session, minus playback and typing:
// nothing talks back, so audio only flows up.
let PG_STT = null;

function setTranscribeStatus(text) {
  el("pg-transcribe-status").textContent = text;
}

function setTranscribeMicIndicator(active) {
  const btn = el("pg-transcribe-talk");
  btn.classList.toggle("live", active);
  btn.setAttribute("aria-pressed", String(active));
  btn.title = active ? "Finish transcribing" : "Start transcribing";
  btn.setAttribute("aria-label", btn.title);
}

function startTranscribeSession() {
  if (PG_STT) return;
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  const model = el("pg-model").value;
  if (!target || !model) {
    setTranscribeStatus("pick a key and a model first");
    return;
  }
  clearTranscriptPlaceholder();
  const url = new URL("/api/transcribe", location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("target", target.id);
  url.searchParams.set("model", model);
  url.searchParams.set("sample_rate", String(STT_SAMPLE_RATE));

  const ws = new WebSocket(url);
  PG_STT = {
    ws,
    micStream: null,
    micCtx: null,
    micNode: null,
    micSource: null,
    micMute: null,
    running: false,
    micStarting: false,
    finishing: false,
    liveBubble: null,
    liveBody: null,
    gotFinalMessage: false,
    firstFinalText: null,
  };
  setTranscribeMicIndicator(true);
  setTranscribeStatus(`connecting to ${target.provider}…`);
  ws.onmessage = (event) => handleTranscribeMessage(event.data);
  ws.onclose = () => {
    // Already torn down via the mic button; this is the socket finishing
    // its own close handshake after that, not a new event.
    if (!PG_STT) return;
    if (!PG_STT.gotFinalMessage) setTranscribeStatus("connection closed");
    endTranscribeSession();
  };
  startTranscribeMic();
}

function handleTranscribeMessage(raw) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return;
  }
  if (!PG_STT || !data || typeof data.type !== "string") return;
  switch (data.type) {
    case "session_started":
      // The microphone's own start reports "listening…"; this only fills
      // the window before permission comes through. Some providers send
      // no session-start frame at all, so nothing else depends on it.
      if (!PG_STT.running) setTranscribeStatus("connected, starting the microphone…");
      break;
    case "interim_transcript":
      updateTranscribeInterim(data.text || "");
      break;
    case "final_transcript":
      finishTranscribeUtterance(data);
      break;
    case "session_ended": {
      PG_STT.gotFinalMessage = true;
      const billed = data.audio_duration_seconds;
      setTranscribeStatus(
        billed != null
          ? `session ended, ${billed}s of audio billed`
          : data.reason
          ? `session ended: ${data.reason}`
          : "session ended"
      );
      PG_STT.ws.close();
      break;
    }
    case "error":
      PG_STT.gotFinalMessage = true;
      setTranscribeStatus(`error: ${data.message}`);
      break;
    default:
      break;
  }
}

// The utterance in progress lives in one bubble, replaced wholesale on
// every interim: the recognizer keeps revising its guess, not only adding
// to it, the same behavior the voice captions handle.
function ensureTranscribeBubble() {
  if (PG_STT.liveBody) return;
  const bubble = addBubble("user");
  const body = document.createElement("div");
  body.className = "result-text";
  bubble.appendChild(body);
  PG_STT.liveBubble = bubble;
  PG_STT.liveBody = body;
}

function updateTranscribeInterim(text) {
  if (!PG_STT || !text) return;
  ensureTranscribeBubble();
  PG_STT.liveBody.textContent = text;
  const transcript = el("pg-transcript");
  transcript.scrollTop = transcript.scrollHeight;
}

// One recognized stretch is done: pin its final text, note the provider's
// confidence when it reports one, and let the next words start fresh.
function finishTranscribeUtterance(data) {
  if (!PG_STT || !data.text) return;
  // The first recognized words double as the conversation's History title,
  // the way a text conversation's first prompt does.
  if (PG_STT.firstFinalText == null) PG_STT.firstFinalText = data.text;
  ensureTranscribeBubble();
  PG_STT.liveBody.textContent = data.text;
  if (typeof data.confidence === "number") {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `confidence ${(data.confidence * 100).toFixed(0)}%`;
    PG_STT.liveBubble.appendChild(meta);
  }
  PG_STT.liveBubble = null;
  PG_STT.liveBody = null;
  const transcript = el("pg-transcript");
  transcript.scrollTop = transcript.scrollHeight;
  // Saved as the words firm up, not only when the session ends, so a page
  // reload or dropped connection mid-session loses nothing already heard.
  saveCurrentConversation(PG_STT.firstFinalText);
}

// The same capture pipeline as the voice session's microphone, at the STT
// rate: getUserMedia -> ScriptProcessor -> downsample -> 16-bit PCM ->
// base64 in a JSON frame.
async function startTranscribeMic() {
  if (!PG_STT || PG_STT.running || PG_STT.micStarting) return;
  PG_STT.micStarting = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setTranscribeStatus(
      err && err.name === "NotAllowedError"
        ? "your browser blocked microphone access, allow it for this page and try again"
        : "could not use the microphone"
    );
    if (PG_STT) PG_STT.micStarting = false;
    return;
  }
  // The session ended while the permission prompt was up.
  if (!PG_STT) {
    stream.getTracks().forEach((track) => track.stop());
    return;
  }
  const context = new (window.AudioContext || window.webkitAudioContext)();
  const source = context.createMediaStreamSource(stream);
  const node = context.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = (event) => {
    if (!PG_STT || !PG_STT.running) return;
    const samples = downsample(
      new Float32Array(event.inputBuffer.getChannelData(0)), context.sampleRate, STT_SAMPLE_RATE
    );
    if (!samples.length) return;
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = Math.round(clamped * 32767);
    }
    PG_STT.ws.send(JSON.stringify({ type: "audio_chunk", pcm_base64: base64OfBytes(pcm.buffer) }));
  };
  source.connect(node);
  // Same silent-routing trick as the voice session: a ScriptProcessor only
  // runs while connected to a destination, and this keeps the microphone
  // from being played back through the speakers.
  const mute = context.createGain();
  mute.gain.value = 0;
  node.connect(mute);
  mute.connect(context.destination);

  PG_STT.micStream = stream;
  PG_STT.micCtx = context;
  PG_STT.micNode = node;
  PG_STT.micSource = source;
  PG_STT.micMute = mute;
  PG_STT.running = true;
  PG_STT.micStarting = false;
  setTranscribeStatus("listening…");
}

function stopTranscribeMic() {
  if (!PG_STT || !PG_STT.running) return;
  PG_STT.running = false;
  const { micStream, micCtx, micNode, micSource, micMute } = PG_STT;
  micNode.onaudioprocess = null;
  micSource.disconnect();
  micNode.disconnect();
  micMute.disconnect();
  micStream.getTracks().forEach((track) => track.stop());
  micCtx.close();
  PG_STT.micStream = null;
  PG_STT.micCtx = null;
  PG_STT.micNode = null;
  PG_STT.micSource = null;
  PG_STT.micMute = null;
}

// First tap starts a session; the second stops the microphone and asks the
// provider to finish, so its billing summary can arrive before the socket
// closes. A tap during that finish window (or on a session that never got
// a microphone) ends the session outright instead of waiting forever.
function toggleTranscribeSession() {
  if (!PG_STT) {
    startTranscribeSession();
    return;
  }
  if (!PG_STT.finishing && PG_STT.running && PG_STT.ws.readyState === WebSocket.OPEN) {
    PG_STT.finishing = true;
    stopTranscribeMic();
    setTranscribeMicIndicator(false);
    setTranscribeStatus("finishing…");
    PG_STT.ws.send(JSON.stringify({ type: "finish" }));
    return;
  }
  endTranscribeSession();
}

// Idempotent: safe from the mic button, from a mode switch away from
// transcribe, and from the socket's own close event, whichever gets here
// first. The final status (the billed seconds, or an error) stays on
// screen rather than being wiped to "not connected".
function endTranscribeSession() {
  if (!PG_STT) return;
  stopTranscribeMic();
  const { ws, gotFinalMessage, firstFinalText } = PG_STT;
  PG_STT = null;
  if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
  setTranscribeMicIndicator(false);
  if (!gotFinalMessage) setTranscribeStatus("not yet started");
  // The session is the whole exchange, so its end is the one save point —
  // unlike text mode, which saves after every reply. Every way a session
  // ends funnels through here, and the save declines on its own when the
  // transcript holds no bubbles at all.
  saveCurrentConversation(firstFinalText);
}

el("pg-voice-new").addEventListener("click", startNewConversation);
el("pg-transcribe-new").addEventListener("click", startNewConversation);

el("pg-transcribe-talk").addEventListener("click", toggleTranscribeSession);

function openLightbox(source) {
  const overlay = document.createElement("div");
  overlay.className = "lightbox";
  overlay.tabIndex = -1;

  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (event) => {
    if (event.key === "Escape") close();
  };

  // A visible control, because clicking the backdrop and pressing Escape
  // both work but neither is discoverable by looking at the screen.
  const dismiss = document.createElement("button");
  dismiss.className = "lightbox-close";
  dismiss.type = "button";
  dismiss.textContent = "\u00d7";
  dismiss.title = "Close (Esc)";
  dismiss.setAttribute("aria-label", "Close the picture");
  dismiss.addEventListener("click", close);
  overlay.appendChild(dismiss);

  const full = document.createElement("img");
  full.src = source;
  full.alt = "The generated picture, full size";
  // Clicking the picture itself should not dismiss it; the backdrop and
  // the close button are the ways out.
  full.addEventListener("click", (event) => event.stopPropagation());
  overlay.appendChild(full);

  overlay.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  document.body.appendChild(overlay);
  dismiss.focus();
}

// Neither image nor video generation reports usage on any current
// provider, so "usage unreported" there is the expected outcome, not a
// symptom that needs its own caption; it would otherwise read as an
// error to a user unsure whether it needs acting on.
function generationCaption(result) {
  const parts = [result.model, formatDuration(result.round_trip_duration_ms)];
  const usage = usageLabel(result.usage);
  if (usage !== "usage unreported") parts.push(usage);
  return parts.join(" · ");
}

function addImageBubble(result) {
  const bubble = addBubble("model");
  (result.images || []).forEach((image) => {
    const picture = document.createElement("img");
    picture.className = "pg-picture";
    picture.alt = "The generated picture";
    picture.src = `data:${image.media_type};base64,${image.base64_data}`;
    picture.title = "Click to see it full size";
    picture.addEventListener("click", () => openLightbox(picture.src));
    bubble.appendChild(picture);
    const save = document.createElement("a");
    save.href = picture.src;
    save.download = `keycall-image.${(image.media_type || "image/png").split("/")[1]}`;
    save.textContent = "Save this picture";
    save.className = "meta";
    bubble.appendChild(save);
  });
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = generationCaption(result);
  bubble.appendChild(meta);
  (result.warnings || []).forEach((warning) => {
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = warning;
    bubble.appendChild(note);
  });
  return bubble;
}

function addVideoBubble(result) {
  const bubble = addBubble("model");
  (result.videos || []).forEach((video) => {
    const player = document.createElement("video");
    player.className = "pg-video";
    player.controls = true;
    // Embedded the same way as a picture: the base64 bytes are already in
    // the response, and a provider's own video_url can expire, so the data
    // URI is the one source guaranteed to still work when this bubble is
    // scrolled back to later.
    player.src = `data:${video.media_type};base64,${video.base64_data}`;
    bubble.appendChild(player);
    const save = document.createElement("a");
    save.href = player.src;
    save.download = `keycall-video.${(video.media_type || "video/mp4").split("/")[1]}`;
    save.textContent = "Save this video";
    save.className = "meta";
    bubble.appendChild(save);
  });
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = generationCaption(result);
  bubble.appendChild(meta);
  (result.warnings || []).forEach((warning) => {
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = warning;
    bubble.appendChild(note);
  });
  return bubble;
}

// --- transcript -------------------------------------------------------------

function transcriptEmpty() {
  const box = el("pg-transcript");
  clear(box);
  const empty = document.createElement("div");
  empty.className = "pg-empty";
  const title = document.createElement("strong");
  title.textContent = "Your conversation will appear here";
  const hint = document.createElement("span");
  hint.textContent = "Pick a key and a model on the left, then type below.";
  empty.appendChild(title);
  empty.appendChild(hint);
  box.appendChild(empty);
  // Nothing to start over from yet: an empty transcript already is a new
  // chat, so the button would only offer to reset a reset.
  el("pg-new").hidden = true;
}

function clearTranscriptPlaceholder() {
  const placeholder = el("pg-transcript").querySelector(".pg-empty");
  if (placeholder) placeholder.remove();
  el("pg-new").hidden = false;
}

function addBubble(kind) {
  clearTranscriptPlaceholder();
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind}`;
  const transcript = el("pg-transcript");
  transcript.appendChild(bubble);
  // The transcript scrolls on its own now, so a new turn has to bring
  // itself into view or it appears to do nothing.
  transcript.scrollTop = transcript.scrollHeight;
  return bubble;
}

function addUserTurn(text, attachmentLabels) {
  const bubble = addBubble("user");
  const labels = attachmentLabels || [];
  const voiceOnly = !text && labels.length === 1 && labels[0] === "recording";
  if (voiceOnly) {
    // A recording with no typed words is still a message. Drawing its shape
    // says what was sent; "(no message)" said the opposite of the truth.
    // The turn owns a copy of the audio so every recording in a session
    // stays playable, however many follow it.
    bubble.appendChild(playableWaveform(PG_AUDIO_SHAPE, PG_MEDIA.audio));
  } else {
    const body = document.createElement("div");
    body.className = "result-text";
    body.textContent = text || "(no message)";
    bubble.appendChild(body);
  }
  if (labels.length && !voiceOnly) {
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = `with a ${labels.join(" and a ")} attached`;
    bubble.appendChild(note);
  }
  return bubble;
}

// Amplitude envelope of the attached recording, 0..1 per bar. Null for a
// picked file, which gets an even shape instead: the point is to show that
// sound was sent, and inventing a contour for bytes we never measured would
// be a drawing of nothing.
let PG_AUDIO_SHAPE = null;

/** A voice turn: play/pause on the left, the clip's shape beside it.
 *  Each turn holds its own object URL, so replaying the third recording
 *  after sending a fifth still works — the composer's copy is revoked when
 *  it clears, and sharing one URL would take the transcript's audio with
 *  it. */
function playableWaveform(levels, part) {
  const wrap = document.createElement("div");
  wrap.className = "turn-voice";
  const audio = new Audio();
  const url = objectUrlFor(part);
  if (url) audio.src = url;

  const play = document.createElement("button");
  play.type = "button";
  play.className = "icon-btn play";
  play.setAttribute("aria-label", "Play this recording");
  const setIcon = () => {
    play.innerHTML = audio.paused ? ICON_PLAY : ICON_PAUSE;
    play.setAttribute("aria-label", audio.paused ? "Play this recording" : "Pause");
  };
  setIcon();
  play.addEventListener("click", () => {
    if (audio.paused) {
      // Only one clip at a time, so starting a second does not talk over
      // the first.
      document.querySelectorAll("audio").forEach((other) => {
        if (other !== audio) other.pause();
      });
      audio.play();
    } else {
      audio.pause();
    }
  });
  audio.addEventListener("play", setIcon);
  audio.addEventListener("pause", setIcon);
  audio.addEventListener("ended", setIcon);
  if (!url) play.disabled = true;

  wrap.appendChild(play);
  wrap.appendChild(waveformEl(levels));
  return wrap;
}

const ICON_PLAY =
  '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">' +
  '<path fill="currentColor" d="M8 5v14l11-7z"/></svg>';
const ICON_PAUSE =
  '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">' +
  '<path fill="currentColor" d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';

/** A blob: URL for an attachment the browser is holding as base64. Blob
 *  rather than a data: URL so the page's media-src stays narrow. */
function objectUrlFor(part) {
  if (!part || !part.data_base64) return null;
  const binary = atob(part.data_base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(
    new Blob([bytes], { type: part.media_type || "audio/wav" })
  );
}

function waveformEl(levels) {
  const bars = levels && levels.length ? levels : new Array(28).fill(0.45);
  const wrap = document.createElement("div");
  wrap.className = "turn-wave";
  wrap.setAttribute("role", "img");
  wrap.setAttribute("aria-label", "a voice recording");
  bars.forEach((level) => {
    const bar = document.createElement("span");
    // Floored so a silent stretch still reads as part of the clip rather
    // than a hole in it.
    bar.style.height = `${Math.max(12, Math.min(1, level) * 100)}%`;
    wrap.appendChild(bar);
  });
  return wrap;
}

/** Reduce recorded samples to a fixed number of bars for the turn bubble. */
function envelopeOf(samples, bars = 28) {
  const out = [];
  const span = Math.floor(samples.length / bars) || 1;
  let peak = 0;
  for (let i = 0; i < bars; i++) {
    let sum = 0;
    let count = 0;
    for (let j = i * span; j < Math.min((i + 1) * span, samples.length); j++) {
      sum += samples[j] * samples[j];
      count++;
    }
    const rms = count ? Math.sqrt(sum / count) : 0;
    out.push(rms);
    peak = Math.max(peak, rms);
  }
  // Normalised against the clip's own peak, so a quiet recording still
  // draws a readable shape rather than a flat line.
  return peak > 0 ? out.map((value) => value / peak) : out;
}

// --- attachments ------------------------------------------------------------

// Pictures, sound files, and documents all travel the same way: the browser
// holds the picked bytes as base64, the server decodes them back into the
// same ImageInput/AudioInput/FileInput a library caller would construct, and
// the adapters apply their own rules. The Playground exercises the same path
// rather than a viewer-only shortcut.
//
// `field` is the JSON key posted to the server. `noun` is what a person
// calls the thing; it appears in the status line and in the refusal, so the
// wording never says "AudioInput" to someone who just spoke into a
// microphone. "recording" covers both a clip captured here and a sound file
// picked from disk, and takes "a" in front of it, which "audio" does not.
const ATTACHMENTS = [
  { id: "image", field: "images", noun: "picture", hasUrl: true },
  { id: "audio", field: "audio", noun: "recording", hasUrl: false },
  { id: "file", field: "files", noun: "document", hasUrl: false, named: true },
];

// Picked bytes per kind, keyed by ATTACHMENTS id.
const PG_MEDIA = {};

ATTACHMENTS.forEach(({ id, noun, named }) => {
  PG_MEDIA[id] = null;
  const picker = el(`pg-${id}-file`);
  const status = el(`pg-${id}-status`);

  el(`pg-${id}-on`).addEventListener("change", () => {
    el(`pg-${id}-panel`).hidden = !el(`pg-${id}-on`).checked;
    updateSuggestedBudget();
  });

  el(`pg-${id}-clear`).addEventListener("click", () => {
    PG_MEDIA[id] = null;
    picker.value = "";
    if (el(`pg-${id}-url`)) el(`pg-${id}-url`).value = "";
    status.textContent = "";
    if (id === "audio") {
      // Remove is reachable mid-recording. Tearing down the capture first
      // stops the microphone and the timer; without this the recorder keeps
      // running and quietly re-attaches a clip the user just discarded.
      if (REC) discardRecording();
      clearRecording();
    }
    updateSendEnabled();
  });

  picker.addEventListener("change", () => {
    const file = picker.files[0];
    if (!file) {
      PG_MEDIA[id] = null;
      status.textContent = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      // readAsDataURL gives "data:<type>;base64,<payload>"; the server wants
      // the payload, and the media type is re-derived from the bytes anyway.
      const encoded = String(reader.result).split(",")[1] || "";
      PG_MEDIA[id] = { data_base64: encoded, media_type: file.type || undefined };
      // Providers show a document's name to the model, so keep it.
      if (named) PG_MEDIA[id].filename = file.name;
      status.textContent = `${file.name} · ${humanSize(file.size)}`;
      // A picked file replaces a recording; leaving the player behind would
      // let someone hear one thing and send another.
      if (id === "audio") clearRecording();
    };
    reader.onerror = () => {
      PG_MEDIA[id] = null;
      status.textContent = `could not read that ${noun}`;
    };
    reader.readAsDataURL(file);
  });
});

// --- reply budget -------------------------------------------------------

// Extra headroom each token-intensive extra tends to need, on top of an
// ordinary reply. Reasoning is the largest by far: hidden reasoning tokens
// share the same budget as the visible answer (a gemini-3.5-flash reply was
// observed cut off at 2048 purely from invisible "high"-tier thinking, with
// no answer text at all), and higher effort spends more of them before the
// first visible token. Search and tools both mean a longer synthesized
// answer, or a second round after a tool result; an attachment usually
// means describing or analyzing something, which runs longer than a bare
// reply. None of these are measured per-model: they are a starting point,
// overridden the moment the reply budget field is edited by hand.
const REASONING_BUDGET_BONUS = { minimal: 512, low: 1536, medium: 4096, high: 10240 };
const WEB_SEARCH_BUDGET_BONUS = 1536;
const TOOLS_BUDGET_BONUS = 512;
const ATTACHMENT_BUDGET_BONUS = 512;
const BUDGET_FLOOR = 1024; // a bare reply with nothing selected
const BUDGET_BASE = 2048; // the field's own starting value
const BUDGET_CEILING = 16384;

// False once the reply budget has been typed into by hand; from then on
// nothing here touches it again, matching how the mode/task pickers never
// override a value the caller set on purpose.
let MAXTOK_TOUCHED = false;
el("pg-maxtok").addEventListener("input", () => {
  MAXTOK_TOUCHED = true;
});

function suggestedBudget() {
  const reasoning = el("pg-reasoning").value;
  const anyExtra =
    reasoning ||
    el("pg-search").checked ||
    el("pg-tools-on").checked ||
    ATTACHMENTS.some(({ id }) => el(`pg-${id}-on`).checked);
  if (!anyExtra) return BUDGET_FLOOR;

  let budget = BUDGET_BASE;
  budget += REASONING_BUDGET_BONUS[reasoning] || 0;
  if (el("pg-search").checked) budget += WEB_SEARCH_BUDGET_BONUS;
  if (el("pg-tools-on").checked) budget += TOOLS_BUDGET_BONUS;
  if (ATTACHMENTS.some(({ id }) => el(`pg-${id}-on`).checked)) {
    budget += ATTACHMENT_BUDGET_BONUS;
  }
  return Math.min(budget, BUDGET_CEILING);
}

function updateSuggestedBudget() {
  if (MAXTOK_TOUCHED) return;
  el("pg-maxtok").value = suggestedBudget();
}

// --- provider timeout -------------------------------------------------------

// A slider rather than a typed field, so an out-of-range or non-numeric
// value cannot be entered at all; the range and step live on the control.
// The label follows every drag, and the server hears one value on
// release, not one per pixel. It validates on its side too.
function paintTimeoutLabel() {
  el("pg-timeout-value").textContent = `${el("pg-timeout").value}s`;
}

el("pg-timeout").addEventListener("input", paintTimeoutLabel);
el("pg-timeout").addEventListener("change", async () => {
  paintTimeoutLabel();
  const data = await api("/api/settings", {
    method: "POST",
    body: { read_timeout: Number(el("pg-timeout").value) },
  });
  if (data.error) {
    // "target required" is the older server's generic POST guard: it has
    // no settings route at all, and this page is newer than the process
    // serving it. Say that, rather than relaying a message about a field
    // this request never needed.
    showToast(
      data.error.message === "target required"
        ? "This viewer's server is older than the page and has no timeout setting yet: stop keycall view and start it again."
        : `Could not set the timeout: ${data.error.message}`
    );
  }
});

// --- recording --------------------------------------------------------------

// Recorded audio is encoded to WAV here rather than handed to MediaRecorder,
// which sounds like the obvious tool and is the wrong one twice over: Chrome
// produces `audio/webm;codecs=opus`, which Gemini does not accept, and which
// KeyCall's byte sniffer does not recognise either, so the attachment would
// be refused before it left the machine. Safari produces mp4 and Firefox ogg,
// so the format would also differ per browser. Capturing raw samples and
// writing a WAV header gives one format that every browser produces, Gemini
// accepts, and the sniffer already identifies.
//
// 16 kHz mono, because speech carries fine at that rate and Gemini resamples
// to it anyway: a minute of 48 kHz stereo is about 11 MB before base64, which
// would exceed the server's 8 MB body cap on its own.
const REC_SAMPLE_RATE = 16000;
// Two minutes is far more than a Playground prompt needs and keeps the
// encoded body inside the cap with room to spare.
const REC_MAX_SECONDS = 120;

let REC = null; // {stream, context, node, source, analyser, chunks, started, timer, frame}

// The composer and the recording bar occupy the same place and never show at
// once: while recording, the only two things to decide are keep or discard.
function recordingUI(active) {
  el("pg-composer").hidden = active;
  el("pg-composer-hint").hidden = active;
  el("pg-recorder").hidden = !active;
  // Picking a file mid-recording would leave two sources of truth.
  el("pg-audio-file").disabled = active;
  if (active) el("pg-rec-accept").focus();
}

function clearRecording() {
  const preview = el("pg-audio-preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.removeAttribute("src");
  el("pg-attached").hidden = true;
  // The shape belongs to the clip that is going away. A picked file gets
  // the even placeholder instead, never the last recording's contour.
  PG_AUDIO_SHAPE = null;
}

async function startRecording() {
  let stream;
  try {
    // Asked for explicitly rather than left to the browser's defaults:
    // these are the processing steps that make a laptop microphone in a
    // room intelligible to a model, and one channel is all that is kept.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    // Denied, dismissed, or no microphone at all. Say which, because "it
    // didn't work" leaves someone poking at the button.
    el("pg-audio-status").textContent =
      err && err.name === "NotAllowedError"
        ? "your browser blocked microphone access, allow it for this page and try again"
        : err && err.name === "NotFoundError"
          ? "no microphone found on this computer"
          : `could not start recording (${(err && err.name) || "unknown error"})`;
    return;
  }
  const context = new (window.AudioContext || window.webkitAudioContext)();
  const source = context.createMediaStreamSource(stream);
  // ScriptProcessorNode is deprecated in favour of AudioWorklet, which needs
  // a separate module file served and fetched. For a local single-page tool
  // capturing a few seconds of speech, the deprecated node is supported
  // everywhere today and keeps this to one file.
  const node = context.createScriptProcessor(4096, 1, 1);
  // Feeds the waveform only. Reading levels from the analyser rather than
  // from the captured chunks keeps drawing independent of buffer size, so
  // the bars move at screen rate instead of in 4096-sample steps.
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const chunks = [];
  node.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    // Hitting the ceiling keeps what was said rather than binning it.
    if (elapsedSeconds() >= REC_MAX_SECONDS) stopRecording();
  };
  source.connect(node);
  // A ScriptProcessor only runs while connected to a destination. Routing it
  // to a muted gain node keeps it processing without playing the microphone
  // back through the speakers, which would howl.
  const mute = context.createGain();
  mute.gain.value = 0;
  node.connect(mute);
  mute.connect(context.destination);

  REC = {
    stream, context, node, source, mute, analyser, chunks,
    started: Date.now(), timer: null, frame: null, levels: [],
  };
  clearRecording();
  el("pg-audio-status").textContent = "";
  REC.timer = setInterval(() => {
    el("pg-rec-time").textContent = clockText(elapsedSeconds());
  }, 250);
  el("pg-rec-time").textContent = "0:00";
  recordingUI(true);
  drawWave();
}

function clockText(seconds) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

// A scrolling history of how loud the microphone has been, newest on the
// right. It is the one thing that distinguishes "recording" from "recording
// nothing", which is the failure a person otherwise only discovers after
// paying for a call.
function drawWave() {
  if (!REC) return;
  const canvas = el("pg-wave");
  const width = canvas.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.floor(width * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(canvas.clientHeight * ratio);
  }
  const ctx = canvas.getContext("2d");
  const samples = new Uint8Array(REC.analyser.fftSize);
  REC.analyser.getByteTimeDomainData(samples);
  let sum = 0;
  for (const sample of samples) {
    const centred = (sample - 128) / 128;
    sum += centred * centred;
  }
  REC.levels.push(Math.sqrt(sum / samples.length));

  const barWidth = 3 * ratio;
  const gap = 2 * ratio;
  const slots = Math.floor(canvas.width / (barWidth + gap));
  if (REC.levels.length > slots) REC.levels.splice(0, REC.levels.length - slots);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle =
    getComputedStyle(canvas).getPropertyValue("color") || "#c69c6d";
  const mid = canvas.height / 2;
  REC.levels.forEach((level, index) => {
    // Square-root scaling: quiet speech still shows movement, which a
    // linear scale flattens into a near-invisible line.
    const height = Math.max(2 * ratio, Math.sqrt(level) * canvas.height * 0.9);
    const x = index * (barWidth + gap);
    ctx.fillRect(x, mid - height / 2, barWidth, height);
  });
  REC.frame = requestAnimationFrame(drawWave);
}

function elapsedSeconds() {
  return REC ? Math.floor((Date.now() - REC.started) / 1000) : 0;
}

/** Release the microphone and the audio graph, returning what was captured.
 *  Split out from stopRecording so discarding mid-recording tears down the
 *  same way keeping it does — a stopped recording must never leave the
 *  microphone light on. */
function teardownRecording() {
  const { stream, context, node, source, mute, analyser, chunks, timer, frame } = REC;
  const rate = context.sampleRate;
  REC = null;
  clearInterval(timer);
  if (frame) cancelAnimationFrame(frame);
  node.onaudioprocess = null;
  source.disconnect();
  node.disconnect();
  mute.disconnect();
  analyser.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  context.close();
  recordingUI(false);
  return { chunks, rate };
}

function discardRecording() {
  teardownRecording();
}

function stopRecording() {
  if (!REC) return;
  const { chunks, rate } = teardownRecording();

  const samples = downsample(flatten(chunks), rate, REC_SAMPLE_RATE);
  if (!samples.length) {
    el("pg-audio-status").textContent = "nothing was recorded — try again";
    return;
  }
  const wav = encodeWav(samples, REC_SAMPLE_RATE);
  const seconds = samples.length / REC_SAMPLE_RATE;
  PG_MEDIA.audio = { data_base64: base64OfBytes(wav), media_type: "audio/wav" };
  // Measured from the clip itself, and kept out of PG_MEDIA so it is never
  // posted: the server has the audio and has no use for our drawing of it.
  PG_AUDIO_SHAPE = envelopeOf(samples);
  // The recording is an attachment like any other, and the toggle is what
  // says an attachment goes with the turn. Setting it keeps one source of
  // truth rather than a second, invisible rule for microphone clips.
  const toggle = el("pg-audio-on");
  toggle.checked = true;
  updateSendEnabled();
  el("pg-audio-panel").hidden = false;
  el("pg-audio-file").value = "";
  // Let the person hear what the model is about to hear, so a silent
  // recording is caught here and not blamed on the model.
  const preview = el("pg-audio-preview");
  preview.src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
  el("pg-attached").hidden = false;
  el("pg-attached").querySelector(".pg-attached-label").textContent =
    `Recording attached · ${clockText(Math.round(seconds))} · ${humanSize(wav.byteLength)}`;
  el("pg-audio-status").textContent =
    `recording · ${seconds.toFixed(1)}s · ${humanSize(wav.byteLength)}`;
  // Back to the message box so the next thing typed, or Ctrl+Enter, works
  // without reaching for the mouse.
  el("pg-prompt").focus();
}

function flatten(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function downsample(samples, from, to) {
  if (to >= from) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i++) {
    // Average the source window rather than picking one sample from it, so
    // the discarded samples don't alias into audible noise.
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), samples.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += samples[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

function encodeWav(samples, rate) {
  // Canonical 44-byte RIFF/WAVE header, 16-bit mono PCM.
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  return buffer;
}

function base64OfBytes(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  // Chunked so a long recording doesn't blow the argument limit on apply.
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

el("pg-mic").addEventListener("click", () => {
  if (!REC) startRecording();
});
el("pg-rec-accept").addEventListener("click", () => stopRecording());
el("pg-rec-cancel").addEventListener("click", () => {
  if (REC) discardRecording();
  clearRecording();
  PG_MEDIA.audio = null;
  el("pg-audio-status").textContent = "";
});
el("pg-attached-remove").addEventListener("click", () => {
  PG_MEDIA.audio = null;
  clearRecording();
  el("pg-audio-on").checked = false;
  el("pg-audio-panel").hidden = true;
  el("pg-audio-status").textContent = "";
  updateSendEnabled();
});

// Enter keeps the recording, Escape bins it. Bound on the document because
// focus sits on a button in the recording bar, and a person reaching for
// Enter should not have to think about which control has focus.
document.addEventListener("keydown", (event) => {
  if (!REC) return;
  if (event.key === "Enter") {
    event.preventDefault();
    stopRecording();
  } else if (event.key === "Escape") {
    event.preventDefault();
    discardRecording();
  }
});

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// What the browser posts for each attachment kind, plus what to write in
// the user's own bubble so the turn shows what went with it.
function attachmentsFromInput() {
  const body = {};
  const labels = [];
  ATTACHMENTS.forEach(({ id, field, noun, hasUrl }) => {
    if (!el(`pg-${id}-on`).checked || el(`pg-${id}-on`).disabled) return;
    const url = hasUrl ? el(`pg-${id}-url`).value.trim() : "";
    const part = url ? { url } : PG_MEDIA[id];
    if (!part) return;
    body[field] = [part];
    labels.push(noun);
  });
  return { body, labels };
}

// Send has nothing to do until there is a model and either text or an
// attachment — clicking it empty used to drop an inert bubble into the
// transcript saying so, which is what a disabled button with a reason
// already says for free.
function updateSendEnabled() {
  const btn = el("pg-run");
  const hasModel = Boolean(el("pg-model").value);
  const hasContent = Boolean(el("pg-prompt").value.trim()) || attachmentsFromInput().labels.length > 0;
  btn.disabled = !hasModel || !hasContent;
  btn.title = !hasModel
    ? "Pick a key and a model on the left first"
    : !hasContent
    ? "Type a message, or attach something, before sending"
    : "";
}

function updateVoiceSendEnabled() {
  const btn = el("pg-voice-send");
  const hasContent = Boolean(el("pg-voice-prompt").value.trim());
  btn.disabled = !hasContent;
  btn.title = hasContent ? "" : "Type a line before sending";
}

// Typing, attaching, and picking a model/key all go through native
// input/change events that bubble to the document, so one delegated pair
// here covers those cases without a listener at each control. The handful
// of paths that mutate state programmatically (a finished recording, a key
// swap gating an attachment off, Clear) call updateSendEnabled() directly
// where that happens, since no native event fires for those.
document.addEventListener("input", (event) => {
  if (!event.target.closest("#playground")) return;
  updateSendEnabled();
  updateVoiceSendEnabled();
});
document.addEventListener("change", (event) => {
  if (!event.target.closest("#playground")) return;
  updateSendEnabled();
  updateVoiceSendEnabled();
});

/** Detach everything after a turn goes out. Each attachment kind clears
 *  its own picker, status line, and toggle, so the composer returns to the
 *  state it had before anything was attached. */
function clearAttachments() {
  ATTACHMENTS.forEach(({ id }) => {
    if (!PG_MEDIA[id] && !el(`pg-${id}-on`).checked) return;
    PG_MEDIA[id] = null;
    el(`pg-${id}-file`).value = "";
    if (el(`pg-${id}-url`)) el(`pg-${id}-url`).value = "";
    el(`pg-${id}-status`).textContent = "";
    el(`pg-${id}-on`).checked = false;
    el(`pg-${id}-panel`).hidden = true;
  });
  clearRecording();
  updateSendEnabled();
}

// Disable an attachment the selected key can never satisfy, and say which
// providers can. Discovering this after a round trip wastes a call and
// reads as a bug; the control should be plainly unavailable instead.
function gateAttachments(off = null) {
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  ATTACHMENTS.forEach(({ id, noun }) => {
    const accepts = target && target.accepts ? target.accepts[id] : null;
    const ok = !target || !accepts || accepts.bytes || accepts.url;
    const toggle = el(`pg-${id}-on`);
    if (!ok && toggle.checked && off) off.push(`send a ${noun}`);
    toggle.disabled = !ok;
    if (!ok) {
      toggle.checked = false;
      el(`pg-${id}-panel`).hidden = true;
      PG_MEDIA[id] = null;
    }
    el(`pg-${id}-toggle`).classList.toggle("pg-toggle-off", !ok);
    if (id === "audio") {
      // The microphone lives in the composer, away from this panel, so it
      // needs the same gate: offering to record into a key that cannot send
      // audio wastes the recording and the round trip.
      const mic = el("pg-mic");
      mic.disabled = !ok;
      // Say where to go, not only what's missing: a disabled control with
      // no route forward is the thing people get stuck on.
      const others = providersAccepting("audio");
      mic.title = ok
        ? "Record from your microphone"
        : `This ${target.provider} key can't send a recording. ` +
          (others.length
            ? keyPhrase("Pick", "in the Key list", others)
            : keyPhrase("Load", "", capableProviders("audio")));
    }
    // Say why in the page, not only in a tooltip: a control that is greyed
    // out with no reason reads as broken.
    const note = el(`pg-${id}-unavailable`);
    note.hidden = ok;
    if (!ok) {
      const others = providersAccepting(id);
      // Worded to keep an article away from a provider id, which gives
      // "a openai key" as often as not.
      note.textContent =
        `This ${target.provider} key can't send a ${noun}. ` +
        (others.length
          ? keyPhrase("Pick", "above", others)
          : `No key you've loaded can. ` + keyPhrase("Load", "", capableProviders(id)));
    }
  });
  updateSendEnabled();
}

// Every provider that takes this kind, loaded or not, straight from the
// catalog the adapters gate on.
function capableProviders(id) {
  return (PROVIDERS_ACCEPTING && PROVIDERS_ACCEPTING[id]) || [];
}

// "Pick a gemini key above." for one, a trailing list for several. Provider
// ids are lowercase and begin with any letter, so a sentence that would
// need "a" or "an" before a list gets rephrased rather than guessed at, and
// the list always comes last so nothing dangles after it.
// Transient notice for a state change the page made on the user's behalf,
// such as turning a toggle off because the new key cannot honour it.
let TOAST_TIMER = null;
function showToast(text) {
  let toast = el("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    el("pg-composer").appendChild(toast);
  }
  toast.textContent = text;
  toast.classList.add("show");
  clearTimeout(TOAST_TIMER);
  TOAST_TIMER = setTimeout(() => toast.classList.remove("show"), 4000);
}

// Providers whose capability flag for `cap` is on, straight from the
// catalog map the server sends.
function providersAble(cap) {
  return Object.keys(PROVIDER_CAPABILITIES)
    .filter((p) => PROVIDER_CAPABILITIES[p][cap])
    .sort();
}

// Same contract as gateAttachments, for the capability toggles: a key
// switch mid-conversation must not leave anything switched on that the
// new provider will refuse after a billable round trip. Controls the new
// key cannot honour are disabled with an inline note; controls the new
// key can honour come back. `off` collects what was force-disabled while
// switched on, for one toast naming everything at once.
function gateCapabilities(off) {
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  const caps = target ? PROVIDER_CAPABILITIES[target.provider] : null;

  const gateToggle = (checkboxId, wrapId, noteId, cap, noun) => {
    const ok = !target || !caps || Boolean(caps[cap]);
    const toggle = el(checkboxId);
    const wasOn = toggle.checked;
    toggle.disabled = !ok;
    el(wrapId).classList.toggle("pg-toggle-off", !ok);
    const note = el(noteId);
    note.hidden = ok;
    if (!ok) {
      toggle.checked = false;
      const loaded = providersAble(cap).filter((p) =>
        TARGETS.some((t) => t.provider === p)
      );
      note.textContent = `This ${target.provider} key can't ${noun}. ` +
        (loaded.length
          ? keyPhrase("Pick", "above", loaded)
          : keyPhrase("Load", "", providersAble(cap)));
      if (wasOn) off.push(noun);
    }
  };

  gateToggle("pg-search", "pg-search-toggle", "pg-search-unavailable",
    "web_search", "search the web");
  gateToggle("pg-cache-system", "pg-cache-toggle", "pg-cache-unavailable",
    "prompt_caching", "cache standing instructions");
  gateToggle("pg-tools-on", "pg-tools-toggle", "pg-tools-unavailable",
    "tool_calling", "offer tools");
  if (el("pg-tools-on").disabled) el("pg-tools-panel").hidden = true;

  // Same contract, for the reasoning-effort select: an unsupported key
  // gets it disabled and reset rather than sending a value the provider
  // will refuse.
  const reasoningOk = !target || !caps || Boolean(caps.reasoning_effort);
  const reasoningSelect = el("pg-reasoning");
  const reasoningWasSet = reasoningSelect.value !== "";
  reasoningSelect.disabled = !reasoningOk;
  const reasoningNote = el("pg-reasoning-unavailable");
  // The note stays hidden whenever the control it explains is off screen:
  // a task mode that hides the reasoning row (picture, video, voice,
  // transcribe) would otherwise leave the hint dangling under nothing.
  reasoningNote.hidden = reasoningOk || el("pg-reasoning-row").hidden;
  if (!reasoningOk) {
    reasoningSelect.value = "";
    const loaded = providersAble("reasoning_effort").filter((p) =>
      TARGETS.some((t) => t.provider === p)
    );
    reasoningNote.textContent = `This ${target.provider} key can't control reasoning effort. ` +
      (loaded.length
        ? keyPhrase("Pick", "above", loaded)
        : keyPhrase("Load", "", providersAble("reasoning_effort")));
    if (reasoningWasSet) off.push("control reasoning effort");
  }

  // A capability gate only says whether reasoning_effort exists at all;
  // "minimal" is narrower than that; OpenAI is the only provider that
  // accepts it, so it stays selectable there and greys out everywhere
  // else instead of being sent to a provider that will refuse it.
  const minimalOption = [...reasoningSelect.options].find((o) => o.value === "minimal");
  const minimalOk = reasoningOk && (!target || target.provider === "openai");
  minimalOption.disabled = !minimalOk;
  if (!minimalOk && reasoningSelect.value === "minimal") {
    reasoningSelect.value = "";
    off.push("use minimal reasoning effort");
  }
  updateSuggestedBudget();

  // The task picker: picking a task rebuilds the Key list down to keys
  // that can serve it, so a task only greys out when no loaded key's
  // provider can serve it at all. The fallback to text covers the one way
  // a selected task can still lose its footing: the key file reloading
  // out from under it. Falling back also ends any voice session in
  // progress, since no key remains that could hold one.
  const anyKeyCan = (cap) =>
    !TARGETS.length ||
    TARGETS.some((t) => {
      const c = PROVIDER_CAPABILITIES[t.provider];
      return !c || Boolean(c[cap]);
    });
  const taskGate = (value, cap, noun) => {
    const ok = anyKeyCan(cap);
    const option = [...el("pg-mode").options].find((o) => o.value === value);
    option.disabled = !ok;
    if (!ok && el("pg-mode").value === value) {
      el("pg-mode").value = "text";
      el("pg-mode").dispatchEvent(new Event("change", { bubbles: true }));
      off.push(noun);
    }
  };
  taskGate("image", "image_generation", "make a picture");
  taskGate("video", "video_generation", "make a video");
  taskGate("voice", "realtime", "hold a voice conversation");
  taskGate("transcribe", "transcription", "transcribe speech");
}

// One pass over everything a key switch can invalidate, ending in a
// single toast when anything was turned off on the user's behalf.
function applyKeyGates() {
  const off = [];
  gateAttachments(off);
  gateCapabilities(off);
  if (off.length) {
    const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
    const who = target ? `this ${target.provider} key` : "this key";
    showToast(`Turned off for ${who}: ${off.join(", ")}.`);
  }
}

function keyPhrase(verb, where, names) {
  const tail = where ? ` ${where}` : "";
  if (!names.length) return `${verb} a key from a provider that can.`;
  if (names.length === 1) return `${verb} a ${names[0]} key${tail}.`;
  return `${verb} a key${tail} from any of these: ${names.join(", ")}.`;
}

function providersAccepting(id) {
  const names = TARGETS.filter(
    (t) => t.accepts && t.accepts[id] && (t.accepts[id].bytes || t.accepts[id].url)
  ).map((t) => t.provider);
  return [...new Set(names)];
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

// The server-assigned id of the conversation now open, or null for one that
// has not been saved yet (nothing sent, or cleared since the last save).
// The title is decided once, from the first prompt, and kept for every
// later save of the same conversation rather than drifting to whatever was
// typed most recently.
let PG_CONVERSATION_ID = null;
let PG_CONVERSATION_TITLE = null;

// Bumped whenever the open conversation changes (New chat, or opening one
// from History). A save is fire-and-forget, so one can finish after the
// user has already moved on; the epoch check keeps that late save from
// writing its id back over the conversation now open.
let PG_CONVERSATION_EPOCH = 0;

const PG_MODE_LABELS = { text: "Text", image: "Picture", video: "Video", voice: "Voice", transcribe: "Transcript" };

function deriveConversationTitle(promptText) {
  const text = (promptText || "").trim();
  if (text) return text.length > 48 ? `${text.slice(0, 48)}…` : text;
  return `${PG_MODE_LABELS[currentMode()] || "New"} conversation`;
}

// Fire-and-forget: this is bookkeeping for the History pane, not something
// a reply should wait on. Saved server-side (not just a JS variable) so a
// conversation survives a page reload, cleared only when the server itself
// stops — the same lifetime the Playground's third pane promises.
//
// Serialized through a chain: transcription saves after every final
// transcript, and two saves in flight together could each see no saved id
// yet and file the same conversation twice. The chain makes each save read
// the id the one before it was assigned.
let PG_SAVE_CHAIN = Promise.resolve();
function saveCurrentConversation(latestPrompt) {
  PG_SAVE_CHAIN = PG_SAVE_CHAIN.then(() => saveConversationSnapshot(latestPrompt));
}

async function saveConversationSnapshot(latestPrompt) {
  const transcript = el("pg-transcript");
  if (!transcript.querySelector(".bubble")) return;
  if (!PG_CONVERSATION_TITLE) PG_CONVERSATION_TITLE = deriveConversationTitle(latestPrompt);
  const targetValue = el("pg-target").value;
  const modelValue = el("pg-model").value;
  const epoch = PG_CONVERSATION_EPOCH;
  const data = await api("/api/conversations", {
    method: "POST",
    body: {
      id: PG_CONVERSATION_ID,
      title: PG_CONVERSATION_TITLE,
      mode: currentMode(),
      target: targetValue ? Number(targetValue) : null,
      model: modelValue || null,
      history: PG_HISTORY,
      transcript_html: transcript.innerHTML,
    },
  });
  if (data.error) return;
  if (epoch === PG_CONVERSATION_EPOCH) PG_CONVERSATION_ID = data.conversation.id;
  loadConversationList();
}

async function loadConversationList() {
  const data = await api("/api/conversations");
  if (data.error) return;
  renderHistoryList(data.conversations);
}

function renderHistoryList(conversations) {
  const list = el("pg-history-list");
  const empty = el("pg-history-empty");
  [...list.querySelectorAll(".pg-history-item")].forEach((item) => item.remove());
  empty.hidden = conversations.length > 0;
  // Nothing to confirm away when the list is already empty — disabled
  // rather than hidden, since Clear is a stable fixture of this pane, and
  // the tooltip says what to do instead of just refusing the click.
  const clearBtn = el("pg-history-clear");
  clearBtn.disabled = conversations.length === 0;
  clearBtn.title = conversations.length === 0
    ? "Nothing saved yet — send a message to start a conversation"
    : "Delete every saved conversation";
  conversations.forEach((conversation) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "pg-history-item";
    if (conversation.id === PG_CONVERSATION_ID) item.classList.add("active");
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = conversation.title;
    const meta = document.createElement("div");
    meta.className = "meta";
    const when = new Date(conversation.updated_at * 1000);
    meta.textContent = `${PG_MODE_LABELS[conversation.mode] || conversation.mode} · ${when.toLocaleString()}`;
    item.appendChild(title);
    item.appendChild(meta);
    item.addEventListener("click", () => openConversation(conversation.id));
    list.appendChild(item);
  });
}

async function openConversation(id) {
  if (id === PG_CONVERSATION_ID) return;
  const data = await api(`/api/conversations?id=${id}`);
  if (data.error) return;
  const conversation = data.conversation;
  PG_CONVERSATION_EPOCH += 1;
  PG_CONVERSATION_ID = conversation.id;
  PG_CONVERSATION_TITLE = conversation.title;
  PG_HISTORY = conversation.history || [];
  clear(el("pg-tool-calls"));
  el("pg-mode").value = conversation.mode;
  await applyMode();
  if (conversation.target != null && TARGETS.some((t) => t.id === conversation.target)) {
    el("pg-target").value = String(conversation.target);
    await loadPlaygroundModels();
    applyKeyGates();
    if (currentMode() === "video") syncVideoDuration(false);
  }
  if (
    conversation.model &&
    [...el("pg-model").options].some((o) => o.value === conversation.model)
  ) {
    el("pg-model").value = conversation.model;
  }
  el("pg-transcript").innerHTML = conversation.transcript_html || "";
  el("pg-new").hidden = false;
  el("pg-transcript").scrollTop = el("pg-transcript").scrollHeight;
  updateSendEnabled();
  loadConversationList();
}

el("pg-history-clear").addEventListener("click", async () => {
  const count = el("pg-history-list").querySelectorAll(".pg-history-item").length;
  if (!count) return;
  const noun = count === 1 ? "conversation" : "conversations";
  const ok = await confirmDialog({
    title: `Clear ${count} saved ${noun}?`,
    message: "Every saved conversation is removed from this session and can't be recovered.",
    confirmLabel: "Clear conversations",
  });
  if (!ok) return;
  const data = await api("/api/conversations/clear", { method: "POST", body: {} });
  if (data.error) return;
  // The conversation on screen right now, whether saved or not, no longer
  // has anything backing it once every saved conversation is gone — reset
  // to a fresh chat rather than leaving its content still on display.
  startNewConversation();
});

el("pg-tools-on").addEventListener("change", () => {
  el("pg-tools-panel").hidden = !el("pg-tools-on").checked;
  updateSuggestedBudget();
});

el("pg-search").addEventListener("change", updateSuggestedBudget);
el("pg-reasoning").addEventListener("change", updateSuggestedBudget);

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
    // The model's turn, then ours, both replayed on the next request. The
    // model may have said something alongside its calls, and that text is
    // part of the same turn.
    const assistantParts = result.text ? [{ kind: "text", text: result.text }, ...calls] : calls;
    PG_HISTORY.push({ role: "assistant", parts: assistantParts });
    PG_HISTORY.push({ role: "user", parts: results });
    clear(panel);
    runGeneration({ continuation: true });
  });
  card.appendChild(send);
  panel.appendChild(card);
}

// Every settled exchange goes into PG_HISTORY so the conversation is
// replayed with the next request, whichever model or key answers it.
function recordExchange({ prompt, labels, data, continuation }) {
  if (data.error) return;
  if (!continuation) {
    const parts = [];
    if (prompt) parts.push({ kind: "text", text: prompt });
    // Media is not replayed on later turns (each replay is billed again),
    // so a short label stands in for what was attached.
    (labels || []).forEach((label) => {
      parts.push({ kind: "text", text: `[attached earlier: ${label}]` });
    });
    if (parts.length) PG_HISTORY.push({ role: "user", parts });
  }
  if (data.tool_calls && data.tool_calls.length) {
    // The tool panel records this assistant turn together with the results
    // when they are sent back; recording it here too would duplicate it.
    saveCurrentConversation(prompt);
    return;
  }
  if (data.text) {
    PG_HISTORY.push({ role: "assistant", parts: [{ kind: "text", text: data.text }] });
  }
  saveCurrentConversation(prompt);
}

el("pg-run").addEventListener("click", () => runGeneration({ continuation: false }));

// Shared by New chat and by Clear (once every saved conversation is gone,
// whatever's on screen no longer has a saved record behind it either).
function startNewConversation() {
  // A session in progress belongs to the conversation being left behind;
  // ending it here also snapshots and saves its transcript before the
  // clear below wipes the screen. No-ops when no session is open.
  endVoiceSession();
  endTranscribeSession();
  PG_CONVERSATION_EPOCH += 1;
  PG_HISTORY = [];
  // Every exchange up to now is already saved (saveCurrentConversation runs
  // after each one), so this just opens a fresh, as-yet-unsaved slot rather
  // than needing one last save on the way out.
  PG_CONVERSATION_ID = null;
  PG_CONVERSATION_TITLE = null;
  clear(el("pg-tool-calls"));
  transcriptEmpty();
  // A new conversation gets a fresh suggestion too, in case the last one
  // was hand-edited for a reply that's now behind it.
  MAXTOK_TOUCHED = false;
  updateSuggestedBudget();
  updateSendEnabled();
  loadConversationList();
}

el("pg-new").addEventListener("click", startNewConversation);

el("pg-prompt").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  // Ctrl/Cmd+Enter always sends. Plain Enter also sends when the box is
  // empty and something is attached: after recording there is no text to
  // break onto a new line, and reaching for a modifier to send a voice
  // message is a step nobody expects.
  const bare =
    !event.metaKey &&
    !event.ctrlKey &&
    !el("pg-prompt").value.trim() &&
    attachmentsFromInput().labels.length > 0;
  if (event.metaKey || event.ctrlKey || bare) {
    event.preventDefault();
    if (!el("pg-run").disabled) runGeneration({ continuation: false });
  }
});

async function runGeneration({ continuation }) {
  const btn = el("pg-run");
  const model = el("pg-model").value;
  const prompt = el("pg-prompt").value.trim();
  const attached = attachmentsFromInput();
  const hasAttachment = attached.labels.length > 0;
  if (!model) {
    // Send is disabled whenever a model isn't picked, so this is only
    // reachable from "Send results" on a tool-call card left over from
    // before the model was cleared.
    const note = addBubble("model");
    note.textContent = "Pick a key and a model on the left first.";
    return;
  }
  if (!prompt && !continuation && !hasAttachment) return;
  if (!continuation) {
    // A fresh send abandons any unanswered tool calls: their turn was never
    // recorded, so history stays free of calls with no results.
    clear(el("pg-tool-calls"));
  }
  let tooling;
  try {
    tooling = toolsFromInput();
  } catch (err) {
    const note = addBubble("model");
    renderGeneration(note, { error: { code: "bad_request", message: err.message } });
    return;
  }
  PG_LAST_REQUEST = { targetId: el("pg-target").value, modelId: model };
  working(
    btn,
    currentMode() === "image" ? "Drawing…" : currentMode() === "video" ? "Rendering…" : "Sending…"
  );
  if (!continuation) {
    addUserTurn(prompt, attached.labels);
    // The turn is on screen now, so leaving the text in the box invites
    // sending it twice.
    el("pg-prompt").value = "";
    // Same reasoning for what was attached: it belongs to the turn that
    // has just been sent, and the transcript holds its own playable copy.
    // Leaving it here would silently attach it to the next turn as well.
    clearAttachments();
  }
  if (currentMode() === "image") {
    const placeholder = addBubble("model");
    // Same ticking clock the text stream shows: a draw can run past a
    // minute, and a static line gives no sense that anything is moving.
    const startedAt = Date.now();
    const paint = () => {
      placeholder.textContent =
        `Drawing. This usually takes longer than text… · ${formatElapsed(startedAt)}`;
    };
    paint();
    const ticker = setInterval(paint, 1000);
    const data = await api("/api/generate/image", {
      method: "POST",
      body: { target: Number(el("pg-target").value), model, prompt },
    });
    clearInterval(ticker);
    placeholder.remove();
    if (data.error) {
      renderGeneration(addBubble("model"), data);
    } else {
      addImageBubble(data);
      saveCurrentConversation(prompt);
    }
    done(btn);
    updateSendEnabled();
    return;
  }
  if (currentMode() === "video") {
    const placeholder = addBubble("model");
    // Video renders run far longer than pictures (minutes, not seconds),
    // so the copy sets that expectation instead of implying something is
    // stuck.
    const startedAt = Date.now();
    const paint = () => {
      placeholder.textContent =
        `Rendering the video. This can take several minutes… · ${formatElapsed(startedAt)}`;
    };
    paint();
    const ticker = setInterval(paint, 1000);
    const data = await api("/api/generate/video", {
      method: "POST",
      body: {
        target: Number(el("pg-target").value),
        model,
        prompt,
        duration_seconds: Number(el("pg-video-duration").value),
      },
    });
    clearInterval(ticker);
    placeholder.remove();
    if (data.error) {
      renderGeneration(addBubble("model"), data);
    } else {
      addVideoBubble(data);
      saveCurrentConversation(prompt);
    }
    done(btn);
    updateSendEnabled();
    return;
  }
  const out = addBubble("model");
  out.textContent = "Waiting for the model…";
  const body = {
    target: Number(el("pg-target").value),
    model,
    prompt,
    system: el("pg-system").value.trim() || undefined,
    cache_system: el("pg-cache-system").checked,
    max_output_tokens: Number(el("pg-maxtok").value) || undefined,
    reasoning_effort: el("pg-reasoning").value || undefined,
    web_search: el("pg-search").checked,
    tools: tooling.tools,
    tool_choice: tooling.choice,
    ...attached.body,
    history: PG_HISTORY.length ? PG_HISTORY : undefined,
  };
  try {
    const data = await streamGeneration(out, body);
    if (data) recordExchange({ prompt, labels: attached.labels, data, continuation });
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
      recordExchange({ prompt, labels: attached.labels, data, continuation });
    }
  }
  done(btn);
  updateSendEnabled();
}

// A finished duration reads in seconds once it takes a second: "75.65 s"
// says slow at a glance where "75649 ms" needs arithmetic.
function formatDuration(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function formatElapsed(startedAt) {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  const m = Math.floor(s / 60);
  return m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

async function streamGeneration(out, body) {
  // One status line, always with the elapsed time: a web search can run
  // half a minute before the first token, and a bare "Waiting…" gives no
  // sense of whether anything is happening.
  const startedAt = Date.now();
  let phase = "Waiting for the model…";
  let meta = null;
  const paint = () => {
    if (!phase) return;
    const label = `${phase} · ${formatElapsed(startedAt)}`;
    if (meta) meta.textContent = label;
    else out.textContent = label;
  };
  const setPhase = (next) => {
    phase = next;
    paint();
  };
  const ticker = setInterval(paint, 1000);
  paint();

  try {
    let res;
    try {
      res = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
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
    meta = document.createElement("div");
    meta.className = "meta";
    card.appendChild(meta);
    out.appendChild(card);
    setPhase("streaming…");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawDelta = false;
    let settled = false;
    let outcome = null;
    let reasoningChars = 0;
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
            phase = null;
          } else if (event.kind === "text_delta") {
            if (!sawDelta) setPhase("streaming…");
            sawDelta = true;
            // Append a text node per delta: textContent += would re-copy the
            // whole accumulated string on every token.
            text.appendChild(document.createTextNode(event.text));
          } else if (event.kind === "reasoning_delta") {
            // A reasoning model can think for a long stretch before its
            // first visible token; show that as progress, not silence.
            reasoningChars += event.chars || 0;
            if (!sawDelta) {
              setPhase(`thinking… ${reasoningChars.toLocaleString()} characters of reasoning so far`);
            }
          } else if (event.kind === "provider_event") {
            // Bounded provider activity kinds relayed by the server. A
            // server-side search is the one to name while the answer has
            // not started.
            if (!sawDelta && /web_search/.test(event.what || "")) {
              setPhase("searching the web…");
            }
          } else if (event.kind === "tool_call_started") {
            // Named before the arguments finish parsing, so show the call is
            // coming without implying it can be acted on yet.
            setPhase(`calling ${event.name}…`);
          } else if (event.kind === "result") {
            renderGeneration(out, event);
            renderToolCalls(event);
            settled = true;
            outcome = event;
            phase = null;
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
    // The settled result, or null when the server reported an error: the
    // caller records successful exchanges into the conversation history.
    return outcome;
  } finally {
    clearInterval(ticker);
  }
}

function usageLabel(usage) {
  // Providers report different fields: a missing total is not a missing
  // count, so fall back to the parts rather than claiming nothing arrived.
  if (!usage) return "usage unreported";
  const reasoning = usage.reasoning_tokens != null ? ` (${usage.reasoning_tokens} reasoning)` : "";
  if (usage.total_tokens != null) return `${usage.total_tokens} tokens${reasoning}`;
  const parts = [];
  if (usage.input_tokens != null) parts.push(`${usage.input_tokens} in`);
  if (usage.output_tokens != null) parts.push(`${usage.output_tokens} out`);
  return parts.length ? `${parts.join(" / ")} tokens${reasoning}` : "usage unreported";
}

function renderGeneration(out, data) {
  clear(out);
  if (data.error) {
    const box = document.createElement("div");
    box.className = "err-box";
    box.textContent = `${data.error.code}: ${data.error.message}`;
    out.appendChild(box);
    // Which model was asked is captured when the request goes out, not read
    // from the picker now: a reply can land after the selection moved on,
    // and condemning whatever happens to be selected would be wrong.
    if (PG_LAST_REQUEST) {
      noteModelOutcome(PG_LAST_REQUEST.targetId, PG_LAST_REQUEST.modelId, data.error.code);
    }
    return;
  }
  const card = document.createElement("div");
  card.className = "card";
  const text = document.createElement("div");
  text.className = "result-text";
  if (data.text) {
    // location.href is the base for resolving a relative link; passing it
    // in keeps the renderer itself free of globals and testable.
    text.appendChild(renderMarkdown(data.text, document, location.href));
  } else {
    text.textContent = "(no text output)";
  }
  card.appendChild(text);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent =
    `${data.model} · ${formatDuration(data.round_trip_duration_ms)} · ${usageLabel(data.usage)}` +
    (data.finish_reason ? ` · ${data.finish_reason}` : "");
  card.appendChild(meta);

  // Warnings the library attached, chief among them a reply cut off by the
  // token budget. These were only rendered on the streaming path, so the
  // plain path showed a truncated answer with nothing but a finish reason
  // in small grey text to explain it.
  (data.warnings || []).forEach((warning) => {
    const note = document.createElement("div");
    note.className = "notice";
    note.textContent = warning;
    card.appendChild(note);
  });

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
      line.textContent = `✓ ${a.model_id} (pos ${a.position}) — ${formatDuration(a.round_trip_duration_ms)}, total tokens: ${tokens}`;
    } else {
      line.textContent = `✗ ${a.model_id} (pos ${a.position}) — ${a.error_code}: ${a.error_message}`;
    }
    card.appendChild(line);
  });
  return card;
}

// --- traces -----------------------------------------------------------------

const TRACE_ROUTE_LABELS = {
  "/api/generate/stream": "Streamed text",
  "/api/generate": "Text",
  "/api/generate/image": "Picture",
  "/api/generate/video": "Video",
  "/api/verify": "Verify",
  "/api/models": "Model list",
};

let tracesTimer = null;

function traceTargetName(id) {
  const t = (TARGETS || []).find((entry) => entry.id === id);
  if (t) return `${t.name} (${t.provider})`;
  return id != null ? `#${id}` : "—";
}


// The rows as last fetched, so search and sort re-render locally without
// another request, and the 2-second auto-refresh can't fight a half-typed
// search or reset a chosen order.
let TRACE_ROWS = [];
// Time sorts by id (true request order), Took by the raw milliseconds:
// both would sort wrongly as their display strings ("9:5" after "18:3",
// "901 ms" after "60.05 s").
let TRACE_SORT = { key: "id", dir: "desc" };

function traceDisplay(r) {
  let outcome = r.status;
  if (r.events != null) outcome += ` · ${r.events} event(s)`;
  if (r.detail) outcome += ` — ${r.detail}`;
  return {
    id: r.id,
    duration_ms: r.duration_ms,
    status: r.status,
    at: r.at,
    what: TRACE_ROUTE_LABELS[r.route] || r.route,
    key: traceTargetName(r.target),
    model: r.model || "—",
    took: formatDuration(r.duration_ms),
    outcome,
  };
}

function renderTraces() {
  const query = el("traces-search").value.trim().toLowerCase();
  let rows = TRACE_ROWS.map(traceDisplay);
  if (query) {
    rows = rows.filter((d) =>
      [d.at, d.what, d.key, d.model, d.took, d.outcome].some((v) =>
        String(v).toLowerCase().includes(query)
      )
    );
  }
  const { key, dir } = TRACE_SORT;
  const numeric = key === "id" || key === "duration_ms";
  rows.sort((a, b) => {
    const cmp = numeric ? a[key] - b[key] : String(a[key]).localeCompare(String(b[key]));
    return dir === "asc" ? cmp : -cmp;
  });

  document.querySelectorAll("#traces-table th").forEach((th) => {
    th.classList.toggle("sorted-asc", th.dataset.sort === key && dir === "asc");
    th.classList.toggle("sorted-desc", th.dataset.sort === key && dir === "desc");
  });

  el("traces-status").textContent =
    TRACE_ROWS.length === 0
      ? "Nothing yet. Use the Playground or Verify, then look back here."
      : rows.length === 0
        ? "No trace matches that search."
        : "";
  const tbody = document.querySelector("#traces-table tbody");
  clear(tbody);
  rows.forEach((d) => {
    const tr = document.createElement("tr");
    [d.at, d.what, d.key, d.model, d.took, d.outcome].forEach((text, i) => {
      const td = document.createElement("td");
      td.textContent = String(text);
      if (i === 5 && d.status !== "ok") td.className = "fail";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

async function loadTraces() {
  const data = await api("/api/traces");
  if (data.error) {
    // api() wraps every failure; a JSON parse failure here means the
    // route answered HTML — an older server with no traces endpoint.
    // Static files reload per request, the Python process doesn't.
    if (data.error.code === "request_failed" && /JSON/i.test(data.error.message)) {
      el("traces-status").textContent =
        "This viewer's server is older than the page and has no traces " +
        "endpoint yet — stop keycall view and start it again.";
      stopTracesTimer();
    } else {
      el("traces-status").textContent = `${data.error.code}: ${data.error.message}`;
    }
    return;
  }
  TRACE_ROWS = data.traces || [];
  // Nothing to confirm away when there are no traces at all, regardless of
  // what the current search filters down to.
  const clearBtn = el("traces-clear");
  clearBtn.disabled = TRACE_ROWS.length === 0;
  clearBtn.title = TRACE_ROWS.length === 0
    ? "No traces recorded yet"
    : "Delete every recorded trace";
  renderTraces();
}

function stopTracesTimer() {
  if (tracesTimer) {
    clearInterval(tracesTimer);
    tracesTimer = null;
  }
}

function startTracesTimer() {
  stopTracesTimer();
  if (el("traces-auto").checked && el("traces").classList.contains("active")) {
    tracesTimer = setInterval(loadTraces, 2000);
  }
}

el("traces-refresh").addEventListener("click", loadTraces);
el("traces-search").addEventListener("input", renderTraces);
document.querySelectorAll("#traces-table th").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (!key) return;
    if (TRACE_SORT.key === key) {
      TRACE_SORT.dir = TRACE_SORT.dir === "asc" ? "desc" : "asc";
    } else {
      // A fresh column starts with its natural reading: newest first for
      // Time, ascending for everything else.
      TRACE_SORT = { key, dir: key === "id" ? "desc" : "asc" };
    }
    renderTraces();
  });
});
el("traces-clear").addEventListener("click", async () => {
  if (!TRACE_ROWS.length) return;
  const noun = TRACE_ROWS.length === 1 ? "trace" : "traces";
  const ok = await confirmDialog({
    title: `Clear ${TRACE_ROWS.length} ${noun}?`,
    message: "Every recorded request in this run is removed and can't be recovered.",
    confirmLabel: "Clear traces",
  });
  if (!ok) return;
  const data = await api("/api/traces/clear", { method: "POST", body: {} });
  if (data.error) {
    el("traces-status").textContent = `${data.error.code}: ${data.error.message}`;
    return;
  }
  loadTraces();
});
el("traces-auto").addEventListener("change", startTracesTimer);

boot();
