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
// {image: [...providers], audio: [...], file: [...]}, from the catalog.
let PROVIDERS_ACCEPTING = {};

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
    // The Playground can only be measured once it is on screen; a hidden
    // tab has no position to measure from.
    if (btn.dataset.tab === "playground") sizePlayground();
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
  PROVIDERS_ACCEPTING = data.providers_accepting || {};
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
  el("dash-models-header").title =
    "Every model this key can reach, of every kind. The Models tab breaks them down.";
  transcriptEmpty();
  sizePlayground();
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
    row.addEventListener("click", () => checkTarget(t.id, row));
    tbody.appendChild(row);
  });
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
  gateAttachments();
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
  const category = currentMode() === "image" ? "image_generation" : "text_generation";
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
    return;
  }
  if (!data.models.length) {
    const none = document.createElement("option");
    none.textContent =
      currentMode() === "image"
        ? "this key has no picture models"
        : "this key has no text models";
    sel.appendChild(none);
    return;
  }
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
  // What a key can accept changes with the key, so re-gate before the user
  // reaches for a control the new one cannot honour.
  gateAttachments();
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

function applyMode() {
  const image = currentMode() === "image";
  el("pg-extras").hidden = image;
  el("pg-maxtok-row").hidden = image;
  el("pg-system-row").hidden = image;
  el("pg-image-mode-note").hidden = !image;
  // An image model takes a description and nothing else, so a microphone
  // in the composer would only offer something that cannot be sent.
  el("pg-mic").hidden = image;
  if (image && REC) discardRecording();
  el("pg-prompt").placeholder = image
    ? "Describe the picture you want. Press Send, or Ctrl+Enter."
    : "Ask anything. Press Send, or Ctrl+Enter.";
  loadPlaygroundModels();
}

el("pg-mode").addEventListener("change", applyMode);

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
  meta.textContent =
    `${result.model} · ${Math.round(result.round_trip_duration_ms)} ms · ` +
    usageLabel(result.usage);
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
}

function clearTranscriptPlaceholder() {
  const placeholder = el("pg-transcript").querySelector(".pg-empty");
  if (placeholder) placeholder.remove();
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
        ? "your browser blocked microphone access — allow it for this page and try again"
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
}

// Disable an attachment the selected key can never satisfy, and say which
// providers can. Discovering this after a round trip wastes a call and
// reads as a bug; the control should be plainly unavailable instead.
function gateAttachments() {
  const target = TARGETS.find((t) => String(t.id) === el("pg-target").value);
  ATTACHMENTS.forEach(({ id, noun }) => {
    const accepts = target && target.accepts ? target.accepts[id] : null;
    const ok = !target || !accepts || accepts.bytes || accepts.url;
    const toggle = el(`pg-${id}-on`);
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
  if (!model || (!prompt && !continuation && !hasAttachment)) {
    // Say which half is missing rather than restating both.
    const note = addBubble("model");
    note.textContent = model
      ? "Type a message below, or attach something, then press Send."
      : "Pick a key and a model on the left first.";
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
    const note = addBubble("model");
    renderGeneration(note, { error: { code: "bad_request", message: err.message } });
    return;
  }
  PG_LAST_REQUEST = { targetId: el("pg-target").value, modelId: model };
  working(btn, currentMode() === "image" ? "Drawing…" : "Sending…");
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
    placeholder.textContent = "Drawing. This usually takes longer than text…";
    const data = await api("/api/generate/image", {
      method: "POST",
      body: { target: Number(el("pg-target").value), model, prompt },
    });
    placeholder.remove();
    if (data.error) {
      renderGeneration(addBubble("model"), data);
    } else {
      addImageBubble(data);
    }
    done(btn);
    return;
  }
  const out = addBubble("model");
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
    ...attached.body,
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
