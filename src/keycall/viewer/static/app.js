// KeyCall viewer frontend. Vanilla ES module, no dependencies (the CSP
// blocks external anything). The auth token arrives once in the page URL's
// ?token=; we read it, then strip it from the address bar so it doesn't
// linger in history, and send it as a header on every /api/* call.

const params = new URLSearchParams(location.search);
const TOKEN = params.get("token") || "";
if (params.has("token")) {
  history.replaceState(null, "", location.pathname);
}

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

async function boot() {
  const health = await api("/api/health");
  el("health").textContent = `keycall ${health.version} · ${health.targets} target(s)`;

  const data = await api("/api/targets");
  TARGETS = data.targets || [];

  renderDashboard();
  fillTargetSelects();
  attachSort(el("dashboard-table"));
  attachSort(el("models-table"));
  if (TARGETS.length) {
    await loadModels();          // fills the cache…
    await loadPlaygroundModels(); // …which this then reuses instantly
  }
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
  const data = await api(`/api/models?target=${id}&category=text_generation`);
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
}

// --- models browser ---------------------------------------------------------

async function loadModels(refresh = false) {
  const id = el("models-target").value;
  const category = el("models-category").value;
  if (id === "") return;
  el("models-status").textContent = "loading…";
  const q = new URLSearchParams({ target: id });
  if (category) q.set("category", category);
  if (refresh) q.set("refresh", "1");
  const data = await api(`/api/models?${q}`);
  const tbody = el("models-table").querySelector("tbody");
  clear(tbody);
  if (data.error) {
    el("models-status").textContent = `${data.error.code}: ${data.error.message}`;
    return;
  }
  el("models-status").textContent =
    `${data.models.length} model(s)${data.from_cache ? " (cached)" : ""} · catalog ${data.catalog_version}`;
  populateCategoryOptions(data);
  // Source only earns its column when it varies (e.g. Gemini mixes
  // provider_metadata with keycall_rule); a constant column is noise.
  const sources = new Set(data.models.map((m) => m.classification_source));
  el("models-table").classList.toggle("hide-source", sources.size <= 1);
  data.models.forEach((m) => {
    const row = document.createElement("tr");
    row.appendChild(td(m.id));
    row.appendChild(td(m.categories.join(", ")));
    row.appendChild(td(m.classification_source));
    row.appendChild(td(m.context_limit ? String(m.context_limit) : "—", "num"));
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
    opt.textContent = c;
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

el("pg-run").addEventListener("click", async () => {
  const btn = el("pg-run");
  const out = el("pg-result");
  const model = el("pg-model").value;
  const prompt = el("pg-prompt").value.trim();
  if (!model || !prompt) {
    out.textContent = "pick a model and enter a prompt";
    return;
  }
  btn.disabled = true;
  clear(out);
  out.textContent = "generating…";
  const data = await api("/api/generate", {
    method: "POST",
    body: {
      target: Number(el("pg-target").value),
      model,
      prompt,
      system: el("pg-system").value.trim() || undefined,
      max_output_tokens: Number(el("pg-maxtok").value) || undefined,
      web_search: el("pg-search").checked,
    },
  });
  btn.disabled = false;
  renderGeneration(out, data);
});

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
  const tokens = data.usage && data.usage.total_tokens != null ? data.usage.total_tokens : "unreported";
  meta.textContent =
    `${data.model} · ${Math.round(data.round_trip_duration_ms)} ms · ${tokens} tokens` +
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
  btn.disabled = true;
  clear(results);
  el("verify-status").textContent = "running…";

  for (const t of TARGETS) {
    const data = await api("/api/verify", {
      method: "POST",
      body: { target: t.id, generate },
    });
    results.appendChild(renderVerify(t, data));
  }
  el("verify-status").textContent = "done";
  btn.disabled = false;
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
      const tokens = a.total_tokens != null ? a.total_tokens : "unreported";
      line.textContent = `✓ ${a.model_id} (pos ${a.position}) — ${Math.round(a.round_trip_duration_ms)} ms, ${tokens} tokens`;
    } else {
      line.textContent = `✗ ${a.model_id} (pos ${a.position}) — ${a.error_code}: ${a.error_message}`;
    }
    card.appendChild(line);
  });
  return card;
}

boot();
