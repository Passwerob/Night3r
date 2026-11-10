const state = {
  items: [],
  decisions: {},
  counts: { keep: 0, skip: 0, unreviewed: 0 },
  filtered: [],
  index: 0,
  dataset: "all",
  status: "all",
  search: "",
  sort: "sequence",
  mode: "review",
  geometryView: "elevated",
  busy: false,
  noteTimer: null,
  queueSignature: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const els = {
  connection: $("#connectionState"),
  progress: $("#reviewProgress"),
  counts: $("#reviewCounts"),
  progressFill: $("#progressFill"),
  title: $("#pairTitle"),
  position: $("#pairPosition"),
  datasetBadge: $("#datasetBadge"),
  splitBadge: $("#splitBadge"),
  stage: $("#visualStage"),
  loading: $("#loadingState"),
  geometryView: $("#geometryView"),
  decision: $("#decisionValue"),
  score: $("#scoreValue"),
  metrics: $("#metricTable"),
  diagnostics: $("#diagnosticList"),
  protocol: $("#protocolList"),
  warning: $("#warningBox"),
  note: $("#noteInput"),
  prev: $("#previousButton"),
  next: $("#nextButton"),
  undo: $("#undoButton"),
  toast: $("#toast"),
  candidateList: $("#candidateList"),
  queueSummary: $("#queueSummary"),
};

function currentItem() {
  return state.filtered[state.index] || null;
}

function decisionFor(id) {
  return state.decisions[id]?.decision || "unreviewed";
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: payload === undefined ? "GET" : "POST",
    headers: payload === undefined ? {} : { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    cache: "no-store",
  });
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

function absorbServerState(serverState) {
  if (serverState.manifest) state.items = serverState.manifest.items;
  state.decisions = serverState.decisions || {};
  state.counts = serverState.counts || state.counts;
  els.undo.disabled = !serverState.can_undo;
}

function setSegment(group, value) {
  $$(`#${group} button`).forEach((button) => button.classList.toggle("active", button.dataset.value === value));
}

function applyFilters(preferredId = null) {
  const query = state.search.trim().toLowerCase();
  let items = state.items.filter((item) => {
    if (state.dataset !== "all" && item.dataset !== state.dataset) return false;
    if (state.status !== "all" && decisionFor(item.id) !== state.status) return false;
    if (query && !`${item.scene} ${item.sample_id}`.toLowerCase().includes(query)) return false;
    return true;
  });
  if (state.sort === "score_desc") items.sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
  if (state.sort === "score_asc") items.sort((a, b) => (a.score ?? Infinity) - (b.score ?? Infinity));
  if (state.sort === "noise_asc") items.sort((a, b) => (a.lowlight_noise ?? Infinity) - (b.lowlight_noise ?? Infinity));
  if (state.sort === "dark_asc") items.sort((a, b) => (a.lowlight_mean ?? Infinity) - (b.lowlight_mean ?? Infinity));
  state.filtered = items;
  if (preferredId) {
    const nextIndex = items.findIndex((item) => item.id === preferredId);
    if (nextIndex >= 0) state.index = nextIndex;
  }
  state.index = Math.max(0, Math.min(state.index, Math.max(0, items.length - 1)));
  renderQueue();
  render();
}

function renderQueue() {
  els.queueSummary.textContent = `${state.filtered.length} shown`;
  els.candidateList.innerHTML = state.filtered.map((item, index) => {
    const decision = decisionFor(item.id);
    return `
      <button class="candidate-item ${decision}" type="button" data-index="${index}" aria-label="Open ${escapeHtml(item.sample_id)}">
        <img class="candidate-thumb" src="${escapeHtml(item.assets.pair.rgb1)}" alt="" loading="lazy">
        <span class="candidate-copy">
          <strong>${escapeHtml(item.scene)}</strong>
          <span>${escapeHtml(`${item.idx1}-${item.idx2} / rank ${item.scene_rank}`)}</span>
          <span>mean ${formatNumber(item.lowlight_mean)} / noise ${formatNumber(item.lowlight_noise, 4)}</span>
        </span>
        <span class="status-dot" aria-label="${decision}"></span>
      </button>`;
  }).join("");
  els.candidateList.querySelectorAll(".candidate-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.index = Number(button.dataset.index);
      render();
    });
  });
}

function updateQueueSelection() {
  const buttons = els.candidateList.querySelectorAll(".candidate-item");
  buttons.forEach((button, index) => button.classList.toggle("active", index === state.index));
  const active = buttons[state.index];
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function mediaPanel(title, subtitle, src, alt) {
  return `
    <article class="media-panel">
      <header><strong>${escapeHtml(title)}</strong><span>${escapeHtml(subtitle)}</span></header>
      <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="eager">
    </article>`;
}

function renderVisual(item) {
  els.stage.className = "visual-stage";
  if (state.mode === "review") {
    const view = state.geometryView;
    const mast = item.metrics.mast3r;
    const night = item.metrics.night3r;
    const metric = (values) => values.median_absrel === undefined ? "shared relative scale" : `AbsRel ${formatNumber(values.median_absrel)}`;
    els.stage.classList.add("grid-8");
    els.stage.innerHTML = [
      mediaPanel("Low-light RGB", "view 1", item.assets.pair.rgb1, `${item.sample_id} low-light RGB view 1`),
      mediaPanel("Low-light RGB", "view 2", item.assets.pair.rgb2, `${item.sample_id} low-light RGB view 2`),
      mediaPanel("Events", "view 1 / positive red / negative blue", item.assets.pair.event1, `${item.sample_id} event view 1`),
      mediaPanel("Events", "view 2 / positive red / negative blue", item.assets.pair.event2, `${item.sample_id} event view 2`),
      mediaPanel("MASt3R depth", metric(mast), item.assets.depth.mast3r, `${item.sample_id} MASt3R depth`),
      mediaPanel("Night3R depth", metric(night), item.assets.depth.night3r, `${item.sample_id} Night3R depth`),
      mediaPanel("MASt3R depth-derived 3D", `${view} / shared camera`, item.assets.geometry.mast3r[view], `${item.sample_id} MASt3R depth-derived geometry`),
      mediaPanel("Night3R depth-derived 3D", `${view} / shared camera`, item.assets.geometry.night3r[view], `${item.sample_id} Night3R depth-derived geometry`),
    ].join("");
  } else if (state.mode === "pair") {
    els.stage.classList.add("grid-4");
    els.stage.innerHTML = [
      mediaPanel("Low-light RGB", "view 1", item.assets.pair.rgb1, `${item.sample_id} low-light RGB view 1`),
      mediaPanel("Low-light RGB", "view 2", item.assets.pair.rgb2, `${item.sample_id} low-light RGB view 2`),
      mediaPanel("Events", "view 1 · positive red / negative blue", item.assets.pair.event1, `${item.sample_id} event view 1`),
      mediaPanel("Events", "view 2 · positive red / negative blue", item.assets.pair.event2, `${item.sample_id} event view 2`),
    ].join("");
  } else if (state.mode === "depth") {
    els.stage.classList.add("grid-4");
    const mast = item.metrics.mast3r;
    const night = item.metrics.night3r;
    const zero = item.metrics.zero;
    const metric = (values) => values.median_absrel === undefined ? "shared relative scale" : `AbsRel ${formatNumber(values.median_absrel)}`;
    els.stage.innerHTML = [
      mediaPanel("MASt3R", metric(mast), item.assets.depth.mast3r, `${item.sample_id} MASt3R depth`),
      mediaPanel("Night3R", metric(night), item.assets.depth.night3r, `${item.sample_id} Night3R depth`),
      mediaPanel("Night3R zero event", metric(zero), item.assets.depth.zero, `${item.sample_id} zero event depth`),
      mediaPanel(item.assets.depth.reference_label, item.has_gt ? "shared metric range" : "diagnostic only", item.assets.depth.reference, `${item.sample_id} depth reference`),
    ].join("");
  } else if (state.mode === "geometry") {
    const view = state.geometryView;
    const geometry = item.assets.geometry;
    const panels = [
      mediaPanel("MASt3R depth-derived 3D", `${view} · shared camera`, geometry.mast3r[view], `${item.sample_id} MASt3R ${view} depth-derived point cloud`),
      mediaPanel("Night3R depth-derived 3D", `${view} · shared camera`, geometry.night3r[view], `${item.sample_id} Night3R ${view} depth-derived point cloud`),
    ];
    if (item.has_gt && geometry.gt) panels.push(mediaPanel("GT depth-derived 3D", `${view} · shared camera`, geometry.gt[view], `${item.sample_id} GT ${view} depth-derived point cloud`));
    els.stage.classList.add(item.has_gt ? "grid-3" : "grid-2");
    els.stage.innerHTML = panels.join("");
  } else {
    els.stage.classList.add("single");
    els.stage.innerHTML = `<img class="overview-image" src="${escapeHtml(item.assets.overview)}" alt="${escapeHtml(item.sample_id)} overview">`;
  }
  els.geometryView.hidden = !["review", "geometry"].includes(state.mode);
  trackImages();
}

function trackImages() {
  const images = [...els.stage.querySelectorAll("img")];
  if (!images.length) return;
  let pending = images.filter((image) => !image.complete).length;
  els.loading.classList.toggle("visible", pending > 0);
  images.forEach((image) => {
    if (image.complete) return;
    const done = () => {
      pending -= 1;
      if (pending <= 0) els.loading.classList.remove("visible");
    };
    image.addEventListener("load", done, { once: true });
    image.addEventListener("error", done, { once: true });
  });
}

function renderMetrics(item) {
  if (item.has_gt) {
    const mast = item.metrics.mast3r;
    const night = item.metrics.night3r;
    els.metrics.innerHTML = `
      <div class="metric-row header"><span>Metric</span><span>MASt3R</span><span>Night3R</span></div>
      <div class="metric-row"><span>Raw AbsRel ↓</span><span>${formatNumber(mast.raw_absrel)}</span><span class="night">${formatNumber(night.raw_absrel)}</span></div>
      <div class="metric-row"><span>Raw δ1 ↑</span><span>${formatNumber(mast.raw_delta1)}</span><span class="night">${formatNumber(night.raw_delta1)}</span></div>
      <div class="metric-row"><span>Aligned AbsRel ↓</span><span>${formatNumber(mast.median_absrel)}</span><span class="night">${formatNumber(night.median_absrel)}</span></div>
      <div class="metric-row"><span>Aligned δ1 ↑</span><span>${formatNumber(mast.median_delta1)}</span><span class="night">${formatNumber(night.median_delta1)}</span></div>`;
  } else {
    const modelDiff = item.metrics.mast3r_vs_night3r || {};
    const eventDiff = item.metrics.night3r_real_vs_zero || {};
    els.metrics.innerHTML = `
      <div class="metric-row header"><span>Diagnostic</span><span>Mean</span><span>P90</span></div>
      <div class="metric-row"><span>Model log-depth diff</span><span>${formatNumber(modelDiff.mean_abs_log)}</span><span>${formatNumber(modelDiff.p90_abs_log)}</span></div>
      <div class="metric-row"><span>Event log-depth diff</span><span>${formatNumber(eventDiff.mean_abs_log)}</span><span>${formatNumber(eventDiff.p90_abs_log)}</span></div>`;
  }
}

function definitionRows(rows) {
  return rows.map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function renderInspector(item) {
  const decision = decisionFor(item.id);
  els.decision.textContent = decision[0].toUpperCase() + decision.slice(1);
  els.decision.className = decision === "unreviewed" ? "" : decision;
  els.score.textContent = `Score ${formatNumber(item.score)}`;
  renderMetrics(item);
  els.diagnostics.innerHTML = definitionRows([
    ["Event occupancy", formatNumber(item.event_occupancy)],
    ["Event activity", formatNumber(item.event_activity_mean)],
    ["BRG gate mean", formatNumber(item.event_gate_mean, 5)],
    ["Low-light mean", formatNumber(item.lowlight_mean)],
    ["Dark noise", formatNumber(item.lowlight_noise, 5)],
  ]);
  els.protocol.innerHTML = definitionRows([
    ["Source", item.protocol.geometry_source],
    ["Scale", item.protocol.scale_policy],
    ["Frame", "view-1 camera"],
    ["Camera", "shared per pair"],
    ["Point stride", item.protocol.point_stride],
  ]);
  els.note.value = state.decisions[item.id]?.note || "";
  if (item.warning) {
    els.warning.hidden = false;
    els.warning.textContent = item.warning;
  } else {
    els.warning.hidden = true;
    els.warning.textContent = "";
  }
}

function renderHeader(item) {
  const reviewed = state.counts.keep + state.counts.skip;
  const total = state.items.length || 1;
  els.progress.textContent = `${reviewed} / ${total} reviewed`;
  els.counts.textContent = `${state.counts.keep} kept · ${state.counts.skip} skipped`;
  els.progressFill.style.width = `${100 * reviewed / total}%`;
  els.title.textContent = item.sample_id;
  els.position.textContent = `${state.index + 1} of ${state.filtered.length} · scene rank ${item.scene_rank}`;
  els.datasetBadge.textContent = item.dataset;
  els.splitBadge.textContent = item.split_label;
  els.splitBadge.classList.toggle("warning", !item.split_label.startsWith("strict-heldout") || !item.has_gt);
  els.prev.disabled = state.index <= 0;
  els.next.disabled = state.index >= state.filtered.length - 1;
}

function render() {
  const item = currentItem();
  const reviewed = state.counts.keep + state.counts.skip;
  const total = state.items.length || 1;
  els.progress.textContent = `${reviewed} / ${total} reviewed`;
  els.counts.textContent = `${state.counts.keep} kept · ${state.counts.skip} skipped`;
  els.progressFill.style.width = `${100 * reviewed / total}%`;
  if (!item) {
    els.stage.className = "visual-stage single";
    els.stage.innerHTML = '<div class="empty-state"><strong>No candidates match the current filters.</strong></div>';
    els.title.textContent = "No matching candidates";
    els.position.textContent = "0 of 0";
    els.prev.disabled = true;
    els.next.disabled = true;
    updateQueueSelection();
    return;
  }
  renderHeader(item);
  renderVisual(item);
  renderInspector(item);
  updateQueueSelection();
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("visible"), 1300);
}

function navigate(delta) {
  if (!state.filtered.length) return;
  state.index = Math.max(0, Math.min(state.filtered.length - 1, state.index + delta));
  render();
}

async function decide(decision) {
  const item = currentItem();
  if (!item || state.busy) return;
  state.busy = true;
  const oldIndex = state.index;
  const nextId = state.filtered[oldIndex + 1]?.id || state.filtered[oldIndex - 1]?.id || null;
  try {
    const result = await api("/api/decision", { id: item.id, decision, note: els.note.value });
    absorbServerState(result);
    state.index = oldIndex;
    applyFilters(state.status === "all" ? nextId : null);
    showToast(decision === "keep" ? "Kept and archived" : decision === "skip" ? "Skipped" : "Decision cleared");
  } catch (error) {
    showToast(`Save failed: ${error.message}`);
  } finally {
    state.busy = false;
  }
}

async function undo() {
  if (state.busy) return;
  state.busy = true;
  const item = currentItem();
  try {
    const result = await api("/api/undo", {});
    absorbServerState(result);
    applyFilters(item?.id);
    showToast("Last decision restored");
  } catch (error) {
    showToast(`Undo failed: ${error.message}`);
  } finally {
    state.busy = false;
  }
}

async function saveNote(itemId, note) {
  if (!itemId || state.busy) return;
  try {
    const result = await api("/api/note", { id: itemId, note });
    absorbServerState(result);
    showToast("Note saved");
  } catch (error) {
    showToast(`Note failed: ${error.message}`);
  }
}

function setMode(mode) {
  state.mode = mode;
  setSegment("viewMode", mode);
  render();
}

function bindSegments(group, callback) {
  $$(`#${group} button`).forEach((button) => button.addEventListener("click", () => callback(button.dataset.value)));
}

function bindEvents() {
  bindSegments("datasetFilter", (value) => { state.dataset = value; state.index = 0; setSegment("datasetFilter", value); applyFilters(); });
  bindSegments("statusFilter", (value) => { state.status = value; state.index = 0; setSegment("statusFilter", value); applyFilters(); });
  bindSegments("viewMode", setMode);
  bindSegments("geometryView", (value) => { state.geometryView = value; setSegment("geometryView", value); render(); });
  $("#searchInput").addEventListener("input", (event) => { state.search = event.target.value; state.index = 0; applyFilters(); });
  $("#sortSelect").addEventListener("change", (event) => { state.sort = event.target.value; state.index = 0; applyFilters(); });
  els.prev.addEventListener("click", () => navigate(-1));
  els.next.addEventListener("click", () => navigate(1));
  $("#keepButton").addEventListener("click", () => decide("keep"));
  $("#skipButton").addEventListener("click", () => decide("skip"));
  $("#clearButton").addEventListener("click", () => decide(null));
  els.undo.addEventListener("click", undo);
  $("#fullscreenButton").addEventListener("click", () => {
    const target = $("#viewer");
    if (document.fullscreenElement) document.exitFullscreen(); else target.requestFullscreen();
  });
  els.note.addEventListener("input", () => {
    clearTimeout(state.noteTimer);
    const itemId = currentItem()?.id;
    const note = els.note.value;
    state.noteTimer = setTimeout(() => saveNote(itemId, note), 750);
  });
  document.addEventListener("keydown", (event) => {
    const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (editing || event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "arrowright" || key === "j") navigate(1);
    else if (key === "arrowleft") navigate(-1);
    else if (key === "k" || key === "enter") decide("keep");
    else if (key === "x" || key === "backspace") { event.preventDefault(); decide("skip"); }
    else if (key === "u") undo();
    else if (key === "f") $("#fullscreenButton").click();
    else if (["1", "2", "3", "4", "5"].includes(key)) setMode({ "1": "review", "2": "pair", "3": "depth", "4": "geometry", "5": "overview" }[key]);
  });
}

async function initialize() {
  bindEvents();
  try {
    const serverState = await api("/api/state");
    absorbServerState(serverState);
    els.connection.textContent = "Saved to disk";
    els.connection.className = "connection online";
    applyFilters();
  } catch (error) {
    els.connection.textContent = "Offline";
    els.connection.className = "connection offline";
    els.stage.innerHTML = `<div class="empty-state"><strong>${escapeHtml(error.message)}</strong></div>`;
  }
}

initialize();
