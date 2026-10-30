import { Latest, getJSON, isAbort, postJSON } from "./api.js";
import * as format from "./format.js";
import {
  renderBankGrid,
  renderSectorGrid,
  renderSwizzleBits,
  sectorUse,
  showBands,
} from "./grid.js";

const $ = (id) => document.getElementById(id);
const THEMES = ["system", "light", "dark"];
const WHATIF_DELAY = 80;

const state = {
  traces: [],
  traceId: null,
  overview: null,
  filter: "all",
  selected: null,
  patterns: null,
  row: null,
  access: null,
  buffer: null,
  bufferFromUser: false,
  change: { kind: "none" },
  fixToken: null,
  touched: false,
  result: null,
  suggestions: null,
};

const latest = {
  trace: new Latest(),
  patterns: new Latest(),
  access: new Latest(),
  buffer: new Latest(),
  whatif: new Latest(),
  suggest: new Latest(),
};

let whatifTimer = null;
// Each choice of a trace or an instruction starts a new selection. Answers that arrive
// for an older selection are dropped, and its requests are cancelled.
let selection = 0;
// Trace requests count apart, since a click on the old trace may start a selection while
// the new trace loads, and the trace answer must still win.
let traceRequests = 0;
// Counters of the requests that mark the grids and the result as busy, so an older
// request that ends never clears the mark of a newer one.
let accessRequests = 0;
let whatifRequests = 0;
// The buffer of the running search. Instructions of one buffer share its ranked fixes,
// which take seconds on a large trace, so a new instruction lets that search finish.
let searching = null;
// Buffer fields typed into since their last change event. An answer that arrives
// meanwhile, such as the check of the other field, leaves them as the user typed them.
const typing = new Set();

function newSelection() {
  selection += 1;
  state.fixToken = null;
  for (const request of [latest.patterns, latest.access, latest.buffer, latest.whatif]) {
    request.abort();
  }
  clearTimeout(whatifTimer);
  return selection;
}

// The fix controls belong to the selection that drew them. A move made while another
// instruction loads would send the old buffer for the new instruction.
function fixCurrent() {
  return state.fixToken === selection && state.selected?.space === "shared";
}

function current(token) {
  return token === selection;
}

function sameBuffer(first, second) {
  return Boolean(first && second && first.start === second.start && first.size === second.size);
}

document.addEventListener("DOMContentLoaded", () => {
  setupTheme();
  followTopBar();
  bindEvents();
  loadTraces().catch(showError);
});

// Loading

async function loadTraces() {
  const listing = await getJSON("/api/traces");
  state.traces = listing.traces;
  if (state.traces.length === 0) {
    const info = await getJSON("/api/info");
    $("empty-roots").textContent = `The page looks under ${info.roots.join(", ")}.`;
    $("empty-state").hidden = false;
    return;
  }
  renderTraceSelect(listing.truncated);
  const wanted = readHash();
  const known = state.traces.some((trace) => trace.id === wanted.trace);
  await openTrace(known ? wanted.trace : state.traces[0].id, known ? wanted : {});
}

async function openTrace(id, wanted = {}) {
  hideError();
  const request = ++traceRequests;
  let overview;
  try {
    overview = await latest.trace.run((signal) => getJSON("/api/trace", { id }, signal));
  } catch (error) {
    // The page keeps showing the trace it had, so the picker names that trace again.
    if (request === traceRequests && state.traceId) {
      $("trace-select").value = state.traceId;
    }
    throw error;
  }
  if (request !== traceRequests) {
    return;
  }
  // Only a trace that loaded ends the selection of the trace on screen.
  newSelection();
  latest.suggest.abort();
  searching = null;
  for (const section of ["result", "code", "suggestions"]) {
    $(section).hidden = true;
  }
  Object.assign(state, {
    traceId: id,
    overview,
    selected: null,
    patterns: null,
    row: null,
    access: null,
    buffer: null,
    bufferFromUser: false,
    result: null,
    suggestions: null,
  });
  $("trace-select").value = id;
  $("layout").hidden = false;
  renderHeader();
  renderInstructions();
  renderSource();
  const records = overview.instructions;
  const chosen =
    records.find((record) => record.pc === wanted.pc && record.space === wanted.space) || records[0];
  if (chosen) {
    await selectInstruction(chosen, wanted.row);
  } else {
    showNoAccesses();
  }
}

// A launch that only ran atomics leaves a trace without a memory instruction.
function showNoAccesses() {
  $("lanes-title").textContent = "No memory accesses";
  $("lanes-subtitle").textContent = "This trace holds no shared or global load or store.";
  for (const id of ["access-stats", "patterns", "grids", "legend"]) {
    $(id).replaceChildren();
  }
  for (const id of ["patterns-title", "pattern-detail", "text-view"]) {
    $(id).textContent = "";
  }
  renderFixPanel();
  history.replaceState(null, "", `#${new URLSearchParams({ trace: state.traceId })}`);
}

async function selectInstruction(record, wantedRow = null) {
  const token = newSelection();
  state.selected = record;
  state.result = null;
  renderInstructionSelection();
  renderSourceSelection();
  const patterns = await latest.patterns.run((signal) =>
    getJSON("/api/patterns", { id: state.traceId, pc: record.pc, space: record.space }, signal),
  );
  if (!current(token)) {
    return;
  }
  state.patterns = patterns;
  const rows = patterns.patterns.map((item) => item.row);
  const row = rows.includes(wantedRow) ? wantedRow : rows[0];
  if (!(await loadAccess(row, token))) {
    return;
  }
  if (record.space === "shared") {
    await prepareBuffer(record, token);
  } else {
    renderFixPanel();
  }
}

// A pattern button keeps the selection that drew it, so a click on the buttons of an
// instruction that the page is leaving does nothing.
async function selectRow(row, token) {
  if (!current(token)) {
    return;
  }
  if ((await loadAccess(row, token)) && state.change.kind !== "none" && fixCurrent()) {
    scheduleWhatif(0);
  }
}

// Load one access of the selected instruction and draw it. Returns whether it was drawn.
async function loadAccess(row, token) {
  const request = ++accessRequests;
  state.row = row;
  $("grids").classList.add("busy");
  let access;
  try {
    access = await latest.access.run((signal) => getJSON("/api/access", { id: state.traceId, row }, signal));
  } finally {
    if (request === accessRequests) {
      $("grids").classList.remove("busy");
    }
  }
  if (!current(token) || request !== accessRequests) {
    return false;
  }
  state.access = access;
  writeHash();
  renderLanes();
  return true;
}

async function prepareBuffer(record, token) {
  const covered = state.buffer && state.bufferFromUser && bufferHolds(state.buffer, state.access);
  if (!covered) {
    let buffer;
    try {
      buffer = await latest.buffer.run((signal) =>
        getJSON("/api/buffer", { id: state.traceId, pc: record.pc }, signal),
      );
    } catch (error) {
      if (!isAbort(error) && current(token)) {
        state.buffer = null;
        renderFixPanel(error.message);
      }
      return;
    }
    if (!current(token)) {
      return;
    }
    state.buffer = buffer;
    state.bufferFromUser = false;
  }
  renderFixPanel();
  showSuggestions(state.buffer);
  if (state.change.kind !== "none") {
    scheduleWhatif(0);
  }
}

function bufferHolds(buffer, access) {
  if (!access || access.space !== "shared") {
    return false;
  }
  const offsets = access.lanes.filter(Boolean).map((lane) => lane.offset);
  return offsets.every((offset) => offset >= buffer.start && offset < buffer.start + buffer.size);
}

// Header, table, and source

function renderTraceSelect(truncated) {
  const select = $("trace-select");
  select.replaceChildren();
  const groups = new Map();
  for (const trace of state.traces) {
    if (!groups.has(trace.folder)) {
      const group = document.createElement("optgroup");
      group.label = trace.folder;
      groups.set(trace.folder, group);
      select.append(group);
    }
    const option = document.createElement("option");
    option.value = trace.id;
    option.textContent = trace.kernel ? `${trace.name} (${trace.kernel})` : trace.name;
    groups.get(trace.folder).append(option);
  }
  if (truncated) {
    const more = document.createElement("option");
    more.disabled = true;
    more.textContent = `Only the first ${state.traces.length} trace files are listed`;
    select.append(more);
  }
  select.disabled = false;
}

function renderHeader() {
  const { overview } = state;
  const info = $("kernel-info");
  info.replaceChildren();
  const name = document.createElement("strong");
  name.textContent = overview.kernel || "kernel";
  info.append(name, ` on ${overview.arch || "an unknown GPU"}, ${format.count(overview.rows)} accesses`);
  const totals = $("totals");
  totals.replaceChildren();
  for (const [unit, label] of [
    ["wavefronts", "Excess wavefronts"],
    ["sectors", "Excess sectors"],
  ]) {
    if (overview.totals[unit]) {
      totals.append(stat(label, format.count(overview.totals[unit].excessive)));
    }
  }
  if (overview.measured) {
    const { matched, total } = overview.measured;
    totals.append(stat("Nsight Compute agrees", `${matched} of ${total}`));
  }
  document.title = `${overview.kernel || "kernel"} - LaneScope`;
}

function renderInstructions() {
  const body = $("instruction-rows");
  body.replaceChildren();
  const records = visibleRecords();
  const peak = Math.max(1, ...state.overview.instructions.map((record) => record.excessive));
  for (const record of records) {
    const row = document.createElement("tr");
    row.dataset.key = format.instructionKey(record);
    row.tabIndex = -1;
    row.setAttribute("aria-selected", "false");
    row.append(cell(record.pc, "mono"), accessCell(record), achievedCell(record), excessCell(record, peak));
    row.title = `${format.plural(record.requests, "request")}, counted in ${record.unit}`;
    row.addEventListener("click", () => selectInstruction(record).catch(showError));
    body.append(row);
  }
  const hidden = state.overview.instructions.length - records.length;
  $("instruction-note").textContent = hidden
    ? `${hidden} more instructions are hidden by the filter.`
    : "";
  renderInstructionSelection();
}

function visibleRecords() {
  const records = state.overview.instructions;
  switch (state.filter) {
    case "shared":
    case "global":
      return records.filter((record) => record.space === state.filter);
    case "excess":
      return records.filter((record) => record.excessive > 0);
    default:
      return records;
  }
}

function accessCell(record) {
  const element = document.createElement("td");
  element.className = "access-cell";
  const what = document.createElement("span");
  what.className = "what";
  what.textContent = `${record.space} ${record.kind}`;
  const where = document.createElement("span");
  where.className = "where";
  where.textContent = `${record.width} B, ${record.line ? `line ${record.line}` : "no source line"}`;
  element.append(what, where);
  return element;
}

function achievedCell(record) {
  const element = cell(format.count(record.achieved), "num");
  if (record.measured) {
    element.append(measuredMark(record));
  }
  const ideal = document.createElement("span");
  ideal.className = "where";
  ideal.textContent = `ideal ${format.count(record.ideal)}`;
  element.append(ideal);
  return element;
}

// A mark after each count of a GPU check, never color alone: a check where Nsight Compute
// confirms the count, a cross where it contradicts it, and a question mark where it has none.
function measuredMark(record) {
  const { achieved, agrees } = record.measured;
  const mark = document.createElement("span");
  if (achieved === null) {
    mark.className = "measured unknown";
    mark.textContent = " ?";
    mark.title = "Nsight Compute reports no count for this instruction";
    return mark;
  }
  mark.className = agrees ? "measured agrees" : "measured differs";
  mark.textContent = agrees ? " \u2713" : " \u2717";
  mark.title = `Nsight Compute measures ${format.count(achieved)}`;
  return mark;
}

function excessCell(record, peak) {
  const element = cell(format.count(record.excessive), "num");
  const meter = document.createElement("span");
  meter.className = "meter";
  const fill = document.createElement("span");
  fill.style.width = `${(100 * record.excessive) / peak}%`;
  meter.append(fill);
  element.append(meter);
  return element;
}

function renderInstructionSelection() {
  const key = state.selected && format.instructionKey(state.selected);
  let focusable = null;
  for (const row of $("instruction-rows").children) {
    const selected = row.dataset.key === key;
    row.setAttribute("aria-selected", String(selected));
    row.tabIndex = -1;
    if (selected) {
      focusable = row;
    }
  }
  focusable = focusable || $("instruction-rows").firstElementChild;
  if (focusable) {
    focusable.tabIndex = 0;
  }
}

function renderSource() {
  const list = $("source");
  list.replaceChildren();
  const lines = state.overview.source;
  $("source-note").hidden = lines.length > 0;
  // Wavefronts and sectors measure different things, so each line keeps one sum per unit.
  const excessByLine = new Map();
  for (const record of state.overview.instructions) {
    if (record.line) {
      const entry = excessByLine.get(record.line) || { wavefronts: 0, sectors: 0 };
      entry[record.unit] += record.excessive;
      excessByLine.set(record.line, entry);
    }
  }
  lines.forEach((text, index) => {
    const number = index + 1;
    const item = document.createElement("li");
    item.dataset.line = number;
    const button = document.createElement("button");
    button.type = "button";
    const label = document.createElement("span");
    label.className = "number";
    label.textContent = number;
    const code = document.createElement("span");
    code.className = "text";
    code.textContent = text;
    // The pane cuts long lines with an ellipsis, and the title shows the whole line on hover.
    code.title = text;
    button.append(label, code);
    const entry = excessByLine.get(number);
    if (entry) {
      const parts = ["wavefronts", "sectors"]
        .filter((unit) => entry[unit] > 0)
        .map((unit) => `${format.count(entry[unit])} excess ${unit}`);
      const badge = document.createElement("span");
      // A line with both units puts its badge under the code, which keeps the code legible.
      badge.className = parts.length > 1 ? "badge stacked" : parts.length ? "badge" : "badge zero";
      badge.textContent = parts.length ? parts.join(", ") : "ideal";
      button.append(badge);
      button.addEventListener("click", () => selectLine(number));
    } else {
      button.disabled = true;
    }
    item.append(button);
    list.append(item);
  });
  renderSourceSelection();
}

function selectLine(number) {
  const records = state.overview.instructions.filter((record) => record.line === number);
  if (records.length === 0) {
    return;
  }
  const keys = records.map(format.instructionKey);
  const current = state.selected ? keys.indexOf(format.instructionKey(state.selected)) : -1;
  // Clicking the line again steps through its instructions, worst first.
  selectInstruction(records[(current + 1) % records.length]).catch(showError);
}

function renderSourceSelection() {
  const line = state.selected?.line;
  for (const item of $("source").children) {
    const selected = Number(item.dataset.line) === line;
    item.classList.toggle("selected", selected);
    if (selected) {
      item.scrollIntoView({ block: "nearest" });
    }
  }
}

// Lanes

function renderLanes() {
  const { selected, access, patterns } = state;
  const line = selected.line ? `, line ${selected.line}` : "";
  $("lanes-title").textContent = `${selected.pc}${line}`;
  const source = selected.source ? `: ${selected.source}` : "";
  $("lanes-subtitle").textContent = `${format.instructionLabel(selected)}${source}`;
  const stats = $("access-stats");
  stats.replaceChildren(
    stat(`${capitalize(selected.unit)}, whole run`, `${format.count(selected.achieved)} / ${format.count(selected.ideal)}`),
    stat("This access", `${access.achieved} / ${access.ideal}`),
  );
  if (access.space === "global") {
    const { used, fetched } = sectorUse(access);
    stats.append(stat("Bytes used", `${format.count(used)} of ${format.count(fetched)}`));
  }
  if (selected.measured) {
    const value = selected.measured.achieved;
    stats.append(stat("Nsight Compute, whole run", value === null ? "no count" : format.count(value)));
  }
  stats.append(stat("Warp", `CTA ${access.cta}, warp ${access.warp}`));
  renderPatterns(patterns);
  renderGrids();
  $("text-view").textContent = access.text;
}

function renderPatterns(patterns) {
  const holder = $("patterns");
  // A pattern chosen by keyboard redraws the strip, and the focus moves to its new button.
  const focused = holder.contains(document.activeElement);
  holder.replaceChildren();
  const unit = state.selected.unit.replace(/s$/, "");
  $("patterns-title").textContent =
    patterns.count === 1 ? "One access pattern" : `${format.count(patterns.count)} access patterns`;
  holder.hidden = patterns.count === 1;
  const peak = Math.max(...patterns.patterns.map((item) => item.achieved));
  const token = selection;
  patterns.patterns.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = item.excessive ? "pattern" : "pattern ideal";
    const selected = item.row === state.row;
    button.setAttribute("aria-pressed", String(selected));
    const description =
      `Pattern ${index + 1}: ${format.plural(item.achieved, unit)} against an ideal of ${item.ideal}, ` +
      `in ${format.plural(item.repeats, "warp")}, first in CTA ${item.cta}, warp ${item.warp}`;
    button.setAttribute("aria-label", description);
    button.title = description;
    const bar = document.createElement("span");
    bar.style.height = `${Math.max(8, (100 * item.achieved) / peak)}%`;
    button.append(bar);
    button.addEventListener("click", () => selectRow(item.row, token).catch(showError));
    holder.append(button);
    if (selected) {
      $("pattern-detail").textContent = description;
      if (focused) {
        button.focus();
      }
    }
  });
  if (patterns.count > patterns.patterns.length) {
    const more = document.createElement("span");
    more.className = "note more";
    more.textContent = `${format.count(patterns.count - patterns.patterns.length)} milder patterns not shown`;
    holder.append(more);
  }
}

function renderGrids() {
  const holder = $("grids");
  const { access, result } = state;
  if (!access) {
    return;
  }
  holder.replaceChildren();
  holder.append(gridFigure("Recorded", access));
  const after = result?.access;
  const affected =
    result &&
    result.instructions.some(
      (item) => item.before.pc === state.selected.pc && item.before.space === state.selected.space,
    );
  if (after && affected && after.row === access.row) {
    holder.append(gridFigure("With the fix", after));
  }
  renderLegend(access.space);
}

function gridFigure(title, access) {
  const figure = document.createElement("figure");
  figure.className = "grid";
  const caption = document.createElement("figcaption");
  const name = document.createElement("strong");
  name.textContent = title;
  const counts = document.createElement("span");
  counts.className = "count";
  counts.textContent =
    access.space === "shared"
      ? `${format.plural(access.achieved, "wavefront")}, ideal ${access.ideal}, ${format.plural(access.conflicts, "bank conflict")}`
      : `${format.plural(access.achieved, "sector")} in ${format.plural(access.lines, "line")}, ideal ${access.ideal}`;
  caption.append(name, counts);
  figure.append(caption);
  if (access.space === "shared") {
    access.phases.forEach((phase, index) => {
      if (access.phases.length > 1) {
        const heading = document.createElement("p");
        heading.className = "phase-title";
        heading.textContent = `Phase ${index + 1}, lanes ${format.laneRanges(phase.lanes)}, ${format.plural(phase.wavefronts, "wavefront")}`;
        figure.append(heading);
      }
      const label = `${title}: bank ${phase.worst_bank} serves ${Math.max(...phase.degrees)} distinct words`;
      figure.append(renderBankGrid(phase, access.lanes, label));
    });
    if (access.paired) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "Lane pairs read identical addresses, so the access runs in fewer phases.";
      figure.append(note);
    }
  } else {
    const label = `${title}: ${access.achieved} sectors for ${access.active} active lanes`;
    figure.append(renderSectorGrid(access, label));
  }
  return figure;
}

function renderLegend(space) {
  const legend = $("legend");
  legend.replaceChildren();
  if (space === "shared") {
    legend.append("Distinct words in the bank of each cell:");
    format.DEGREE_BINS.forEach((_, step) => {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = `swatch d${step}`;
      item.append(swatch, format.degreeLabel(step));
      legend.append(item);
    });
    legend.append("Bars give the same count per bank.");
  } else {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "swatch touched";
    item.append(swatch, "sector that the lane touches");
    legend.append(
      item,
      "Bars give the bytes of each 32-byte sector that the warp uses. Labels give the offset of each 128-byte line from the first one.",
    );
  }
}

// Fix panel

function renderFixPanel(message = null) {
  const shared = state.selected?.space === "shared";
  const form = $("fix-form");
  form.hidden = !(shared && state.buffer);
  $("fix-note").hidden = !form.hidden;
  let note = globalNote(state.access);
  if (!state.selected) {
    note = "This trace has no memory instruction to change.";
  } else if (shared) {
    note = "This instruction has no buffer to change.";
  }
  $("fix-note").textContent = message || note;
  if (form.hidden) {
    state.fixToken = null;
    $("result").hidden = true;
    $("code").hidden = true;
    $("suggestions").hidden = true;
    return;
  }
  state.fixToken = selection;
  const { buffer } = state;
  writeBufferFields(buffer);
  $("buffer-note").textContent = `${format.bytes(buffer.size)} of ${buffer.element_bytes}-byte elements, accesses up to ${buffer.widest} B`;
  const rowSelect = $("pad-row");
  const previous = Number(rowSelect.value);
  rowSelect.replaceChildren();
  for (const size of buffer.row_sizes) {
    const option = document.createElement("option");
    option.value = size;
    option.textContent = format.bytes(size);
    rowSelect.append(option);
  }
  const preferred = buffer.row_sizes.includes(previous)
    ? previous
    : buffer.row_sizes.includes(128)
      ? 128
      : buffer.row_sizes[Math.min(buffer.row_sizes.length - 1, 4)];
  rowSelect.value = String(preferred);
  const padStep = Math.max(4, buffer.widest);
  const pad = $("pad-bytes");
  pad.step = String(padStep);
  pad.max = String(Math.max(64, padStep * 4));
  $("swizzle-base").min = String(Math.log2(buffer.widest));
  writeControls(state.change);
  // The browser snaps each control into its new limits, so the change follows the controls.
  if (state.change.kind !== "none") {
    state.change = readControls();
  }
  $("result").hidden = state.change.kind === "none";
  $("code").hidden = state.change.kind === "none";
}

function globalNote(access) {
  const intro = "The what-if engine changes shared memory layouts only.";
  if (!access || access.space !== "global") {
    return intro;
  }
  const { used, fetched } = sectorUse(access);
  const percent = fetched ? Math.round((100 * used) / fetched) : 100;
  const reading =
    used < fetched
      ? ` This warp fetches ${format.count(fetched)} bytes in ${format.plural(access.achieved, "sector")} and uses ${format.count(used)} of them (${percent}%). Lanes that read neighboring addresses share sectors and fetch fewer bytes.`
      : ` This warp uses every byte of its ${format.plural(access.achieved, "sector")}.`;
  return intro + reading;
}

function readControls() {
  const kind = document.querySelector('input[name="kind"]:checked').value;
  if (kind === "padding") {
    return {
      kind,
      row_bytes: Number($("pad-row").value),
      pad_bytes: Number($("pad-bytes").value),
    };
  }
  if (kind === "swizzle") {
    return {
      kind,
      bits: Number($("swizzle-bits").value),
      base: Number($("swizzle-base").value),
      shift: Number($("swizzle-shift").value),
    };
  }
  return { kind: "none" };
}

function writeControls(change) {
  for (const input of document.querySelectorAll('input[name="kind"]')) {
    input.checked = input.value === change.kind;
  }
  if (change.kind === "padding") {
    if ([...$("pad-row").options].some((option) => Number(option.value) === change.row_bytes)) {
      $("pad-row").value = String(change.row_bytes);
    }
    $("pad-bytes").value = String(change.pad_bytes);
  }
  if (change.kind === "swizzle") {
    $("swizzle-bits").value = String(change.bits);
    $("swizzle-base").value = String(change.base);
    // The shift slider still has the bits of the earlier swizzle as its minimum, and the
    // browser would raise a smaller shift to that minimum.
    $("swizzle-shift").min = String(change.bits);
    $("swizzle-shift").value = String(change.shift);
  }
  syncControlLabels();
}

function syncControlLabels() {
  const kind = document.querySelector('input[name="kind"]:checked').value;
  $("padding-controls").hidden = kind !== "padding";
  $("swizzle-controls").hidden = kind !== "swizzle";
  $("pad-bytes-value").textContent = format.bytes(Number($("pad-bytes").value));
  const bits = Number($("swizzle-bits").value);
  const shift = $("swizzle-shift");
  shift.min = String(bits);
  if (Number(shift.value) < bits) {
    shift.value = String(bits);
  }
  $("swizzle-bits-value").textContent = String(bits);
  const base = Number($("swizzle-base").value);
  $("swizzle-base-value").textContent = `${base} (${format.bytes(2 ** base)} chunks)`;
  $("swizzle-shift-value").textContent = shift.value;
  const figure = $("swizzle-bits-figure");
  figure.replaceChildren(renderSwizzleBits({ bits, base, shift: Number(shift.value) }));
}

function onControlsChanged() {
  state.touched = true;
  syncControlLabels();
  state.change = readControls();
  if (state.change.kind === "none") {
    clearTimeout(whatifTimer);
    latest.whatif.abort();
    state.result = null;
    $("result").hidden = true;
    $("code").hidden = true;
    renderGrids();
    return;
  }
  if (fixCurrent()) {
    scheduleWhatif(WHATIF_DELAY);
  }
}

function onKindChanged() {
  const kind = document.querySelector('input[name="kind"]:checked').value;
  // A first switch to a kind starts from the best ranked fix of that kind.
  const best = state.suggestions?.changes.find((item) => item.params.kind === kind);
  if (best && !state.touched) {
    writeControls(best.params);
  }
  onControlsChanged();
}

// The server checks a typed buffer and gives the row sizes that fit it.
async function onBufferEdited() {
  // Like the other fix controls, the buffer fields belong to the selection that drew them.
  if (!fixCurrent()) {
    return;
  }
  const start = format.parseInteger($("buffer-start").value);
  const size = format.parseInteger($("buffer-size").value);
  if (!Number.isInteger(start) || !Number.isInteger(size) || size <= 0) {
    restoreBufferFields();
    showFixError("The buffer needs a start, such as 0x0, and a size in bytes, such as 4096.");
    return;
  }
  // The server refuses such a buffer as well, but the browser logs each refusal as an error.
  if (start + size > state.buffer.max_end) {
    restoreBufferFields();
    showFixError(`A shared buffer must end at or below ${format.hex(state.buffer.max_end)}.`);
    return;
  }
  const token = selection;
  let buffer;
  try {
    buffer = await latest.buffer.run((signal) =>
      getJSON("/api/buffer", { id: state.traceId, start, size }, signal),
    );
  } catch (error) {
    if (!isAbort(error) && current(token) && fixCurrent()) {
      restoreBufferFields();
      showFixError(error.message);
    }
    return;
  }
  if (!current(token) || !fixCurrent()) {
    return;
  }
  state.buffer = buffer;
  state.bufferFromUser = true;
  renderFixPanel();
  showSuggestions(buffer);
  if (state.change.kind !== "none") {
    scheduleWhatif(0);
  }
}

// The fields go back to the buffer that the predictions still use.
function restoreBufferFields() {
  if (state.buffer) {
    writeBufferFields(state.buffer);
  }
}

function writeBufferFields(buffer) {
  const texts = { "buffer-start": format.hex(buffer.start), "buffer-size": String(buffer.size) };
  for (const [id, text] of Object.entries(texts)) {
    if (!typing.has(id)) {
      $(id).value = text;
    }
  }
}

function scheduleWhatif(delay) {
  clearTimeout(whatifTimer);
  whatifTimer = setTimeout(() => runWhatif().catch(showError), delay);
}

async function runWhatif() {
  const change = state.change;
  if (change.kind === "none" || !state.buffer || !fixCurrent()) {
    return;
  }
  const token = selection;
  const request = ++whatifRequests;
  $("result").classList.add("busy");
  $("result").hidden = false;
  let result = null;
  let failure = null;
  try {
    result = await latest.whatif.run((signal) =>
      postJSON(
        "/api/whatif",
        {
          id: state.traceId,
          buffer: { start: state.buffer.start, size: state.buffer.size },
          change,
          row: state.row,
        },
        signal,
      ),
    );
  } catch (error) {
    if (isAbort(error)) {
      return;
    }
    failure = error;
  } finally {
    if (request === whatifRequests) {
      $("result").classList.remove("busy");
    }
  }
  // The answer may belong to a change or an instruction that the page left meanwhile.
  if (!current(token) || request !== whatifRequests || JSON.stringify(change) !== JSON.stringify(state.change)) {
    return;
  }
  state.result = result;
  if (failure) {
    showFixError(failure.message);
  } else {
    renderResult();
    renderCode();
  }
  renderGrids();
}

function renderResult() {
  const { result } = state;
  const holder = $("result");
  holder.hidden = false;
  holder.replaceChildren();
  const unit = state.selected.unit;
  const totals = result.totals[unit] || result.totals.wavefronts;
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = `Excess ${unit} of the kernel, predicted from the recorded addresses`;
  const hero = document.createElement("div");
  hero.className = "hero";
  const from = document.createElement("span");
  from.className = "from";
  from.textContent = `${format.count(totals.excessive.before)} →`;
  const figure = document.createElement("span");
  figure.className = "figure";
  figure.textContent = format.count(totals.excessive.after);
  hero.append(from, figure, deltaBadge(totals.excessive.before, totals.excessive.after));
  holder.append(label, hero);
  const notes = [];
  if (result.extra_bytes) {
    notes.push(`The buffer grows by ${format.bytes(result.extra_bytes)}.`);
  } else {
    notes.push("The buffer keeps its size.");
  }
  if (result.later_shift) {
    notes.push(`Shared buffers above it move up by ${format.bytes(result.later_shift)}.`);
  }
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = notes.join(" ");
  holder.append(note, affectedTable(result.instructions));
}

function affectedTable(items) {
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const [name, numeric] of [
    ["Instruction", false],
    ["Before", true],
    ["After", true],
    ["", true],
  ]) {
    const th = document.createElement("th");
    th.textContent = name;
    if (numeric) {
      th.className = "num";
    }
    head.append(th);
  }
  table.append(head);
  for (const { before, after } of items) {
    const row = document.createElement("tr");
    const name = cell(before.pc, "mono");
    const where = document.createElement("span");
    where.className = "where";
    where.textContent = `${before.kind}${before.line ? `, line ${before.line}` : ""}`;
    name.append(where);
    row.append(name, cell(format.count(before.achieved), "num"), cell(format.count(after.achieved), "num"));
    const badge = document.createElement("td");
    badge.className = "num";
    badge.append(deltaBadge(before.achieved, after.achieved, true));
    row.append(badge);
    table.append(row);
  }
  return table;
}

// A change with an icon and words, never color alone. The compact form gives the percent.
function deltaBadge(before, after, compact = false) {
  const change = format.delta(before, after);
  const badge = document.createElement("span");
  badge.className = `delta ${change.direction}`;
  const icon = change.direction === "better" ? "▼" : change.direction === "worse" ? "▲" : "●";
  let words = `${change.text} (${change.percent})`;
  if (change.direction === "same") {
    words = "same";
  } else if (compact) {
    words = change.percent;
  }
  badge.textContent = `${icon} ${words}`;
  return badge;
}

function renderCode() {
  const holder = $("code-forms");
  holder.replaceChildren();
  $("code").hidden = false;
  for (const form of state.result.forms) {
    const block = document.createElement("div");
    block.className = "form-block";
    const head = document.createElement("div");
    head.className = "form-head";
    const label = document.createElement("span");
    label.textContent = form.label;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "ghost-button copy";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => copyText(form.code, copy));
    head.append(label, copy);
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = form.code;
    pre.append(code);
    block.append(head, pre);
    holder.append(block);
  }
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch {
    button.textContent = "Select and copy";
  }
  setTimeout(() => {
    button.textContent = "Copy";
  }, 1500);
}

function showFixError(message) {
  const holder = $("result");
  holder.hidden = false;
  holder.replaceChildren();
  const error = document.createElement("p");
  error.className = "error";
  error.textContent = message;
  holder.append(error);
  $("code").hidden = true;
}

// Show the ranked fixes of a buffer, from memory when they are known for that buffer.
function showSuggestions(buffer) {
  if (state.suggestions && sameBuffer(state.suggestions.buffer, buffer)) {
    renderSuggestions();
  } else if (sameBuffer(searching, buffer)) {
    showSearching();
  } else {
    runSuggestions(buffer).catch(showError);
  }
}

function showSearching() {
  $("suggestions").hidden = false;
  $("suggestions-note").textContent = "Searching paddings and swizzles on the recorded addresses...";
  $("suggestion-list").replaceChildren();
}

async function runSuggestions(buffer) {
  const traceId = state.traceId;
  state.suggestions = null;
  searching = buffer;
  showSearching();
  let answer;
  try {
    answer = await latest.suggest.run((signal) =>
      getJSON("/api/suggest", { id: traceId, start: buffer.start, size: buffer.size }, signal),
    );
  } catch (error) {
    if (!isAbort(error) && sameBuffer(state.buffer, buffer)) {
      $("suggestions-note").textContent = error.message;
    }
    return;
  } finally {
    if (searching === buffer) {
      searching = null;
    }
  }
  if (traceId !== state.traceId || !sameBuffer(state.buffer, buffer)) {
    return;
  }
  // The list stays known for the buffer, and it appears once a shared instruction is selected.
  state.suggestions = answer;
  if (state.selected?.space === "shared") {
    renderSuggestions();
  }
}

function renderSuggestions() {
  const { suggestions } = state;
  $("suggestions").hidden = false;
  const list = $("suggestion-list");
  list.replaceChildren();
  if (suggestions.found === 0) {
    $("suggestions-note").textContent = "No padding or swizzle lowers the excess wavefronts of this buffer.";
    return;
  }
  $("suggestions-note").textContent = `${suggestions.found} changes lower the excess. The best ones come first.`;
  for (const item of suggestions.changes) {
    const entry = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    const what = document.createElement("span");
    what.className = "what";
    what.textContent = item.change;
    const after = document.createElement("span");
    after.className = "after";
    after.textContent = `${format.count(item.excessive_before)} → ${format.count(item.excessive_after)} excess`;
    const extra = document.createElement("span");
    extra.className = "extra";
    const notes = [item.extra_bytes ? `adds ${format.bytes(item.extra_bytes)}` : "adds no bytes"];
    // The global reads of a 16-byte copy into the buffer grow when a change splits its lanes.
    const sectors = item.excessive_sectors_after - item.excessive_sectors_before;
    if (sectors !== 0) {
      const more = sectors > 0 ? "more" : "fewer";
      notes.push(`${format.count(Math.abs(sectors))} ${more} excess sectors`);
    }
    extra.textContent = notes.join(", ");
    button.append(what, after, extra);
    button.addEventListener("click", () => {
      if (!fixCurrent()) {
        return;
      }
      state.touched = true;
      writeControls(item.params);
      state.change = readControls();
      scheduleWhatif(0);
    });
    entry.append(button);
    list.append(entry);
  }
}

// Tooltip

function bindTooltip() {
  const grids = $("grids");
  grids.addEventListener("pointermove", (event) => {
    const target = event.target.closest("rect.cell, path.bar");
    const svg = event.target.closest("svg");
    if (svg) {
      showBands(svg, target?.classList.contains("cell") ? target : null);
    }
    if (!target) {
      hideTooltip();
      return;
    }
    showTooltip(tooltipLines(target.dataset), event.clientX, event.clientY);
  });
  grids.addEventListener("pointerleave", () => {
    hideTooltip();
    for (const svg of grids.querySelectorAll("svg")) {
      showBands(svg, null);
    }
  });
}

function tooltipLines(data) {
  if (data.word !== undefined) {
    const bank = Number(data.bank);
    return [
      `Lane ${data.lane}`,
      `bank ${bank}, byte offset ${format.hex(Number(data.offset))}`,
      `word ${Number(data.slot) + 1} of ${data.degree} distinct words in bank ${bank}`,
      `bank ${bank} takes ${format.plural(Number(data.degree), "wavefront")}`,
    ];
  }
  // A sector goes by its offset from the first line, the same offset as the line labels.
  if (data.address !== undefined) {
    return [
      `Lane ${data.lane}`,
      `address ${data.address}`,
      `sector at ${format.offsetLabel(Number(data.offset))}`,
    ];
  }
  if (data.bank !== undefined) {
    return [
      `Bank ${data.bank}`,
      `${format.plural(Number(data.degree), "distinct word")}, so ${format.plural(Number(data.degree), "wavefront")}`,
    ];
  }
  return [
    `Sector at ${format.offsetLabel(Number(data.offset))}`,
    `${data.used} of 32 bytes used`,
    format.plural(Number(data.lanes), "lane"),
  ];
}

function showTooltip(lines, x, y) {
  const tip = $("tooltip");
  tip.replaceChildren();
  const [first, ...rest] = lines;
  const strong = document.createElement("strong");
  strong.textContent = first;
  tip.append(strong);
  for (const line of rest) {
    const element = document.createElement("div");
    element.className = "line";
    element.textContent = line;
    tip.append(element);
  }
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - box.width - 8);
  const top = y + 14 + box.height > window.innerHeight ? y - box.height - 10 : y + 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
}

// Events, theme, and the address bar

function bindEvents() {
  $("trace-select").addEventListener("change", (event) =>
    openTrace(event.target.value).catch(showError),
  );
  for (const input of document.querySelectorAll('input[name="filter"]')) {
    input.addEventListener("change", () => {
      state.filter = input.value;
      renderInstructions();
    });
  }
  $("instruction-rows").addEventListener("keydown", onTableKey);
  for (const input of document.querySelectorAll('input[name="kind"]')) {
    input.addEventListener("change", onKindChanged);
  }
  for (const id of ["pad-row", "pad-bytes", "swizzle-bits", "swizzle-base", "swizzle-shift"]) {
    $(id).addEventListener("input", onControlsChanged);
  }
  for (const id of ["buffer-start", "buffer-size"]) {
    $(id).addEventListener("input", () => typing.add(id));
    // A field left without a change, such as one typed back to its old value, is free again.
    $(id).addEventListener("blur", () => typing.delete(id));
    $(id).addEventListener("change", () => {
      typing.delete(id);
      onBufferEdited().catch(showError);
    });
  }
  $("buffer-infer").addEventListener("click", () => {
    if (state.selected?.space !== "shared") {
      return;
    }
    // The buffer in use stays until the inferred one arrives, so the controls keep working.
    state.bufferFromUser = false;
    prepareBuffer(state.selected, selection).catch(showError);
  });
  $("fix-form").addEventListener("submit", (event) => event.preventDefault());
  bindTooltip();
  window.addEventListener("hashchange", () => {
    const wanted = readHash();
    if (wanted.trace && wanted.trace !== state.traceId) {
      openTrace(wanted.trace, wanted).catch(showError);
    }
  });
}

function onTableKey(event) {
  const rows = [...$("instruction-rows").children];
  const index = rows.indexOf(document.activeElement);
  if (index < 0) {
    return;
  }
  let next = null;
  if (event.key === "ArrowDown") {
    next = rows[Math.min(rows.length - 1, index + 1)];
  } else if (event.key === "ArrowUp") {
    next = rows[Math.max(0, index - 1)];
  } else if (event.key === "Enter" || event.key === " ") {
    next = rows[index];
  }
  if (!next) {
    return;
  }
  event.preventDefault();
  next.focus();
  const record = state.overview.instructions.find(
    (item) => format.instructionKey(item) === next.dataset.key,
  );
  selectInstruction(record).catch(showError);
}

// The style sheet keeps focused controls below the top bar, whose height grows as it wraps.
function followTopBar() {
  const bar = document.querySelector(".topbar");
  new ResizeObserver(() => {
    document.documentElement.style.setProperty("--topbar-height", `${bar.offsetHeight}px`);
  }).observe(bar);
}

function setupTheme() {
  let theme = "system";
  try {
    theme = localStorage.getItem("lanescope-theme") || "system";
  } catch {
    theme = "system";
  }
  applyTheme(THEMES.includes(theme) ? theme : "system");
  $("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme || "system";
    const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
    applyTheme(next);
    try {
      localStorage.setItem("lanescope-theme", next);
    } catch {
      // The page works without a stored theme.
    }
  });
}

function applyTheme(theme) {
  if (theme === "system") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = theme;
  }
  $("theme-label").textContent = `Theme: ${theme}`;
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const row = params.get("row");
  return {
    trace: params.get("trace"),
    pc: params.get("pc"),
    space: params.get("space"),
    row: row === null ? null : Number(row),
  };
}

function writeHash() {
  const params = new URLSearchParams({
    trace: state.traceId,
    pc: state.selected.pc,
    space: state.selected.space,
    row: String(state.row),
  });
  history.replaceState(null, "", `#${params}`);
}

// Small builders

function stat(label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value;
  wrapper.append(term, detail);
  return wrapper;
}

function cell(content, className) {
  const element = document.createElement("td");
  element.className = className;
  element.textContent = content;
  return element;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function showError(error) {
  if (isAbort(error)) {
    return;
  }
  const banner = $("banner");
  banner.textContent = error.message || String(error);
  banner.hidden = false;
}

function hideError() {
  $("banner").hidden = true;
}
