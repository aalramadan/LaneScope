import { isAbort, postJSON } from "./api.js";
import { showError } from "./banner.js";
import { renderDiagnosis } from "./diagnosis.js";
import { $, tableCell } from "./dom.js";
import * as format from "./format.js";
import { renderGrids } from "./lanes.js";
import {
  delayWhatif,
  fixControlsMatchSelection,
  isCurrentSelection,
  latest,
  selectionToken,
  state,
} from "./state.js";

// Counts the what-if requests that mark the result as busy, so an older request that ends
// never clears the mark of a newer one.
let whatifRequests = 0;

export function scheduleWhatif(delay) {
  delayWhatif(() => runWhatif().catch(showError), delay);
}

async function runWhatif() {
  const change = state.change;
  if (change.kind === "none" || !state.buffer || !fixControlsMatchSelection()) {
    return;
  }
  const token = selectionToken();
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
  if (!isCurrentSelection(token) || request !== whatifRequests || JSON.stringify(change) !== JSON.stringify(state.change)) {
    return;
  }
  state.result = result;
  if (failure) {
    showFixError(failure.message);
  } else {
    renderWhatifResult();
    renderCodeForms();
  }
  renderGrids();
  renderDiagnosis();
}

function renderWhatifResult() {
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
    const name = tableCell(before.pc, "mono");
    const where = document.createElement("span");
    where.className = "where";
    where.textContent = `${before.kind}${before.line ? `, line ${before.line}` : ""}`;
    name.append(where);
    row.append(name, tableCell(format.count(before.achieved), "num"), tableCell(format.count(after.achieved), "num"));
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

function renderCodeForms() {
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

export function showFixError(message) {
  const holder = $("result");
  holder.hidden = false;
  holder.replaceChildren();
  const error = document.createElement("p");
  error.className = "error";
  error.textContent = message;
  holder.append(error);
  $("code").hidden = true;
}
