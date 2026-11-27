import { $, stat } from "./dom.js";
import * as format from "./format.js";
import { state } from "./state.js";

export function renderTraceSelect(truncated) {
  const select = $("trace-select");
  // The picker names no trace while the summary is on screen.
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Summary of every launch";
  placeholder.hidden = true;
  select.replaceChildren(placeholder);
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
    const kernel = trace.kernel_name || trace.kernel;
    option.textContent = kernel ? `${trace.name} (${kernel})` : trace.name;
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

// A trace opened from the summary may lie beyond the listing, and the picker then adds it.
export function showInPicker(id, overview) {
  const select = $("trace-select");
  if (![...select.options].some((option) => option.value === id)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = `${id.split("/").pop()} (${overview.name || overview.kernel || "kernel"})`;
    select.append(option);
  }
  select.value = id;
}

export function renderHeader() {
  const { overview } = state;
  const info = $("kernel-info");
  info.replaceChildren();
  const name = document.createElement("strong");
  name.textContent = overview.name || overview.kernel || "kernel";
  name.title = overview.full_name || "";
  info.append(name, ` on ${overview.arch || "an unknown GPU"}, ${format.count(overview.rows)} accesses`);
  const totals = $("totals");
  totals.replaceChildren();
  for (const [unit, label] of [
    ["wavefronts", "Excess wavefronts"],
    ["sectors", "Excess sectors"],
  ]) {
    if (overview.totals[unit]) {
      const title = `${format.TERMS[unit]} ${format.TERMS.excess}`;
      totals.append(stat(label, format.count(overview.totals[unit].excessive), title));
    }
  }
  if (overview.measured) {
    const { matched, total } = overview.measured;
    totals.append(stat("Nsight Compute agrees", `${matched} of ${total}`));
  }
  document.title = `${overview.name || overview.kernel || "kernel"} - LaneScope`;
}
