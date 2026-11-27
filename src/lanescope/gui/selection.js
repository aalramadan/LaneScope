import { writeHash } from "./address.js";
import { getJSON } from "./api.js";
import { prepareBuffer } from "./buffer.js";
import { $ } from "./dom.js";
import { renderFixPanel } from "./fix-form.js";
import { renderInstructionSelection } from "./instructions.js";
import { renderLanes } from "./lanes.js";
import { renderSourceSelection } from "./source.js";
import { fixControlsMatchSelection, isCurrentSelection, latest, newSelection, state } from "./state.js";
import { scheduleWhatif } from "./whatif.js";

// Counts the access requests that mark the grids as busy, so an older request that ends
// never clears the mark of a newer one.
let accessRequests = 0;

export async function selectInstruction(record, wantedRow = null) {
  const token = newSelection();
  state.selected = record;
  state.result = null;
  renderInstructionSelection();
  renderSourceSelection();
  const patterns = await latest.patterns.run((signal) =>
    getJSON("/api/patterns", { id: state.traceId, pc: record.pc, space: record.space }, signal),
  );
  if (!isCurrentSelection(token)) {
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
export async function selectRow(row, token) {
  if (!isCurrentSelection(token)) {
    return;
  }
  if ((await loadAccess(row, token)) && state.change.kind !== "none" && fixControlsMatchSelection()) {
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
  if (!isCurrentSelection(token) || request !== accessRequests) {
    return false;
  }
  state.access = access;
  writeHash();
  renderLanes();
  return true;
}
