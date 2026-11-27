import { showError } from "./banner.js";
import { $, cell, detail, meter } from "./dom.js";
import * as format from "./format.js";
import { state } from "./state.js";

// A chosen row loads its instruction, which redraws this table, so the page hands over
// that action at start.
let selectInstruction = null;

export function setInstructionSelect(select) {
  selectInstruction = select;
}

export function renderInstructions() {
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
  mark.textContent = agrees ? " ✓" : " ✗";
  mark.title = `Nsight Compute measures ${format.count(achieved)}`;
  return mark;
}

function excessCell(record, peak) {
  const element = cell(format.count(record.excessive), "num");
  element.append(meter(record.excessive, peak));
  if (record.excessive > 0) {
    const all = state.overview.totals[record.unit]?.achieved ?? 0;
    const part = detail(`${format.percent(record.excessive, all)} of all`);
    part.title = format.TERMS.share;
    element.append(part);
  }
  return element;
}

export function renderInstructionSelection() {
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

export function onTableKey(event) {
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
