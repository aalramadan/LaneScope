import { Latest } from "./api.js";

export const state = {
  view: null,
  summary: null,
  summaryUnit: "wavefronts",
  summaryUnitChosen: false,
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
  suggestionError: null,
};

export const latest = {
  summary: new Latest(),
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

export function newSelection() {
  selection += 1;
  state.fixToken = null;
  for (const request of [latest.patterns, latest.access, latest.buffer, latest.whatif]) {
    request.abort();
  }
  clearTimeout(whatifTimer);
  return selection;
}

export function selectionToken() {
  return selection;
}

// The fix controls belong to the selection that drew them. A move made while another
// instruction loads would send the old buffer for the new instruction.
export function fixControlsMatchSelection() {
  return state.fixToken === selection && state.selected?.space === "shared";
}

export function isCurrentSelection(token) {
  return token === selection;
}

export function sameBuffer(first, second) {
  return Boolean(first && second && first.start === second.start && first.size === second.size);
}

// A new selection cancels the pending what-if, so its timer lives with the selection.
export function delayWhatif(run, delay) {
  clearTimeout(whatifTimer);
  whatifTimer = setTimeout(run, delay);
}

export function cancelWhatif() {
  clearTimeout(whatifTimer);
}
