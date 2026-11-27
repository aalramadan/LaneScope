import { clearHash, readHash, writeTraceHash } from "./address.js";
import { getJSON } from "./api.js";
import { hideError, showError } from "./banner.js";
import { $ } from "./dom.js";
import { renderFixPanel } from "./fix-form.js";
import { renderInstructions } from "./instructions.js";
import { selectInstruction } from "./selection.js";
import { renderSource } from "./source.js";
import { latest, newSelection, state } from "./state.js";
import { abortSuggestions } from "./suggestions.js";
import { renderProgress, renderSummary } from "./summary.js";
import { renderHeader, renderTraceSelect, showInPicker } from "./topbar.js";

// While the server adds up the traces, the page asks for its progress this often.
const SUMMARY_POLL = 400;

// Trace requests count apart, since a click on the old trace may start a selection while
// the new trace loads, and the trace answer must still win.
let traceRequests = 0;
// Each showing of the summary polls on its own, and a newer one ends the older loop.
let summaryRequests = 0;

export async function loadTraces() {
  const listing = await getJSON("/api/traces");
  state.traces = listing.traces;
  if (state.traces.length === 0) {
    const info = await getJSON("/api/info");
    $("empty-roots").textContent = `The page looks under ${info.roots.join(", ")}.`;
    $("empty-state").hidden = false;
    return;
  }
  renderTraceSelect(listing.truncated);
  $("summary-button").hidden = state.traces.length < 2;
  const wanted = readHash();
  let failed = false;
  if (wanted.trace) {
    try {
      await openTrace(wanted.trace, wanted);
      return;
    } catch (error) {
      showError(error);
      failed = true;
    }
  }
  // A page of many launches opens on their summary, and a page of one launch on that launch.
  if (state.traces.length > 1) {
    await showSummary({ keepError: failed });
  } else {
    await openTrace(state.traces[0].id);
  }
}

// The summary ranks the lines of every trace under the page by their extra passes. A summary
// that the user asks for from a launch gets an entry of its own in the history, and Back
// returns to that launch.
export async function showSummary({ push = false, keepError = false } = {}) {
  if (!keepError) {
    hideError();
  }
  newSelection();
  latest.trace.abort();
  abortSuggestions();
  traceRequests += 1;
  state.view = "summary";
  state.traceId = null;
  $("layout").hidden = true;
  $("summary-view").hidden = false;
  $("summary-button").setAttribute("aria-pressed", "true");
  $("trace-select").value = "";
  $("kernel-info").replaceChildren();
  $("totals").replaceChildren();
  document.title = "Summary - LaneScope";
  clearHash({ push });
  if (state.summary) {
    renderSummary(state.summary, state.summaryUnit, openPlace);
  }
  await pollSummary();
}

async function pollSummary() {
  const request = ++summaryRequests;
  for (;;) {
    const answer = await latest.summary.run((signal) => getJSON("/api/summary", {}, signal));
    if (request !== summaryRequests || state.view !== "summary") {
      return;
    }
    if (answer.state === "done") {
      // A first summary without bank conflicts but with wasted sectors opens on global memory.
      const { wavefronts, sectors } = answer.totals;
      if (!state.summary && !state.summaryUnitChosen && wavefronts.excessive === 0 && sectors.excessive > 0) {
        state.summaryUnit = "sectors";
        document.querySelector('input[name="summary-unit"][value="sectors"]').checked = true;
      }
      state.summary = answer;
      renderSummary(answer, state.summaryUnit, openPlace);
      return;
    }
    if (!state.summary) {
      renderProgress(answer);
    }
    await new Promise((resolve) => setTimeout(resolve, SUMMARY_POLL));
  }
}

export function openPlace(place) {
  openTrace(place.id, { pc: place.pc, space: place.space }).catch(showError);
}

export async function openTrace(id, wanted = {}) {
  hideError();
  const request = ++traceRequests;
  let overview;
  try {
    overview = await latest.trace.run((signal) => getJSON("/api/trace", { id }, signal));
  } catch (error) {
    // The page keeps what it showed. The picker therefore names that trace again, or the summary.
    if (request === traceRequests) {
      $("trace-select").value = state.traceId ?? "";
    }
    throw error;
  }
  if (request !== traceRequests) {
    return;
  }
  // A launch opened from the summary gets an entry of its own in the history, and Back
  // returns to the summary. A launch that Back or Forward reaches keeps its entry.
  if (state.view === "summary" && !location.hash) {
    writeTraceHash(id, { push: true });
  }
  // Only a trace that loaded ends the selection of the trace on screen.
  newSelection();
  abortSuggestions();
  latest.summary.abort();
  state.view = "trace";
  $("summary-view").hidden = true;
  $("summary-button").setAttribute("aria-pressed", "false");
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
    suggestionError: null,
  });
  showInPicker(id, overview);
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
  $("diagnosis").hidden = true;
  $("lanes-subtitle").textContent = "This trace holds no shared or global load or store.";
  $("lanes-origin").hidden = true;
  for (const id of ["access-stats", "patterns", "grids", "legend"]) {
    $(id).replaceChildren();
  }
  for (const id of ["patterns-title", "pattern-detail", "text-view"]) {
    $(id).textContent = "";
  }
  renderFixPanel();
  writeTraceHash(state.traceId);
}
