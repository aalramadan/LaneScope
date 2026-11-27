import { getJSON, isAbort } from "./api.js";
import { showError } from "./banner.js";
import { renderDiagnosis } from "./diagnosis.js";
import { $ } from "./dom.js";
import { readControls, writeControls } from "./fix-form.js";
import * as format from "./format.js";
import { fixControlsMatchSelection, latest, sameBuffer, state } from "./state.js";
import { scheduleWhatif } from "./whatif.js";

// The buffer of the running search. Instructions of one buffer share its ranked fixes,
// which take seconds on a large trace, so a new instruction lets that search finish.
let searching = null;

// Show the ranked fixes of a buffer, from memory when they are known for that buffer.
export function showSuggestions(buffer) {
  if (state.suggestions && sameBuffer(state.suggestions.buffer, buffer)) {
    renderSuggestions();
  } else if (sameBuffer(searching, buffer)) {
    showSearching();
  } else {
    runSuggestions(buffer).catch(showError);
  }
}

// A new trace or the summary ends the search of the trace it leaves.
export function abortSuggestions() {
  latest.suggest.abort();
  searching = null;
}

export function applySuggestion(item) {
  if (!fixControlsMatchSelection()) {
    return;
  }
  state.touched = true;
  writeControls(item.params);
  state.change = readControls();
  scheduleWhatif(0);
}

function showSearching() {
  $("suggestions").hidden = false;
  $("suggestions-note").textContent = "Searching paddings and swizzles on the recorded addresses...";
  $("suggestion-list").replaceChildren();
}

async function runSuggestions(buffer) {
  const traceId = state.traceId;
  state.suggestions = null;
  state.suggestionError = null;
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
      state.suggestionError = error.message;
      renderDiagnosis();
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
  renderDiagnosis();
  const caution = state.selected?.data_dependent
    ? " These conflicts follow the values that the kernel reads, therefore a fix ranked on this recording may help less on other inputs."
    : "";
  if (suggestions.found === 0) {
    $("suggestions-note").textContent = `No padding or swizzle lowers the excess wavefronts of this buffer.${caution}`;
    return;
  }
  $("suggestions-note").textContent =
    `${suggestions.found} changes lower the excess. The best ones come first.${caution}`;
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
    button.addEventListener("click", () => applySuggestion(item));
    entry.append(button);
    list.append(entry);
  }
}
