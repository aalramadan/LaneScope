import { getJSON, isAbort } from "./api.js";
import { renderDiagnosis } from "./diagnosis.js";
import { $ } from "./dom.js";
import { renderFixPanel, writeBufferFields } from "./fix-form.js";
import * as format from "./format.js";
import { fixControlsMatchSelection, isCurrentSelection, latest, selectionToken, state } from "./state.js";
import { showSuggestions } from "./suggestions.js";
import { scheduleWhatif, showFixError } from "./whatif.js";

export async function prepareBuffer(record, token) {
  state.suggestionError = null;
  const covered = state.buffer && state.bufferFromUser && bufferHolds(state.buffer, state.access);
  if (!covered) {
    let buffer;
    try {
      buffer = await latest.buffer.run((signal) =>
        getJSON("/api/buffer", { id: state.traceId, pc: record.pc }, signal),
      );
    } catch (error) {
      if (!isAbort(error) && isCurrentSelection(token)) {
        state.buffer = null;
        state.suggestionError = error.message;
        renderFixPanel(error.message);
        renderDiagnosis();
      }
      return;
    }
    if (!isCurrentSelection(token)) {
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

// The server checks a typed buffer and gives the row sizes that fit it.
export async function onBufferEdited() {
  // Like the other fix controls, the buffer fields belong to the selection that drew them.
  if (!fixControlsMatchSelection()) {
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
  const token = selectionToken();
  let buffer;
  try {
    buffer = await latest.buffer.run((signal) =>
      getJSON("/api/buffer", { id: state.traceId, start, size }, signal),
    );
  } catch (error) {
    if (!isAbort(error) && isCurrentSelection(token) && fixControlsMatchSelection()) {
      restoreBufferFields();
      showFixError(error.message);
    }
    return;
  }
  if (!isCurrentSelection(token) || !fixControlsMatchSelection()) {
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
