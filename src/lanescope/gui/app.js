import { readHash } from "./address.js";
import { showError } from "./banner.js";
import { onBufferEdited, prepareBuffer } from "./buffer.js";
import { setFixActions } from "./diagnosis.js";
import { $ } from "./dom.js";
import { hideFix, onControlsChanged, onKindChanged, typing } from "./fix-form.js";
import * as format from "./format.js";
import { onTableKey, renderInstructions, setInstructionSelect } from "./instructions.js";
import { setPatternSelect } from "./lanes.js";
import { loadTraces, openPlace, openTrace, showSummary } from "./navigation.js";
import { selectInstruction, selectRow } from "./selection.js";
import { setLineSelect } from "./source.js";
import { selectionToken, state } from "./state.js";
import { applySuggestion } from "./suggestions.js";
import { renderSummary } from "./summary.js";
import { setupTheme } from "./theme.js";
import { bindTooltip } from "./tooltip.js";

document.addEventListener("DOMContentLoaded", () => {
  for (const element of document.querySelectorAll("[data-term]")) {
    element.title = format.TERMS[element.dataset.term];
  }
  connectViews();
  setupTheme();
  followTopBar();
  bindEvents();
  loadTraces().catch(showError);
});

// Views that draw what loads them get their actions here, which keeps the imports one way.
function connectViews() {
  setInstructionSelect(selectInstruction);
  setLineSelect(selectInstruction);
  setPatternSelect(selectRow);
  setFixActions({ apply: applySuggestion, hide: hideFix });
}

function bindEvents() {
  $("trace-select").addEventListener("change", (event) => {
    if (event.target.value) {
      openTrace(event.target.value).catch(showError);
    }
  });
  $("summary-button").addEventListener("click", () => showSummary({ push: true }).catch(showError));
  for (const input of document.querySelectorAll('input[name="summary-unit"]')) {
    input.addEventListener("change", () => {
      state.summaryUnit = input.value;
      state.summaryUnitChosen = true;
      if (state.summary) {
        renderSummary(state.summary, state.summaryUnit, openPlace);
      }
    });
  }
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
    prepareBuffer(state.selected, selectionToken()).catch(showError);
  });
  $("fix-form").addEventListener("submit", (event) => event.preventDefault());
  bindTooltip();
  // Back from a launch opened in the summary returns to the summary, whose address has no hash.
  const follow = () => {
    const wanted = readHash();
    if (wanted.trace && wanted.trace !== state.traceId) {
      openTrace(wanted.trace, wanted).catch(showError);
    } else if (!wanted.trace && state.traces.length > 1 && state.view !== "summary") {
      showSummary().catch(showError);
    }
  };
  window.addEventListener("hashchange", follow);
}

// The style sheet keeps focused controls below the top bar, whose height grows as it wraps.
function followTopBar() {
  const bar = document.querySelector(".topbar");
  new ResizeObserver(() => {
    document.documentElement.style.setProperty("--topbar-height", `${bar.offsetHeight}px`);
  }).observe(bar);
}
