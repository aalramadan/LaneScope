import { renderDiagnosis } from "./diagnosis.js";
import { $ } from "./dom.js";
import * as format from "./format.js";
import { renderSwizzleBits, sectorUse } from "./grid.js";
import { renderGrids } from "./lanes.js";
import { cancelWhatif, fixControlsMatchSelection, latest, selectionToken, state } from "./state.js";
import { scheduleWhatif } from "./whatif.js";

const WHATIF_DELAY = 80;

// Buffer fields typed into since their last change event. An answer that arrives
// meanwhile, such as the check of the other field, leaves them as the user typed them.
export const typing = new Set();

export function renderFixPanel(message = null) {
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
  state.fixToken = selectionToken();
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

export function readControls() {
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

export function writeControls(change) {
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

export function onControlsChanged() {
  state.touched = true;
  syncControlLabels();
  state.change = readControls();
  if (state.change.kind === "none") {
    cancelWhatif();
    latest.whatif.abort();
    state.result = null;
    $("result").hidden = true;
    $("code").hidden = true;
    renderGrids();
    renderDiagnosis();
    return;
  }
  if (fixControlsMatchSelection()) {
    scheduleWhatif(WHATIF_DELAY);
  }
}

export function onKindChanged() {
  const kind = document.querySelector('input[name="kind"]:checked').value;
  // A first switch to a kind starts from the best ranked fix of that kind.
  const best = state.suggestions?.changes.find((item) => item.params.kind === kind);
  if (best && !state.touched) {
    writeControls(best.params);
  }
  onControlsChanged();
}

export function hideFix() {
  writeControls({ kind: "none" });
  onControlsChanged();
}

export function writeBufferFields(buffer) {
  const texts = { "buffer-start": format.hex(buffer.start), "buffer-size": String(buffer.size) };
  for (const [id, text] of Object.entries(texts)) {
    if (!typing.has(id)) {
      $(id).value = text;
    }
  }
}
