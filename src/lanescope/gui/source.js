import { showError } from "./banner.js";
import { $ } from "./dom.js";
import * as format from "./format.js";
import { state } from "./state.js";

// A clicked line loads one of its instructions, which redraws this panel, so the page hands
// over that action at start.
let selectInstruction = null;

export function setLineSelect(select) {
  selectInstruction = select;
}

export function renderSource() {
  const list = $("source");
  list.replaceChildren();
  const lines = state.overview.source;
  $("source-note").hidden = lines.length > 0;
  // Wavefronts and sectors measure different things, so each line keeps one sum per unit.
  const excessByLine = new Map();
  for (const record of state.overview.instructions) {
    if (record.line) {
      const entry = excessByLine.get(record.line) || { wavefronts: 0, sectors: 0 };
      entry[record.unit] += record.excessive;
      excessByLine.set(record.line, entry);
    }
  }
  lines.forEach((text, index) => {
    const number = index + 1;
    const item = document.createElement("li");
    item.dataset.line = number;
    const button = document.createElement("button");
    button.type = "button";
    const label = document.createElement("span");
    label.className = "number";
    label.textContent = number;
    const code = document.createElement("span");
    code.className = "text";
    code.textContent = text;
    // The pane cuts long lines with an ellipsis, and the title shows the whole line on hover.
    code.title = text;
    button.append(label, code);
    const entry = excessByLine.get(number);
    if (entry) {
      const parts = ["wavefronts", "sectors"]
        .filter((unit) => entry[unit] > 0)
        .map((unit) => `${format.count(entry[unit])} excess ${unit}`);
      const badge = document.createElement("span");
      // A line with both units puts its badge under the code, which keeps the code legible.
      badge.className = parts.length > 1 ? "badge stacked" : parts.length ? "badge" : "badge zero";
      badge.textContent = parts.length ? parts.join(", ") : "ideal";
      button.append(badge);
      button.addEventListener("click", () => selectLine(number));
    } else {
      button.disabled = true;
    }
    item.append(button);
    list.append(item);
  });
  renderSourceSelection();
}

function selectLine(number) {
  const records = state.overview.instructions.filter((record) => record.line === number);
  if (records.length === 0) {
    return;
  }
  const keys = records.map(format.instructionKey);
  const current = state.selected ? keys.indexOf(format.instructionKey(state.selected)) : -1;
  // Clicking the line again steps through its instructions, worst first.
  selectInstruction(records[(current + 1) % records.length]).catch(showError);
}

export function renderSourceSelection() {
  const line = state.selected?.line;
  for (const item of $("source").children) {
    const selected = Number(item.dataset.line) === line;
    item.classList.toggle("selected", selected);
    if (selected) {
      item.scrollIntoView({ block: "nearest" });
    }
  }
}
