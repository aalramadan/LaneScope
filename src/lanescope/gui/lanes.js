import { showError } from "./banner.js";
import { renderDiagnosis } from "./diagnosis.js";
import { $, stat } from "./dom.js";
import * as format from "./format.js";
import { renderBankGrid, renderSectorGrid, sectorUse } from "./grid.js";
import { selectionToken, state } from "./state.js";

// A pattern button loads its access, which redraws these lanes, so the page hands over
// that action at start.
let selectRow = null;

export function setPatternSelect(select) {
  selectRow = select;
}

export function renderLanes() {
  const { selected, access, patterns } = state;
  const line = selected.line ? `, line ${selected.line}` : "";
  $("lanes-title").textContent = `${selected.pc}${line}`;
  const source = selected.source ? `: ${selected.source}` : "";
  $("lanes-subtitle").textContent = `${format.instructionLabel(selected)}${source}`;
  const origin = $("lanes-origin");
  origin.hidden = !selected.origin;
  origin.textContent = selected.origin ? originLabel(selected.origin) : "";
  origin.title = selected.origin?.file ?? "";
  const stats = $("access-stats");
  stats.replaceChildren(
    stat(
      `${capitalize(selected.unit)}, whole run`,
      `${format.count(selected.achieved)} / ${format.count(selected.ideal)}`,
      `${format.TERMS[selected.unit]} The second number is the fewest the accesses could take.`,
    ),
    stat("This access", `${access.achieved} / ${access.ideal}`),
  );
  if (access.space === "global") {
    const { used, fetched } = sectorUse(access);
    stats.append(stat("Bytes used", `${format.count(used)} of ${format.count(fetched)}`));
  }
  if (selected.measured) {
    const value = selected.measured.achieved;
    stats.append(stat("Nsight Compute, whole run", value === null ? "no count" : format.count(value)));
  }
  stats.append(stat("Warp", `CTA ${access.cta}, warp ${access.warp}`));
  renderDiagnosis();
  renderPatterns(patterns);
  renderGrids();
  $("text-view").textContent = access.text;
}

function originLabel({ file, line, text }) {
  const name = file.split(/[\\/]/).pop();
  return `inlined from ${name}:${line}${text ? `: ${text}` : ""}`;
}

function renderPatterns(patterns) {
  const holder = $("patterns");
  // A pattern chosen by keyboard redraws the strip, and the focus moves to its new button.
  const focused = holder.contains(document.activeElement);
  holder.replaceChildren();
  const unit = state.selected.unit.replace(/s$/, "");
  $("patterns-title").textContent =
    patterns.count === 1 ? "One access pattern" : `${format.count(patterns.count)} access patterns`;
  holder.hidden = patterns.count === 1;
  const peak = Math.max(...patterns.patterns.map((item) => item.achieved));
  const token = selectionToken();
  patterns.patterns.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = item.excessive ? "pattern" : "pattern ideal";
    const selected = item.row === state.row;
    button.setAttribute("aria-pressed", String(selected));
    const description =
      `Pattern ${index + 1}: ${format.plural(item.achieved, unit)} against an ideal of ${item.ideal}, ` +
      `in ${format.plural(item.repeats, "warp")}, first in CTA ${item.cta}, warp ${item.warp}`;
    button.setAttribute("aria-label", description);
    button.title = description;
    const bar = document.createElement("span");
    bar.style.height = `${Math.max(8, (100 * item.achieved) / peak)}%`;
    button.append(bar);
    button.addEventListener("click", () => selectRow(item.row, token).catch(showError));
    holder.append(button);
    if (selected) {
      $("pattern-detail").textContent = description;
      if (focused) {
        button.focus();
      }
    }
  });
  if (patterns.count > patterns.patterns.length) {
    const more = document.createElement("span");
    more.className = "note more";
    more.textContent = `${format.count(patterns.count - patterns.patterns.length)} milder patterns not shown`;
    holder.append(more);
  }
}

export function renderGrids() {
  const holder = $("grids");
  const { access, result } = state;
  if (!access) {
    return;
  }
  holder.replaceChildren();
  holder.append(gridFigure("Recorded", access));
  const after = result?.access;
  const affected =
    result &&
    result.instructions.some(
      (item) => item.before.pc === state.selected.pc && item.before.space === state.selected.space,
    );
  if (after && affected && after.row === access.row) {
    holder.append(gridFigure("With the fix", after));
  }
  renderLegend(access.space);
}

function gridFigure(title, access) {
  const figure = document.createElement("figure");
  figure.className = "grid";
  const caption = document.createElement("figcaption");
  const name = document.createElement("strong");
  name.textContent = title;
  const counts = document.createElement("span");
  counts.className = "count";
  counts.textContent =
    access.space === "shared"
      ? `${format.plural(access.achieved, "wavefront")}, ideal ${access.ideal}, ${format.plural(access.conflicts, "bank conflict")}`
      : `${format.plural(access.achieved, "sector")} in ${format.plural(access.lines, "line")}, ideal ${access.ideal}`;
  caption.append(name, counts);
  figure.append(caption);
  if (access.space === "shared") {
    access.phases.forEach((phase, index) => {
      if (access.phases.length > 1) {
        const heading = document.createElement("p");
        heading.className = "phase-title";
        heading.textContent = `Phase ${index + 1}, lanes ${format.laneRanges(phase.lanes)}, ${format.plural(phase.wavefronts, "wavefront")}`;
        figure.append(heading);
      }
      const label = `${title}: bank ${phase.worst_bank} serves ${Math.max(...phase.degrees)} distinct words`;
      figure.append(renderBankGrid(phase, access.lanes, label));
    });
    if (access.paired) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "Lane pairs read identical addresses, so the access runs in fewer phases.";
      figure.append(note);
    }
  } else {
    const label = `${title}: ${access.achieved} sectors for ${access.active} active lanes`;
    figure.append(renderSectorGrid(access, label));
  }
  return figure;
}

function renderLegend(space) {
  const legend = $("legend");
  legend.replaceChildren();
  if (space === "shared") {
    legend.append("Distinct words in the bank of each cell:");
    format.DEGREE_BINS.forEach((_, step) => {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = `swatch d${step}`;
      item.append(swatch, format.degreeLabel(step));
      legend.append(item);
    });
    legend.append("Bars give the same count per bank.");
  } else {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "swatch touched";
    item.append(swatch, "sector that the lane touches");
    legend.append(
      item,
      "Bars give the bytes of each 32-byte sector that the warp uses. Labels give the offset of each 128-byte line from the first one.",
    );
  }
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
