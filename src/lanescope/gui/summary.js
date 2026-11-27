import * as format from "./format.js";
import { $, cellSubline, meter, stat, tableCell } from "./dom.js";

// The rows that the table of lines shows, worst first. The summary keeps more.
export const SHOWN_LINES = 50;

const UNITS = {
  wavefronts: { noun: "passes", memory: "Shared-memory passes" },
  sectors: { noun: "sectors", memory: "Global-memory sectors" },
};

// The templates whose instances the table of kernels lists, by run and template. They stay
// open while the page switches the unit.
const openTemplates = new Set();

const DATA_DEPENDENT_TITLE =
  "The banks of this line follow the values that the kernel reads, such as the bins of a histogram. " +
  "No padding or swizzle removes these conflicts.";

export function renderProgress(answer) {
  $("summary-content").hidden = true;
  $("summary-progress").hidden = false;
  $("summary-progress-text").textContent =
    `Adding up the accesses of each launch, ${format.count(answer.done)} of ${launchCountText(answer.total)} done.`;
  const bar = $("summary-progress-bar");
  bar.max = Math.max(answer.total, 1);
  bar.value = answer.done;
}

// Fill the view with a summary in one unit. Choosing a line or a kernel calls `open` with
// the place of its worst access: the id of the trace, the PC, and the space.
export function renderSummary(data, unit, open) {
  $("summary-progress").hidden = true;
  $("summary-content").hidden = false;
  const { noun, memory } = UNITS[unit];
  const runs = data.runs;
  const where = runs.length === 1 ? runs[0] : `${format.count(runs.length)} runs`;
  const instances = data.kernels.reduce((sum, kernel) => sum + kernel.instances.length, 0);
  const templates = new Set(data.kernels.map((kernel) => `${kernel.run}\n${kernel.template}`)).size;
  const grouped = templates < instances ? ` in ${format.plural(templates, "template")}` : "";
  $("summary-subtitle").textContent =
    `${where}: ${launchCountText(data.traces)} of ${format.plural(instances, "kernel")}${grouped}`;
  const total = data.totals[unit];
  $("summary-stats").replaceChildren(
    stat(`${memory}, whole run`, `${format.count(total.achieved)} / ${format.count(total.ideal)}`,
      "The second number is the fewest that the same accesses could take."),
    stat(`Extra ${noun}`, `${format.count(total.excessive)} (${format.percent(total.excessive, total.achieved)})`),
    stat("Launches", format.count(data.traces)),
    stat("Kernels", format.count(instances)),
  );
  if (data.times.length === 1) {
    const [timed] = data.times;
    $("summary-stats").append(
      stat("Kernel time", format.duration(timed.total_ns), `The median of ${format.plural(timed.rounds, "round")} of lanescope time.`),
    );
  }
  renderNotes(data);
  renderLines(data, unit, open);
  renderKernels(data, unit, open);
}

function renderNotes(data) {
  const notes = [];
  if (data.without_lines) {
    notes.push(
      `No source lines in ${format.count(data.without_lines)} of ${launchCountText(data.traces)}. ` +
        "Build the program with -lineinfo, such as nvcc -lineinfo, to see the line of each instruction.",
    );
  }
  if (data.dropped.traces) {
    notes.push(
      `The record buffer filled up in ${format.count(data.dropped.traces)} of ${launchCountText(data.traces)} and left out ` +
        `${format.count(data.dropped.records)} accesses, therefore their counts are low. Record again with a larger ` +
        "--capacity or with fewer CTAs, such as --ctas 0:8.",
    );
  }
  if (data.unreadable) {
    notes.push(`The page skipped ${format.plural(data.unreadable, "file")} without a trace.`);
  }
  if (data.truncated) {
    notes.push(`The summary adds up only the first ${format.count(data.traces + data.unreadable)} trace files.`);
  }
  $("summary-notes").replaceChildren(
    ...notes.map((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    }),
  );
}

function renderLines(data, unit, open) {
  const { noun } = UNITS[unit];
  const lines = data.lines.filter((line) => line.unit === unit);
  const shown = lines.slice(0, SHOWN_LINES);
  $("summary-lines-title").textContent = `Lines with the most extra ${noun}`;
  const peak = Math.max(1, ...shown.map((line) => line.excessive));
  $("summary-lines").replaceChildren(
    ...shown.map((line) => {
      const row = openingRow(line.worst, open);
      row.append(
        extraCell(line, peak),
        tableCell(format.percent(line.excessive, line.kernel_achieved), "num"),
        lineCell(line),
        kernelCell(line, data.runs.length > 1, line.instances),
        tableCell(format.count(line.launches), "num"),
      );
      return row;
    }),
  );
  $("summary-lines-note").textContent =
    lines.length === 0
      ? `No line takes extra ${noun}.`
      : lines.length > shown.length
        ? `The table shows the ${shown.length} worst of ${format.count(lines.length)} lines.`
        : "";
}

function renderKernels(data, unit, open) {
  const withRun = data.runs.length > 1;
  const kernels = byExcess(data.kernels, unit);
  const peak = Math.max(1, ...kernels.map((kernel) => kernel.totals[unit].excessive));
  // The kernel time of each timed run, which the column of each row takes its part of.
  const runTimes = data.times.length ? new Map(data.times.map((timed) => [timed.run, timed.total_ns])) : null;
  for (const header of document.querySelectorAll("[data-time-column]")) {
    header.hidden = !runTimes;
  }
  $("summary-kernels").replaceChildren(
    ...kernels.flatMap((kernel) => {
      const row = kernelRow(kernel, unit, peak, open, kernelCell(kernel, withRun), runTimes);
      if (kernel.instances.length < 2) {
        return [row];
      }
      const key = `${kernel.run}\n${kernel.template}`;
      const rows = byExcess(kernel.instances, unit).map((instance) => {
        const name = kernelCell(instance, false);
        name.classList.add("instance-name");
        const shown = kernelRow(instance, unit, peak, open, name, runTimes);
        shown.classList.add("instance");
        shown.hidden = !openTemplates.has(key);
        return shown;
      });
      row.cells[2].append(instancesToggle(key, rows));
      return [row, ...rows];
    }),
  );
}

function byExcess(entries, unit) {
  const found = entries.filter((entry) => entry.totals[unit].achieved > 0);
  return found.sort((a, b) => b.totals[unit].excessive - a.totals[unit].excessive);
}

function kernelRow(entry, unit, peak, open, name, runTimes) {
  const counts = entry.totals[unit];
  const row = openingRow(entry.worst[unit], open);
  const extra = tableCell(format.count(counts.excessive), "num");
  extra.append(meter(counts.excessive, peak));
  row.append(
    extra,
    tableCell(format.percent(counts.excessive, counts.achieved), "num"),
    name,
    tableCell(format.count(entry.launches), "num"),
  );
  if (runTimes) {
    const timed = entry.time_ns !== null;
    row.append(
      tableCell(timed ? format.duration(entry.time_ns) : "–", "num"),
      tableCell(timed ? format.percent(entry.time_ns, runTimes.get(entry.run) ?? 0) : "–", "num"),
    );
  }
  return row;
}

// A button that shows or hides the instances of a template without opening the row it sits in.
function instancesToggle(key, rows) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "instances-toggle";
  button.textContent = format.plural(rows.length, "instance");
  const show = (open) => {
    button.setAttribute("aria-expanded", String(open));
    button.title = open ? "Hide the instances of this template" : "List the instances of this template";
    for (const row of rows) {
      row.hidden = !open;
    }
  };
  show(openTemplates.has(key));
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    const open = !openTemplates.has(key);
    if (open) {
      openTemplates.add(key);
    } else {
      openTemplates.delete(key);
    }
    show(open);
  });
  button.addEventListener("keydown", (event) => event.stopPropagation());
  return button;
}

// A row that opens its place on a click or on Enter, or a plain row without a place.
function openingRow(place, open) {
  const row = document.createElement("tr");
  if (!place) {
    row.className = "clean";
    return row;
  }
  row.tabIndex = 0;
  row.addEventListener("click", () => open(place));
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open(place);
    }
  });
  return row;
}

function extraCell(line, peak) {
  const element = tableCell(format.count(line.excessive), "num");
  element.append(meter(line.excessive, peak));
  if (line.ideal > 0) {
    element.append(cellSubline(`${format.multiple(line.achieved / line.ideal)} the ideal`));
  }
  return element;
}

function lineCell(line) {
  const element = document.createElement("td");
  const what = document.createElement("span");
  what.className = "what";
  const file = baseName(line.file);
  const call = file ? `${file}:${line.line}` : `line ${line.line}`;
  if (line.origin) {
    what.textContent = `${baseName(line.origin.file)}:${line.origin.line}`;
    what.title = line.origin.file;
  } else {
    what.textContent = line.line === null ? `PC ${line.pc}` : file ? call : `Line ${line.line}`;
  }
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = ` ${line.space} ${line.kind}, ${line.width} B`;
  what.append(kind);
  if (line.data_dependent) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "follows the input";
    tag.title = DATA_DEPENDENT_TITLE;
    what.append(tag);
  }
  element.append(what);
  const text = line.origin ? line.origin.text : line.source;
  if (text) {
    const code = cellSubline(text, "where mono");
    code.title = text;
    element.append(code);
  } else if (line.line === null) {
    element.append(cellSubline("no source line"));
  }
  if (line.origin) {
    const inlined = cellSubline(`inlined at ${call}`);
    inlined.title = line.source ? `${call}: ${line.source}` : call;
    element.append(inlined);
  }
  return element;
}

function baseName(path) {
  return path.split(/[\\/]/).pop();
}

function kernelCell(entry, withRun, instances = 1) {
  const element = document.createElement("td");
  const name = document.createElement("span");
  name.className = "what mono";
  name.textContent = entry.name || "kernel";
  name.title = entry.full_name || entry.name || "";
  element.append(name);
  if (instances > 1) {
    element.append(cellSubline(format.plural(instances, "instance")));
  }
  if (withRun) {
    element.append(cellSubline(entry.run));
  }
  return element;
}

function launchCountText(value) {
  return `${format.count(value)} ${value === 1 ? "launch" : "launches"}`;
}
