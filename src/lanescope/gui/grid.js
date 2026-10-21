import { WARP_SIZE, degreeStep, offsetLabel } from "./format.js";

export const BANKS = 32;
export const WORD_BYTES = 4;
export const SECTOR_BYTES = 32;
export const SECTORS_PER_LINE = 4;
export const CELL = 10;
export const STEP = CELL + 2;

const LEFT = 50;
const TOP = 16;
const BARS = 36;
const AXIS = 16;
const BAR_WIDTH = 8;
const LINE_GAP = 8;
// Width of one character of the axis text, which sets how far apart line labels must be.
const LABEL_CHARACTER = 6.5;
const LABEL_MARGIN = 8;
const SVG_NS = "http://www.w3.org/2000/svg";

export function bankGridModel(phase, lanes) {
  const peak = Math.max(...phase.degrees);
  const gridTop = TOP + BARS + AXIS;
  const rowOf = new Map(phase.lanes.map((lane, index) => [lane, index]));
  const cells = phase.cells.map(([lane, bank, word, slot]) => {
    const degree = phase.degrees[bank];
    return {
      lane,
      bank,
      word,
      slot,
      degree,
      // The first word of a lane starts at the lane's own byte, which may lie inside
      // the word. Each further word of a wide lane starts at its first byte.
      offset: Math.max(lanes[lane].offset, word * WORD_BYTES),
      step: degreeStep(degree),
      x: LEFT + bank * STEP,
      y: gridTop + rowOf.get(lane) * STEP,
    };
  });
  const bars = phase.degrees.map((degree, bank) => ({
    bank,
    degree,
    x: LEFT + bank * STEP + (CELL - BAR_WIDTH) / 2,
    height: degree ? Math.max(2, (BARS * degree) / peak) : 0,
  }));
  return {
    width: LEFT + BANKS * STEP + 8,
    height: gridTop + phase.lanes.length * STEP + 4,
    baseline: TOP + BARS,
    gridTop,
    peak,
    worstBank: phase.worst_bank,
    rows: phase.lanes.map((lane, index) => ({ lane, y: gridTop + index * STEP })),
    columns: Array.from({ length: BANKS }, (_, bank) => LEFT + bank * STEP),
    cells,
    bars,
  };
}

// The bytes of a sector that the distinct lane addresses use, at most 32.
export function usedBytes(sector, lanes, width) {
  const start = sector.offset;
  const end = start + SECTOR_BYTES;
  const offsets = new Set(sector.lanes.map((lane) => lanes[lane].offset));
  let used = 0;
  for (const offset of offsets) {
    used += Math.max(0, Math.min(end, offset + width) - Math.max(start, offset));
  }
  return Math.min(used, SECTOR_BYTES);
}

export function sectorGridModel(access) {
  const gridTop = TOP + BARS + AXIS;
  let x = LEFT;
  let previous = null;
  const columns = access.sectors.map((sector) => {
    const line = Math.floor(sector.sector / SECTORS_PER_LINE);
    if (previous !== null && line !== previous) {
      x += LINE_GAP;
    }
    const used = usedBytes(sector, access.lanes, access.width);
    const column = {
      offset: sector.offset,
      line,
      lanes: sector.lanes,
      used,
      x,
      barHeight: Math.max(2, (BARS * used) / SECTOR_BYTES),
    };
    x += STEP;
    previous = line;
    return column;
  });
  const lines = [];
  for (const column of columns) {
    const last = lines.at(-1);
    if (last && last.line === column.line) {
      last.end = column.x + CELL;
    } else {
      lines.push({
        line: column.line,
        start: column.x,
        end: column.x + CELL,
        label: offsetLabel(column.line * SECTORS_PER_LINE * SECTOR_BYTES),
      });
    }
  }
  // Labels of narrow lines would overlap, so a line gets a label only when its label clears
  // the last one and the right edge of the grid.
  const width = x + 8;
  let labeledTo = -Infinity;
  for (const line of lines) {
    const half = (line.label.length * LABEL_CHARACTER) / 2;
    const center = (line.start + line.end) / 2;
    line.labeled = center - half >= labeledTo + LABEL_MARGIN && center + half <= width;
    if (line.labeled) {
      labeledTo = center + half;
    }
  }
  const rows = Array.from({ length: WARP_SIZE }, (_, lane) => ({
    lane,
    y: gridTop + lane * STEP,
    active: access.lanes[lane] !== null,
  }));
  const cells = columns.flatMap((column) =>
    column.lanes.map((lane) => ({ lane, offset: column.offset, x: column.x, y: rows[lane].y })),
  );
  return {
    width,
    height: gridTop + WARP_SIZE * STEP + 4,
    baseline: TOP + BARS,
    gridTop,
    columns,
    lines,
    rows,
    cells,
  };
}

// The bytes that a global access fetches in whole sectors and the bytes its lanes use.
export function sectorUse(access) {
  const used = access.sectors.reduce(
    (total, sector) => total + usedBytes(sector, access.lanes, access.width),
    0,
  );
  return { used, fetched: access.sectors.length * SECTOR_BYTES };
}

// Bits of a byte offset that a swizzle reads and flips, from the highest bit down.
export function swizzleBits({ bits, base, shift }, minimum = 12) {
  const total = Math.max(minimum, base + shift + bits);
  return Array.from({ length: total }, (_, index) => {
    const bit = total - 1 - index;
    let role = "none";
    if (bit >= base && bit < base + bits) {
      role = "target";
    } else if (bit >= base + shift && bit < base + shift + bits) {
      role = "source";
    }
    return { bit, role };
  });
}

export function renderBankGrid(phase, lanes, label) {
  const model = bankGridModel(phase, lanes);
  const root = svgRoot(model.width, model.height, label);
  root.append(bandsLayer(model.width, model.height));
  const peakBar = model.bars[model.worstBank];
  root.append(
    text(peakBar.x + BAR_WIDTH / 2, model.baseline - peakBar.height - 4, String(model.peak), "value"),
  );
  for (const bar of model.bars) {
    if (bar.degree) {
      const path = node("path", { class: "bar", d: barPath(bar.x, model.baseline, bar.height) });
      path.dataset.bank = bar.bank;
      path.dataset.degree = bar.degree;
      root.append(path);
    }
  }
  root.append(text(LEFT - 6, model.baseline + 12, "bank", "axis", "end"));
  for (const bank of [0, 8, 16, 24, BANKS - 1]) {
    root.append(text(model.columns[bank] + CELL / 2, model.baseline + 12, String(bank), "axis"));
  }
  const empty = node("g", { class: "empties" });
  for (const row of model.rows) {
    for (const x of model.columns) {
      empty.append(node("rect", { class: "empty", x, y: row.y, width: CELL, height: CELL, rx: 2 }));
    }
  }
  root.append(empty);
  model.rows.forEach((row, index) => {
    if (index % 4 === 0 || index === model.rows.length - 1) {
      root.append(text(LEFT - 6, row.y + CELL - 1, `lane ${row.lane}`, "axis", "end"));
    }
  });
  const data = node("g", { class: "cells" });
  for (const cell of model.cells) {
    const rect = node("rect", {
      class: `cell d${cell.step}`,
      x: cell.x,
      y: cell.y,
      width: CELL,
      height: CELL,
      rx: 2,
    });
    Object.assign(rect.dataset, {
      lane: cell.lane,
      bank: cell.bank,
      word: cell.word,
      slot: cell.slot,
      degree: cell.degree,
      offset: cell.offset,
    });
    data.append(rect);
  }
  root.append(data);
  return root;
}

export function renderSectorGrid(access, label) {
  const model = sectorGridModel(access);
  const root = svgRoot(model.width, model.height, label);
  root.append(bandsLayer(model.width, model.height));
  for (const line of model.lines.filter((item) => item.labeled)) {
    root.append(text((line.start + line.end) / 2, model.baseline + 12, line.label, "axis"));
  }
  for (const column of model.columns) {
    const path = node("path", {
      class: "bar",
      d: barPath(column.x + (CELL - BAR_WIDTH) / 2, model.baseline, column.barHeight),
    });
    Object.assign(path.dataset, { offset: column.offset, used: column.used, lanes: column.lanes.length });
    root.append(path);
  }
  const empty = node("g", { class: "empties" });
  for (const row of model.rows) {
    for (const column of model.columns) {
      empty.append(
        node("rect", { class: "empty", x: column.x, y: row.y, width: CELL, height: CELL, rx: 2 }),
      );
    }
  }
  root.append(empty);
  for (const row of model.rows) {
    if (row.lane % 4 === 0 || row.lane === WARP_SIZE - 1) {
      root.append(
        text(LEFT - 6, row.y + CELL - 1, `lane ${row.lane}`, row.active ? "axis" : "axis inactive", "end"),
      );
    }
  }
  const data = node("g", { class: "cells" });
  for (const cell of model.cells) {
    const rect = node("rect", {
      class: "cell touched",
      x: cell.x,
      y: cell.y,
      width: CELL,
      height: CELL,
      rx: 2,
    });
    Object.assign(rect.dataset, {
      lane: cell.lane,
      offset: cell.offset,
      address: access.lanes[cell.lane].address,
    });
    data.append(rect);
  }
  root.append(data);
  return root;
}

export function renderSwizzleBits(params) {
  const bits = swizzleBits(params);
  const box = 18;
  const width = bits.length * box + 2;
  const root = svgRoot(width, box + 18, "Bits of the byte offset that the swizzle reads and flips");
  bits.forEach(({ bit, role }, index) => {
    const x = 1 + index * box;
    root.append(node("rect", { class: `bit ${role}`, x, y: 1, width: box - 2, height: box - 2, rx: 3 }));
    root.append(text(x + (box - 2) / 2, box + 13, String(bit), "axis"));
  });
  return root;
}

// Bands that follow the pointer through the row and the column of a cell.
function bandsLayer(width, height) {
  const layer = node("g", { class: "bands" });
  layer.append(node("rect", { class: "band row-band", x: 0, y: 0, width, height: STEP, visibility: "hidden" }));
  layer.append(
    node("rect", { class: "band column-band", x: 0, y: 0, width: STEP, height, visibility: "hidden" }),
  );
  return layer;
}

export function showBands(svg, cell) {
  const row = svg.querySelector(".row-band");
  const column = svg.querySelector(".column-band");
  if (!row || !column) {
    return;
  }
  if (!cell) {
    row.setAttribute("visibility", "hidden");
    column.setAttribute("visibility", "hidden");
    return;
  }
  const x = Number(cell.getAttribute("x"));
  const y = Number(cell.getAttribute("y"));
  row.setAttribute("y", y - 1);
  column.setAttribute("x", x - 1);
  row.setAttribute("visibility", "visible");
  column.setAttribute("visibility", "visible");
}

function svgRoot(width, height, label) {
  const root = node("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    "aria-label": label,
  });
  root.classList.add("grid-svg");
  return root;
}

function barPath(x, baseline, height) {
  // Rounded data end at the top, square end at the baseline.
  const radius = Math.min(4, height / 2, BAR_WIDTH / 2);
  const top = baseline - height;
  const right = x + BAR_WIDTH;
  return (
    `M${x},${baseline} V${top + radius} Q${x},${top} ${x + radius},${top} ` +
    `H${right - radius} Q${right},${top} ${right},${top + radius} V${baseline} Z`
  );
}

function text(x, y, content, className, anchor = "middle") {
  const element = node("text", { x, y, class: className, "text-anchor": anchor });
  element.textContent = content;
  return element;
}

function node(tag, attributes) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}
