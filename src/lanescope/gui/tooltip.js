import { $ } from "./dom.js";
import * as format from "./format.js";
import { showBands } from "./grid.js";

export function bindTooltip() {
  const grids = $("grids");
  grids.addEventListener("pointermove", (event) => {
    const target = event.target.closest("rect.cell, path.bar");
    const svg = event.target.closest("svg");
    if (svg) {
      showBands(svg, target?.classList.contains("cell") ? target : null);
    }
    if (!target) {
      hideTooltip();
      return;
    }
    showTooltip(tooltipLines(target.dataset), event.clientX, event.clientY);
  });
  grids.addEventListener("pointerleave", () => {
    hideTooltip();
    for (const svg of grids.querySelectorAll("svg")) {
      showBands(svg, null);
    }
  });
}

function tooltipLines(data) {
  if (data.word !== undefined) {
    const bank = Number(data.bank);
    return [
      `Lane ${data.lane}`,
      `bank ${bank}, byte offset ${format.hex(Number(data.offset))}`,
      `word ${Number(data.slot) + 1} of ${data.degree} distinct words in bank ${bank}`,
      `bank ${bank} takes ${format.plural(Number(data.degree), "wavefront")}`,
    ];
  }
  // A sector goes by its offset from the first line, the same offset as the line labels.
  if (data.address !== undefined) {
    return [
      `Lane ${data.lane}`,
      `address ${data.address}`,
      `sector at ${format.offsetLabel(Number(data.offset))}`,
    ];
  }
  if (data.bank !== undefined) {
    return [
      `Bank ${data.bank}`,
      `${format.plural(Number(data.degree), "distinct word")}, so ${format.plural(Number(data.degree), "wavefront")}`,
    ];
  }
  return [
    `Sector at ${format.offsetLabel(Number(data.offset))}`,
    `${data.used} of 32 bytes used`,
    format.plural(Number(data.lanes), "lane"),
  ];
}

function showTooltip(lines, x, y) {
  const tip = $("tooltip");
  tip.replaceChildren();
  const [first, ...rest] = lines;
  const strong = document.createElement("strong");
  strong.textContent = first;
  tip.append(strong);
  for (const line of rest) {
    const element = document.createElement("div");
    element.className = "line";
    element.textContent = line;
    tip.append(element);
  }
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - box.width - 8);
  const top = y + 14 + box.height > window.innerHeight ? y - box.height - 10 : y + 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
}
