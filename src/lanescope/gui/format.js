export const WARP_SIZE = 32;

// Five ordinal steps for the distinct words that one bank serves, the same
// bins as the HTML report: 1, 2, 3 to 4, 5 to 8, and 9 or more.
export const DEGREE_BINS = [
  [1, 1],
  [2, 2],
  [3, 4],
  [5, 8],
  [9, 32],
];

const NUMBER = new Intl.NumberFormat("en-US");

export function degreeStep(degree) {
  const step = DEGREE_BINS.findIndex(([low, high]) => degree >= low && degree <= high);
  if (step < 0) {
    throw new RangeError(`a bank serves 1 to 32 distinct words, not ${degree}`);
  }
  return step;
}

export function degreeLabel(step) {
  const [low, high] = DEGREE_BINS[step];
  return low === high ? String(low) : `${low} to ${high}`;
}

export function count(value) {
  return NUMBER.format(value);
}

// A count with its noun, such as 1 wavefront or 4,096 wavefronts.
export function plural(value, noun) {
  return `${NUMBER.format(value)} ${noun}${value === 1 ? "" : "s"}`;
}

export function hex(value) {
  return `0x${value.toString(16)}`;
}

// A byte offset from the first line of a global access, such as +0x80.
export function offsetLabel(value) {
  return `+${hex(value)}`;
}

export function bytes(value) {
  return `${NUMBER.format(value)} B`;
}

// Lanes as ranges, such as 0-7, 16, 24-31.
export function laneRanges(lanes) {
  if (lanes.length === 0) {
    return "none";
  }
  const parts = [];
  let start = lanes[0];
  let previous = lanes[0];
  for (const lane of lanes.slice(1)) {
    if (lane !== previous + 1) {
      parts.push(start === previous ? `${start}` : `${start}-${previous}`);
      start = lane;
    }
    previous = lane;
  }
  parts.push(start === previous ? `${start}` : `${start}-${previous}`);
  return parts.join(", ");
}

// The change between two counts, where a lower count is better.
export function delta(before, after) {
  const change = after - before;
  const direction = change < 0 ? "better" : change > 0 ? "worse" : "same";
  const sign = change < 0 ? "−" : change > 0 ? "+" : "";
  const percent =
    before === 0 ? (change === 0 ? "0%" : "new") : `${sign}${Math.round((Math.abs(change) / before) * 100)}%`;
  return { direction, text: `${sign}${NUMBER.format(Math.abs(change))}`, percent };
}

// Integers as the page shows them, decimal or hexadecimal with 0x.
export function parseInteger(text) {
  const value = String(text).trim().toLowerCase();
  if (/^0x[0-9a-f]+$/.test(value)) {
    return Number.parseInt(value.slice(2), 16);
  }
  if (/^[0-9]+$/.test(value)) {
    return Number.parseInt(value, 10);
  }
  return Number.NaN;
}

export function instructionKey(record) {
  return `${record.pc}/${record.space}`;
}

export function instructionLabel(record) {
  return `${record.space} ${record.kind}, ${record.width} B`;
}
