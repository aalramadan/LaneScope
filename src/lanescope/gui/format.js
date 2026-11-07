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

export const TERMS = {
  pc: "The address of the instruction in the machine code of the kernel.",
  achieved:
    "Passes of shared memory, or 32-byte sectors of global memory, that the instruction took over the whole run.",
  excess: "Passes or sectors above the fewest that the same accesses could take.",
  wavefronts: "A wavefront is one pass of shared memory. In one pass, each of the 32 banks hands out one 4-byte word.",
  sectors: "A sector is a 32-byte piece of global memory. The GPU always reads whole sectors.",
};

const KIND_NAMES = { load: "load", store: "store", atomic: "atomic operation" };
const GLOBAL_VERBS = { load: "fetches", store: "writes", atomic: "touches" };

// How many times the fewest, such as 32× or 1.4×.
export function multiple(ratio) {
  return ratio >= 10 ? `${Math.round(ratio)}×` : `${ratio.toFixed(1).replace(/\.0$/, "")}×`;
}

export function share(fraction) {
  if (fraction >= 0.995) {
    return "all";
  }
  return fraction < 0.01 ? "less than 1%" : `${Math.round(fraction * 100)}%`;
}

// The verdict on one instruction in plain words. The title names the problem, the cause
// explains the warp shown, the cost gives the whole run and its part of the kernel's waste,
// and the advice tells how to fix a global access. The totals are those of the kernel per unit.
export function diagnosis(record, access, totals) {
  const kind = KIND_NAMES[record.kind] ?? record.kind;
  const shared = record.unit === "wavefronts";
  const verb = shared ? "takes" : (GLOBAL_VERBS[record.kind] ?? "touches");
  if (record.excessive === 0) {
    const clean = Object.values(totals).every((total) => total.excessive === 0);
    return {
      tone: "ok",
      title: shared ? "No bank conflicts" : "No wasted sectors",
      badge: "",
      cause: shared
        ? `Every warp of this ${kind} takes the fewest passes of shared memory it can.`
        : `Every warp of this ${kind} uses all the bytes of the 32-byte sectors it ${verb}.`,
      cost: clean
        ? "No instruction of this kernel takes extra passes or sectors."
        : "The instructions with extra passes or sectors come first in the table.",
      advice: "",
    };
  }
  const noun = shared ? "passes" : "sectors";
  const kernelExcess = totals[record.unit]?.excessive ?? 0;
  const part = share(kernelExcess > 0 ? record.excessive / kernelExcess : 0);
  const where = record.line ? "This line" : "This instruction";
  return {
    tone: "warn",
    title: shared ? "Bank conflicts" : "Wasted sectors",
    badge: record.ideal > 0 ? `${multiple(record.achieved / record.ideal)} the ${noun} it needs` : "",
    cause: shared ? sharedCause(access, kind) : globalCause(access, kind),
    cost:
      `Over the whole run, this ${kind} ${verb} ${count(record.achieved)} ${noun} where ${count(record.ideal)} would do. ` +
      `${where} causes ${part === "all" ? "all the" : `${part} of the`} extra ${noun} of the kernel.`,
    advice: shared
      ? ""
      : `Fix by letting neighboring threads touch neighboring addresses. A warp then fills each sector it ${verb}.`,
  };
}

function sharedCause(access, kind) {
  if (access.excessive === 0) {
    return `The warp shown has no conflict, but other warps of this ${kind} do. The access patterns list them worst first.`;
  }
  const words = Math.max(...access.phases.flatMap((phase) => phase.degrees));
  if (words <= 1) {
    return `The warp shown takes ${count(access.achieved)} passes where ${count(access.ideal)} would do.`;
  }
  return (
    `In the warp shown, up to ${words} threads hit different words in the same bank. ` +
    "A bank hands out one word per pass, therefore these threads take turns."
  );
}

function globalCause(access, kind) {
  if (access.excessive === 0) {
    return `The warp shown wastes no sector, but other warps of this ${kind} do. The access patterns list them worst first.`;
  }
  return (
    `The threads of the warp shown touch addresses spread over ${count(access.achieved)} sectors of 32 bytes, ` +
    `where ${count(access.ideal)} would hold their data.`
  );
}

// A ranked change as the object of "Fix by", such as "padding each 128-byte row by 4 bytes".
export function fixName(params) {
  if (params.kind === "padding") {
    return `padding each ${count(params.row_bytes)}-byte row by ${plural(params.pad_bytes, "byte")}`;
  }
  return `swizzling the byte offsets with Swizzle<${params.bits},${params.base},${params.shift}>`;
}

// What the top-ranked change does, such as "removes all 3,968 extra passes on this buffer".
export function fixEffect(item) {
  const before = item.excessive_before;
  const after = item.excessive_after;
  const parts = [
    after === 0
      ? `removes all ${count(before)} extra passes on this buffer`
      : `removes ${count(before - after)} of the ${count(before)} extra passes on this buffer`,
  ];
  if (item.extra_bytes) {
    parts.push(`adds ${bytes(item.extra_bytes)} of shared memory`);
  }
  const sectors = item.excessive_sectors_after - item.excessive_sectors_before;
  if (sectors > 0) {
    parts.push(`adds ${count(sectors)} excess sectors of global memory`);
  }
  return parts.length === 3 ? `${parts[0]}, ${parts[1]}, and ${parts[2]}` : parts.join(" and ");
}
