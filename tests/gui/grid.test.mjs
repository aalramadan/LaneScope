import assert from "node:assert/strict";
import { test } from "node:test";

import {
  STEP,
  bankGridModel,
  sectorGridModel,
  sectorUse,
  swizzleBits,
  usedBytes,
} from "../../src/lanescope/gui/grid.js";

const lanes = Array.from({ length: 32 }, (_, lane) => lane);

function phaseOf(offsets) {
  const bankWords = Array.from({ length: 32 }, () => new Set());
  for (const offset of offsets) {
    bankWords[(offset / 4) % 32].add(offset / 4);
  }
  const words = bankWords.map((found) => [...found].sort((a, b) => a - b));
  const degrees = words.map((found) => found.length);
  return {
    lanes,
    wavefronts: Math.max(...degrees),
    worst_bank: degrees.indexOf(Math.max(...degrees)),
    degrees,
    cells: offsets.map((offset, lane) => {
      const word = offset / 4;
      const bank = word % 32;
      return [lane, bank, word, words[bank].indexOf(word)];
    }),
  };
}

// The bank grid of a 4-byte access, with the lane offsets as the server sends them.
function modelOf(offsets) {
  return bankGridModel(phaseOf(offsets), offsets.map((offset) => ({ offset })));
}

test("a column read puts every lane in bank 0 at the darkest step", () => {
  const model = modelOf(lanes.map((lane) => 128 * lane));
  assert.equal(model.peak, 32);
  assert.equal(model.worstBank, 0);
  assert.ok(model.cells.every((cell) => cell.bank === 0 && cell.step === 4));
  assert.deepEqual(
    model.cells.map((cell) => cell.slot),
    lanes,
  );
  assert.equal(model.bars.filter((bar) => bar.degree > 0).length, 1);
  assert.equal(model.cells[5].y - model.cells[4].y, STEP);
});

test("a row read spreads the lanes over every bank at the lightest step", () => {
  const model = modelOf(lanes.map((lane) => 4 * lane));
  assert.equal(model.peak, 1);
  assert.ok(model.cells.every((cell, lane) => cell.bank === lane && cell.step === 0));
  assert.ok(model.bars.every((bar) => bar.degree === 1));
});

test("used bytes count each distinct address once", () => {
  const all = lanes.map((lane) => ({ offset: 4 * lane }));
  assert.equal(usedBytes({ offset: 0, lanes: [0, 1, 2, 3, 4, 5, 6, 7] }, all, 4), 32);
  const broadcast = lanes.map(() => ({ offset: 8 }));
  assert.equal(usedBytes({ offset: 0, lanes }, broadcast, 4), 4);
});

test("sector columns leave a gap between lines and label each line", () => {
  const access = {
    width: 4,
    lanes: lanes.map((lane) => (lane < 2 ? { address: "0x0", offset: 128 * lane } : null)),
    sectors: [
      { sector: 0, offset: 0, lanes: [0] },
      { sector: 4, offset: 128, lanes: [1] },
    ],
  };
  const model = sectorGridModel(access);
  assert.deepEqual(
    model.lines.map((line) => line.label),
    ["+0x0", "+0x80"],
  );
  assert.ok(model.columns[1].x - model.columns[0].x > STEP);
  assert.equal(model.rows.filter((row) => row.active).length, 2);
  assert.equal(model.cells.length, 2);
});

test("swizzle bits mark the bits read above the bits flipped", () => {
  const bits = swizzleBits({ bits: 5, base: 2, shift: 5 });
  assert.equal(bits.length, 12);
  const roles = Object.fromEntries(bits.map(({ bit, role }) => [bit, role]));
  assert.deepEqual([0, 1].map((bit) => roles[bit]), ["none", "none"]);
  assert.ok([2, 3, 4, 5, 6].every((bit) => roles[bit] === "target"));
  assert.ok([7, 8, 9, 10, 11].every((bit) => roles[bit] === "source"));
  assert.equal(swizzleBits({ bits: 3, base: 4, shift: 9 }).length, 16);
});

test("narrow lines share their labels so the labels never overlap", () => {
  const access = {
    width: 4,
    lanes: lanes.map((lane) => ({ address: "0x0", offset: 4096 * lane })),
    sectors: lanes.map((lane) => ({ sector: 128 * lane, offset: 4096 * lane, lanes: [lane] })),
  };
  const model = sectorGridModel(access);
  const labeled = model.lines.filter((line) => line.labeled);
  const spans = labeled.map((line) => {
    const half = (line.label.length * 6.5) / 2;
    const center = (line.start + line.end) / 2;
    return [center - half, center + half];
  });
  assert.ok(labeled.length > 1 && labeled.length < 32);
  assert.ok(spans.slice(1).every(([left], index) => left >= spans[index][1]));
  assert.ok(spans.at(-1)[1] <= model.width);
  assert.deepEqual(sectorUse(access), { used: 128, fetched: 1024 });
});

test("labels of lines far apart keep clear of each other however long they grow", () => {
  const access = {
    width: 4,
    lanes: lanes.map((lane) => ({ address: "0x0", offset: lane * 2 ** 36 })),
    sectors: lanes.map((lane) => ({ sector: lane * 2 ** 31, offset: lane * 2 ** 36, lanes: [lane] })),
  };
  const model = sectorGridModel(access);
  const labeled = model.lines.filter((line) => line.labeled);
  const spans = labeled.map((line) => {
    const half = (line.label.length * 6.5) / 2;
    const center = (line.start + line.end) / 2;
    return [center - half, center + half];
  });
  assert.ok(spans.slice(1).every(([left], index) => left >= spans[index][1]));
  assert.ok(spans.at(-1)[1] <= model.width);
});

test("each cell of a wide load names the byte offset of its own word", () => {
  // Lane 31 reads 16 bytes at 0x1f00, one word in each of banks 0 to 3.
  const cells = [0, 1, 2, 3].map((bank) => [31, bank, 0x1f00 / 4 + bank, 0]);
  const degrees = Array.from({ length: 32 }, (_, bank) => (bank < 4 ? 1 : 0));
  const offsets = lanes.map((lane) => (lane === 31 ? { offset: 0x1f00 } : null));
  const model = bankGridModel({ lanes: [31], wavefronts: 1, worst_bank: 0, degrees, cells }, offsets);
  assert.deepEqual(
    model.cells.map((cell) => cell.offset),
    [0x1f00, 0x1f04, 0x1f08, 0x1f0c],
  );
});

test("each cell of a byte load names the byte that its lane reads", () => {
  // Lanes 0 to 3 read one byte each of the word at 0x40, which lies in bank 16.
  const cells = [0, 1, 2, 3].map((lane) => [lane, 16, 0x40 / 4, 0]);
  const degrees = Array.from({ length: 32 }, (_, bank) => (bank === 16 ? 1 : 0));
  const offsets = lanes.map((lane) => (lane < 4 ? { offset: 0x40 + lane } : null));
  const model = bankGridModel(
    { lanes: [0, 1, 2, 3], wavefronts: 1, worst_bank: 16, degrees, cells },
    offsets,
  );
  assert.deepEqual(
    model.cells.map((cell) => cell.offset),
    [0x40, 0x41, 0x42, 0x43],
  );
});

test("each sector cell names the offset of its sector from the first line", () => {
  const access = {
    width: 4,
    lanes: lanes.map((lane) => ({ address: "0x0", offset: 4096 * lane })),
    sectors: lanes.map((lane) => ({ sector: 128 * lane, offset: 4096 * lane, lanes: [lane] })),
  };
  const model = sectorGridModel(access);
  assert.equal(model.cells.at(-1).offset, 0x1f000);
  assert.equal(model.columns.at(-1).offset, 0x1f000);
  assert.equal(model.lines.at(-1).label, "+0x1f000");
});
