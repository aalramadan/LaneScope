import assert from "node:assert/strict";
import { test } from "node:test";

import * as format from "../../src/lanescope/gui/format.js";

test("degree steps follow the five bins of the report", () => {
  assert.deepEqual([1, 2, 3, 4, 5, 8, 9, 32].map(format.degreeStep), [0, 1, 2, 2, 3, 3, 4, 4]);
  assert.throws(() => format.degreeStep(0), RangeError);
  assert.equal(format.degreeLabel(2), "3 to 4");
  assert.equal(format.degreeLabel(0), "1");
});

test("lane ranges join neighboring lanes", () => {
  assert.equal(format.laneRanges([0, 1, 2, 3, 8, 10, 11]), "0-3, 8, 10-11");
  assert.equal(format.laneRanges([]), "none");
});

test("deltas treat a lower count as better and name the percent", () => {
  assert.deepEqual(format.delta(3968, 0), { direction: "better", text: "−3,968", percent: "−100%" });
  assert.deepEqual(format.delta(128, 160), { direction: "worse", text: "+32", percent: "+25%" });
  assert.deepEqual(format.delta(0, 0), { direction: "same", text: "0", percent: "0%" });
  assert.equal(format.delta(0, 4).percent, "new");
});

test("integers parse from decimal and hexadecimal text only", () => {
  assert.equal(format.parseInteger("0x120"), 288);
  assert.equal(format.parseInteger(" 4096 "), 4096);
  assert.ok(Number.isNaN(format.parseInteger("12px")));
  assert.ok(Number.isNaN(format.parseInteger("-4")));
});

test("plural forms follow the count", () => {
  assert.equal(format.plural(1, "wavefront"), "1 wavefront");
  assert.equal(format.plural(4096, "wavefront"), "4,096 wavefronts");
});

test("offset labels count bytes from the first line in hexadecimal", () => {
  assert.equal(format.offsetLabel(0), "+0x0");
  assert.equal(format.offsetLabel(3968), "+0xf80");
});

const SHARED_LOAD = { kind: "load", unit: "wavefronts", width: 4, line: 10, achieved: 4096, ideal: 128, excessive: 3968 };
const TOTALS = { wavefronts: { excessive: 3968 }, sectors: { excessive: 0 } };

function sharedAccess(degrees, achieved, ideal) {
  return { phases: [{ degrees }], achieved, ideal, excessive: achieved - ideal };
}

test("a bank conflict names its threads, its cost, and its part of the kernel", () => {
  const verdict = format.diagnosis(SHARED_LOAD, sharedAccess([32, 1, 1], 32, 1), TOTALS);
  assert.equal(verdict.tone, "warn");
  assert.equal(verdict.title, "Bank conflicts");
  assert.equal(verdict.badge, "32× the passes it needs");
  assert.match(verdict.cause, /up to 32 threads hit different words in the same bank/);
  assert.equal(
    verdict.cost,
    "Over the whole run, this load takes 4,096 passes where 128 would do. This line causes all the extra passes of the kernel.",
  );
  assert.equal(verdict.advice, "");
});

test("a warp without a conflict points to the other warps of the instruction", () => {
  const verdict = format.diagnosis(SHARED_LOAD, sharedAccess([1, 1], 1, 1), TOTALS);
  assert.match(verdict.cause, /^The warp shown has no conflict, but other warps of this load do\./);
});

test("an instruction without a line takes its part of the kernel's waste as an instruction", () => {
  const record = { ...SHARED_LOAD, line: null, kind: "store", achieved: 128, ideal: 32, excessive: 96 };
  const verdict = format.diagnosis(record, sharedAccess([4], 4, 1), { wavefronts: { excessive: 736 } });
  assert.match(verdict.cost, /this store takes 128 passes where 32 would do\. This instruction causes 13% of the extra passes/);
});

test("a scattered global load counts its sectors and advises neighboring addresses", () => {
  const record = { kind: "load", unit: "sectors", width: 4, line: 7, achieved: 2048, ideal: 256, excessive: 1792 };
  const access = { achieved: 32, ideal: 4, excessive: 28 };
  const verdict = format.diagnosis(record, access, { sectors: { excessive: 1792 } });
  assert.equal(verdict.title, "Wasted sectors");
  assert.equal(verdict.badge, "8× the sectors it needs");
  assert.match(verdict.cause, /spread over 32 sectors of 32 bytes, where 4 would hold their data/);
  assert.match(verdict.cost, /this load fetches 2,048 sectors where 256 would do/);
  assert.match(verdict.advice, /^Fix by letting neighboring threads touch neighboring addresses\./);
});

test("a clean instruction says whether the rest of the kernel is clean", () => {
  const record = { kind: "store", unit: "sectors", width: 4, line: 5, achieved: 512, ideal: 512, excessive: 0 };
  const access = { achieved: 4, ideal: 4, excessive: 0 };
  const clean = format.diagnosis(record, access, { sectors: { excessive: 0 }, wavefronts: { excessive: 0 } });
  assert.equal(clean.tone, "ok");
  assert.equal(clean.title, "No wasted sectors");
  assert.match(clean.cause, /sectors it writes\.$/);
  assert.equal(clean.cost, "No instruction of this kernel takes extra passes or sectors.");
  const elsewhere = format.diagnosis(record, access, TOTALS);
  assert.equal(elsewhere.cost, "The instructions with extra passes or sectors come first in the table.");
});

test("multiples and shares round to what a reader needs", () => {
  assert.equal(format.multiple(32), "32×");
  assert.equal(format.multiple(1.4), "1.4×");
  assert.equal(format.multiple(4), "4×");
  assert.equal(format.share(1), "all");
  assert.equal(format.share(0.13), "13%");
  assert.equal(format.share(0.001), "less than 1%");
});

test("a ranked fix reads as the object of a sentence with its effect", () => {
  assert.equal(format.fixName({ kind: "padding", row_bytes: 128, pad_bytes: 4 }), "padding each 128-byte row by 4 bytes");
  assert.equal(format.fixName({ kind: "swizzle", bits: 5, base: 2, shift: 5 }), "swizzling the byte offsets with Swizzle<5,2,5>");
  const item = { excessive_before: 192, excessive_after: 64, extra_bytes: 0, excessive_sectors_before: 0, excessive_sectors_after: 0 };
  assert.equal(format.fixEffect(item), "removes 128 of the 192 extra passes on this buffer");
  const padded = { ...item, excessive_after: 0, extra_bytes: 128, excessive_sectors_after: 8 };
  assert.equal(
    format.fixEffect(padded),
    "removes all 192 extra passes on this buffer, adds 128 B of shared memory, and adds 8 excess sectors of global memory",
  );
});
