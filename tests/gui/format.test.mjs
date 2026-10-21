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
