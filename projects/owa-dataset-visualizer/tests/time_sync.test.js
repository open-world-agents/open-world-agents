import assert from "node:assert/strict";
import test from "node:test";

import { TimeSync } from "../src/mcap.js";

test("TimeSync uses screen message pts to compute base time", () => {
  const sync = new TimeSync();
  sync.initFromScreenMessage(5000n, { media_ref: { pts_ns: 1000 } });

  assert.equal(sync.getBasePtsTime(), 4000n);
  assert.equal(sync.videoTimeToMcap(0), 4000n);
  assert.equal(sync.videoTimeToMcap(0.001), 4000n + 1000000n);
});

test("TimeSync returns zero when base time is unset", () => {
  const sync = new TimeSync();
  assert.equal(sync.videoTimeToMcap(1), 0n);
});
