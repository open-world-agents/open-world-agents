import assert from "node:assert/strict";
import test from "node:test";

import { StateManager } from "../src/state.js";
import { RI_MOUSE_WHEEL, SCREEN_HEIGHT, SCREEN_WIDTH, TOPICS, WHEEL_DECAY_MS } from "../src/constants.js";

test("StateManager recenters raw mouse input before applying deltas", () => {
  const manager = new StateManager();
  manager.state.mouse.x = 0;
  manager.state.mouse.y = 0;
  manager.recenterIntervalMs = 100;

  const time = 200n * 1000000n;
  manager.processMessage(TOPICS.MOUSE_RAW, { last_x: 10, last_y: 5 }, time);

  assert.equal(manager.lastRecenterTime, time);
  assert.equal(manager.state.mouse.x, SCREEN_WIDTH / 2 + 10);
  assert.equal(manager.state.mouse.y, SCREEN_HEIGHT / 2 + 5);
});

test("StateManager clears wheel indicator after decay window", () => {
  const manager = new StateManager();

  manager.processMessage(
    TOPICS.MOUSE_RAW,
    { button_flags: RI_MOUSE_WHEEL, button_data: 120 },
    0n,
  );
  assert.equal(manager.state.mouse.wheel, 1);

  manager.lastWheelTime = performance.now() - WHEEL_DECAY_MS - 1;
  manager.decayWheel();
  assert.equal(manager.state.mouse.wheel, 0);
});

test("StateManager update topics switch with mouse mode", () => {
  const manager = new StateManager();
  manager.mouseMode = "absolute";

  const topics = manager.getUpdateTopics();
  assert.ok(topics.includes(TOPICS.MOUSE));
  assert.ok(!topics.includes(TOPICS.MOUSE_RAW));
});
