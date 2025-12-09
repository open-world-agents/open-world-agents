/**
 * State management and message handling
 */
import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  MOUSE_VK_MAP,
  WHEEL_DECAY_MS,
  BUTTON_PRESS_FLAGS,
  BUTTON_RELEASE_FLAGS,
  RI_MOUSE_WHEEL,
  TOPICS,
} from "./constants.js";

function createInitialState() {
  return {
    keyboard: new Set(),
    mouse: { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 2, buttons: new Set(), wheel: 0 },
    window: null,
  };
}

// Message handlers
function handleKeyboardState(state, data) {
  state.keyboard = new Set((data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]));
}

function handleKeyboard(state, data) {
  if (MOUSE_VK_MAP[data.vk]) return;
  if (data.event_type === "press") state.keyboard.add(data.vk);
  else if (data.event_type === "release") state.keyboard.delete(data.vk);
}

function handleMouseRaw(state, data, time, opts = {}) {
  const { recenterIntervalMs = 0, lastRecenterTime = 0n, onRecenter, onWheel } = opts;

  if (recenterIntervalMs > 0 && time - lastRecenterTime >= BigInt(recenterIntervalMs) * 1000000n) {
    state.mouse.x = SCREEN_WIDTH / 2;
    state.mouse.y = SCREEN_HEIGHT / 2;
    onRecenter?.(time);
  }

  state.mouse.x = Math.max(0, Math.min(SCREEN_WIDTH - 1, state.mouse.x + (data.last_x ?? 0)));
  state.mouse.y = Math.max(0, Math.min(SCREEN_HEIGHT - 1, state.mouse.y + (data.last_y ?? 0)));

  const flags = data.button_flags ?? 0;
  for (const [f, btn] of Object.entries(BUTTON_PRESS_FLAGS)) if (flags & Number(f)) state.mouse.buttons.add(btn);
  for (const [f, btn] of Object.entries(BUTTON_RELEASE_FLAGS)) if (flags & Number(f)) state.mouse.buttons.delete(btn);

  if (flags & RI_MOUSE_WHEEL) {
    const delta = (data.button_data << 16) >> 16;
    if (delta !== 0) {
      state.mouse.wheel = Math.sign(delta);
      onWheel?.();
    }
  }
}

function handleMouseState(state, data) {
  state.mouse.x = data.x ?? state.mouse.x;
  state.mouse.y = data.y ?? state.mouse.y;
  state.mouse.buttons = new Set(data.buttons || []);
}

function handleMouse(state, data, onWheel) {
  state.mouse.x = data.x ?? state.mouse.x;
  state.mouse.y = data.y ?? state.mouse.y;

  if (data.event_type === "click" && data.button) {
    if (data.pressed) state.mouse.buttons.add(data.button);
    else state.mouse.buttons.delete(data.button);
  } else if (data.event_type === "scroll") {
    const dy = data.dy ?? 0;
    if (dy !== 0) {
      state.mouse.wheel = dy > 0 ? 1 : -1;
      onWheel?.();
    }
  }
}

export class StateManager {
  constructor() {
    this.state = createInitialState();
    this.mouseMode = "raw"; // "raw" | "absolute"
    this.recenterIntervalMs = 0;
    this.lastRecenterTime = 0n; // bigint
    this.lastProcessedTime = 0n; // bigint
    this.lastWheelTime = 0;
    this.isLoading = false;
  }

  /** @param {bigint} recenterTime */
  reset(recenterTime = 0n) {
    this.state = createInitialState();
    this.lastRecenterTime = recenterTime;
  }

  /** @param {string} topic @param {Object} data @param {bigint} time */
  processMessage(topic, data, time) {
    const onWheel = () => {
      this.lastWheelTime = performance.now();
    };
    const onRecenter = (t) => {
      this.lastRecenterTime = t;
    };

    switch (topic) {
      case TOPICS.KEYBOARD_STATE:
        handleKeyboardState(this.state, data);
        break;
      case TOPICS.KEYBOARD:
        handleKeyboard(this.state, data);
        break;
      case TOPICS.MOUSE_RAW:
        handleMouseRaw(this.state, data, time, {
          recenterIntervalMs: this.recenterIntervalMs,
          lastRecenterTime: this.lastRecenterTime,
          onRecenter,
          onWheel,
        });
        break;
      case TOPICS.MOUSE_STATE:
        handleMouseState(this.state, data);
        break;
      case TOPICS.MOUSE:
        handleMouse(this.state, data, onWheel);
        break;
      case TOPICS.WINDOW:
        this.state.window = data;
        break;
    }
  }

  decayWheel() {
    if (this.state.mouse.wheel !== 0 && performance.now() - this.lastWheelTime > WHEEL_DECAY_MS) {
      this.state.mouse.wheel = 0;
    }
  }

  getMouseTopic() {
    return this.mouseMode === "raw" ? TOPICS.MOUSE_RAW : TOPICS.MOUSE;
  }

  getUpdateTopics() {
    return [TOPICS.KEYBOARD_STATE, TOPICS.KEYBOARD, TOPICS.MOUSE_STATE, this.getMouseTopic(), TOPICS.WINDOW];
  }

  applyKeyboardState(data) {
    this.state.keyboard = new Set((data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]));
  }

  applyMouseState(data) {
    this.state.mouse.x = data.x ?? SCREEN_WIDTH / 2;
    this.state.mouse.y = data.y ?? SCREEN_HEIGHT / 2;
    this.state.mouse.buttons = new Set(data.buttons || []);
  }

  applyWindowState(data) {
    this.state.window = data;
  }
}
