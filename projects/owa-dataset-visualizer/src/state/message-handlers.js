/**
 * Message handlers for each MCAP topic
 * @module state/message-handlers
 */

import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  MOUSE_VK_MAP,
  BUTTON_PRESS_FLAGS,
  BUTTON_RELEASE_FLAGS,
  RI_MOUSE_WHEEL,
  TOPICS,
} from "../constants.js";

/**
 * Handle keyboard/state topic message
 * @param {Object} state - Current state object
 * @param {Object} data - Message data
 */
export function handleKeyboardState(state, data) {
  const keyVks = (data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]);
  state.keyboard = new Set(keyVks);
}

/**
 * Handle keyboard topic message (individual key events)
 * @param {Object} state - Current state object
 * @param {Object} data - Message data with event_type and vk
 */
export function handleKeyboard(state, data) {
  if (MOUSE_VK_MAP[data.vk]) return; // Skip mouse buttons

  if (data.event_type === "press") {
    state.keyboard.add(data.vk);
  } else if (data.event_type === "release") {
    state.keyboard.delete(data.vk);
  }
}

/**
 * Handle mouse/raw topic message
 * @param {Object} state - Current state object
 * @param {Object} data - Raw mouse event data
 * @param {bigint} time - Message timestamp
 * @param {Object} options - Additional options
 * @param {number} options.recenterIntervalMs - Recenter interval in ms (0 = disabled)
 * @param {bigint} options.lastRecenterTime - Last recenter timestamp
 * @param {function} options.onRecenter - Callback when recenter occurs
 * @param {function} options.onWheel - Callback when wheel event occurs
 */
export function handleMouseRaw(state, data, time, options = {}) {
  const { recenterIntervalMs = 0, lastRecenterTime = 0n, onRecenter, onWheel } = options;

  // Recenter check
  if (recenterIntervalMs > 0) {
    const intervalNs = BigInt(recenterIntervalMs) * 1000000n;
    if (time - lastRecenterTime >= intervalNs) {
      state.mouse.x = SCREEN_WIDTH / 2;
      state.mouse.y = SCREEN_HEIGHT / 2;
      onRecenter?.(time);
    }
  }

  // Accumulate relative movement
  state.mouse.x = Math.max(0, Math.min(SCREEN_WIDTH - 1, state.mouse.x + (data.last_x ?? 0)));
  state.mouse.y = Math.max(0, Math.min(SCREEN_HEIGHT - 1, state.mouse.y + (data.last_y ?? 0)));

  // Button flags
  const flags = data.button_flags ?? 0;
  for (const [flag, btn] of Object.entries(BUTTON_PRESS_FLAGS)) {
    if (flags & Number(flag)) state.mouse.buttons.add(btn);
  }
  for (const [flag, btn] of Object.entries(BUTTON_RELEASE_FLAGS)) {
    if (flags & Number(flag)) state.mouse.buttons.delete(btn);
  }

  // Wheel events (button_data contains delta as signed 16-bit)
  if (flags & RI_MOUSE_WHEEL) {
    const delta = (data.button_data << 16) >> 16; // Sign extend
    state.mouse.wheel = delta > 0 ? 1 : -1;
    onWheel?.();
  }
}

/**
 * Handle mouse/state topic message (full state snapshot)
 * @param {Object} state - Current state object
 * @param {Object} data - Mouse state data
 */
export function handleMouseState(state, data) {
  state.mouse.x = data.x ?? state.mouse.x;
  state.mouse.y = data.y ?? state.mouse.y;
  state.mouse.buttons = new Set(data.buttons || []);
}

/**
 * Handle mouse topic message (individual mouse events)
 * @param {Object} state - Current state object
 * @param {Object} data - Mouse event data
 * @param {function} onWheel - Callback when wheel event occurs
 */
export function handleMouse(state, data, onWheel) {
  if (data.event_type === "move") {
    state.mouse.x = data.x ?? state.mouse.x;
    state.mouse.y = data.y ?? state.mouse.y;
  } else if (data.event_type === "click") {
    state.mouse.x = data.x ?? state.mouse.x;
    state.mouse.y = data.y ?? state.mouse.y;
    if (data.button) {
      if (data.pressed) state.mouse.buttons.add(data.button);
      else state.mouse.buttons.delete(data.button);
    }
  } else if (data.event_type === "scroll") {
    state.mouse.x = data.x ?? state.mouse.x;
    state.mouse.y = data.y ?? state.mouse.y;
    const dy = data.dy ?? 0;
    if (dy !== 0) {
      state.mouse.wheel = dy > 0 ? 1 : -1;
      onWheel?.();
    }
  }
}

/**
 * Handle window topic message
 * @param {Object} state - Current state object
 * @param {Object} data - Window data
 */
export function handleWindow(state, data) {
  state.window = data;
}

/**
 * Get the appropriate handler for a topic
 * @param {string} topic - Topic name
 * @returns {function|null} Handler function or null
 */
export function getHandler(topic) {
  switch (topic) {
    case TOPICS.KEYBOARD_STATE:
      return handleKeyboardState;
    case TOPICS.KEYBOARD:
      return handleKeyboard;
    case TOPICS.MOUSE_RAW:
      return handleMouseRaw;
    case TOPICS.MOUSE_STATE:
      return handleMouseState;
    case TOPICS.MOUSE:
      return handleMouse;
    case TOPICS.WINDOW:
      return handleWindow;
    default:
      return null;
  }
}

