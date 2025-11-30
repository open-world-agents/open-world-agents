/**
 * State management for the visualizer
 * @module state
 */

import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  MOUSE_VK_MAP,
  WHEEL_DECAY_MS,
  TOPICS,
} from "../constants.js";
import {
  handleKeyboardState,
  handleKeyboard,
  handleMouseRaw,
  handleMouseState,
  handleMouse,
  handleWindow,
} from "./message-handlers.js";

/**
 * Creates a fresh initial state object
 * @returns {Object} Initial state
 */
function createInitialState() {
  return {
    keyboard: new Set(),
    mouse: {
      x: SCREEN_WIDTH / 2,
      y: SCREEN_HEIGHT / 2,
      buttons: new Set(),
      wheel: 0,
    },
    window: null,
  };
}

/**
 * Manages application state and message processing
 */
export class StateManager {
  constructor() {
    /** @type {Object} Current input state */
    this.state = createInitialState();

    /** @type {"raw"|"absolute"} Mouse input mode */
    this.mouseMode = "raw";

    /** @type {number} Recenter interval in milliseconds (0 = disabled) */
    this.recenterIntervalMs = 0;

    /** @type {bigint} Last recenter timestamp */
    this.lastRecenterTime = 0n;

    /** @type {bigint} Last processed message timestamp */
    this.lastProcessedTime = 0n;

    /** @type {number} Last wheel event time (performance.now()) */
    this.lastWheelTime = 0;

    /** @type {boolean} Whether state is currently being loaded */
    this.isLoading = false;
  }

  /**
   * Reset state to initial values
   * @param {bigint} [recenterTime=0n] - Time to set as last recenter time
   */
  reset(recenterTime = 0n) {
    this.state = createInitialState();
    this.lastRecenterTime = recenterTime;
  }

  /**
   * Process a single message and update state
   * @param {string} topic - Message topic
   * @param {Object} data - Message data
   * @param {bigint} time - Message timestamp
   */
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
        handleWindow(this.state, data);
        break;
    }
  }

  /**
   * Decay wheel indicator if enough time has passed
   */
  decayWheel() {
    if (this.state.mouse.wheel !== 0 && performance.now() - this.lastWheelTime > WHEEL_DECAY_MS) {
      this.state.mouse.wheel = 0;
    }
  }

  /**
   * Get the mouse topic based on current mode
   * @returns {string} Topic name
   */
  getMouseTopic() {
    return this.mouseMode === "raw" ? TOPICS.MOUSE_RAW : TOPICS.MOUSE;
  }

  /**
   * Get list of topics to subscribe to for updates
   * @returns {string[]} Array of topic names
   */
  getUpdateTopics() {
    return [
      TOPICS.KEYBOARD_STATE,
      TOPICS.KEYBOARD,
      TOPICS.MOUSE_STATE,
      this.getMouseTopic(),
      TOPICS.WINDOW,
    ];
  }

  /**
   * Apply keyboard state from keyboard/state message
   * Filters out mouse button VKs
   * @param {Object} data - keyboard/state message data
   */
  applyKeyboardState(data) {
    const keyVks = (data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]);
    this.state.keyboard = new Set(keyVks);
  }

  /**
   * Apply mouse state from mouse/state message
   * @param {Object} data - mouse/state message data
   */
  applyMouseState(data) {
    this.state.mouse.x = data.x ?? SCREEN_WIDTH / 2;
    this.state.mouse.y = data.y ?? SCREEN_HEIGHT / 2;
    this.state.mouse.buttons = new Set(data.buttons || []);
  }
}

