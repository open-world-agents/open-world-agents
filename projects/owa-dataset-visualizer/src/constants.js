/**
 * Application constants and configuration
 * @module constants
 */

// ============================================================================
// Screen and Display
// ============================================================================

/** Default screen width for mouse coordinate normalization */
export const SCREEN_WIDTH = 1920;

/** Default screen height for mouse coordinate normalization */
export const SCREEN_HEIGHT = 1080;

/** Height of the overlay canvas in pixels */
export const OVERLAY_HEIGHT = 220;

// ============================================================================
// Keyboard Layout
// ============================================================================

/** Size of each key in pixels */
export const KEY_SIZE = 32;

/** Margin between keys in pixels */
export const KEY_MARGIN = 3;

/** Number of columns in the keyboard layout */
export const KEYBOARD_COLUMNS = 14;

// ============================================================================
// Mouse Button Mappings
// ============================================================================

/**
 * Maps Windows Virtual Key codes to mouse button names
 * Used in keyboard/state to filter out mouse buttons
 * @type {Object<number, string>}
 */
export const MOUSE_VK_MAP = {
  1: "left",
  2: "right",
  4: "middle",
  5: "x1",
  6: "x2",
};

/**
 * Raw input button press flags (from Windows RawInput API)
 * @type {Object<number, string>}
 */
export const BUTTON_PRESS_FLAGS = {
  0x0001: "left", // RI_MOUSE_LEFT_BUTTON_DOWN
  0x0004: "right", // RI_MOUSE_RIGHT_BUTTON_DOWN
  0x0010: "middle", // RI_MOUSE_MIDDLE_BUTTON_DOWN
};

/**
 * Raw input button release flags (from Windows RawInput API)
 * @type {Object<number, string>}
 */
export const BUTTON_RELEASE_FLAGS = {
  0x0002: "left", // RI_MOUSE_LEFT_BUTTON_UP
  0x0008: "right", // RI_MOUSE_RIGHT_BUTTON_UP
  0x0020: "middle", // RI_MOUSE_MIDDLE_BUTTON_UP
};

/**
 * Raw input mouse wheel flag
 * When set, button_data contains wheel delta as signed 16-bit
 */
export const RI_MOUSE_WHEEL = 0x0400;

/**
 * Raw input horizontal mouse wheel flag
 * When set, button_data contains horizontal wheel delta
 */
export const RI_MOUSE_HWHEEL = 0x0800;

// ============================================================================
// Timing
// ============================================================================

/** Duration to show wheel indicator in milliseconds */
export const WHEEL_DECAY_MS = 150;

// ============================================================================
// Colors
// ============================================================================

/**
 * Color palette for overlay rendering
 * @type {Object<string, string>}
 */
export const COLORS = {
  // Keyboard
  keyBackground: "#333",
  keyPressed: "#50b0ab",
  keyBorder: "#555",
  keyText: "#fff",

  // Mouse
  mouseBody: "#282828",
  mouseBorder: "#888",
  mouseInactive: "#444",
  mouseLeft: "#e74c3c",
  mouseRight: "#3498db",
  mouseMiddle: "#f1c40f",
  mouseWheel: "#2ecc71",

  // Minimap
  minimapBorder: "#fff",
  minimapCursor: "#0f0",
};

// ============================================================================
// Topics
// ============================================================================

/**
 * MCAP topic names
 * @type {Object<string, string>}
 */
export const TOPICS = {
  KEYBOARD: "keyboard",
  KEYBOARD_STATE: "keyboard/state",
  MOUSE: "mouse",
  MOUSE_RAW: "mouse/raw",
  MOUSE_STATE: "mouse/state",
  WINDOW: "window",
  SCREEN: "screen",
};
