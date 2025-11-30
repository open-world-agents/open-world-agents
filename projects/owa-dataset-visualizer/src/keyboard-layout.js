/**
 * Keyboard layout matching convert_overlay.py
 * Format: [row, col, width, label, vkCode, isArrow]
 */
export const KEYBOARD_LAYOUT = [
  // Row 0: Function keys
  [0, 0, 1, "ESC", 0x1B, false],
  [0, 1, 1, "F1", 0x70, false], [0, 2, 1, "F2", 0x71, false],
  [0, 3, 1, "F3", 0x72, false], [0, 4, 1, "F4", 0x73, false],
  [0, 5, 1, "F5", 0x74, false], [0, 6, 1, "F6", 0x75, false],
  [0, 7, 1, "F7", 0x76, false], [0, 8, 1, "F8", 0x77, false],
  [0, 9, 1, "F9", 0x78, false], [0, 10, 1, "F10", 0x79, false],
  [0, 11, 1, "F11", 0x7A, false], [0, 12, 1, "F12", 0x7B, false],
  [0, 13, 1, "BACK", 0x08, false],

  // Row 1: Numbers
  [1, 0, 1, "~", 0xC0, false],
  [1, 1, 1, "1", 0x31, false], [1, 2, 1, "2", 0x32, false],
  [1, 3, 1, "3", 0x33, false], [1, 4, 1, "4", 0x34, false],
  [1, 5, 1, "5", 0x35, false], [1, 6, 1, "6", 0x36, false],
  [1, 7, 1, "7", 0x37, false], [1, 8, 1, "8", 0x38, false],
  [1, 9, 1, "9", 0x39, false], [1, 10, 1, "0", 0x30, false],
  [1, 11, 1, "-", 0xBD, false], [1, 12, 1, "=", 0xBB, false],
  [1, 13, 1, "\\", 0xDC, false],

  // Row 2: QWERTY
  [2, 0, 1, "TAB", 0x09, false],
  [2, 1, 1, "Q", 0x51, false], [2, 2, 1, "W", 0x57, false],
  [2, 3, 1, "E", 0x45, false], [2, 4, 1, "R", 0x52, false],
  [2, 5, 1, "T", 0x54, false], [2, 6, 1, "Y", 0x59, false],
  [2, 7, 1, "U", 0x55, false], [2, 8, 1, "I", 0x49, false],
  [2, 9, 1, "O", 0x4F, false], [2, 10, 1, "P", 0x50, false],
  [2, 11, 1, "[", 0xDB, false], [2, 12, 1, "]", 0xDD, false],
  [2, 13, 1, "ENT", 0x0D, false],

  // Row 3: Home row
  [3, 0, 1, "CAPS", 0x14, false],
  [3, 1, 1, "A", 0x41, false], [3, 2, 1, "S", 0x53, false],
  [3, 3, 1, "D", 0x44, false], [3, 4, 1, "F", 0x46, false],
  [3, 5, 1, "G", 0x47, false], [3, 6, 1, "H", 0x48, false],
  [3, 7, 1, "J", 0x4A, false], [3, 8, 1, "K", 0x4B, false],
  [3, 9, 1, "L", 0x4C, false], [3, 10, 1, ";", 0xBA, false],
  [3, 11, 1, "'", 0xDE, false],
  [3, 12, 1, "UP", 0x26, true],
  [3, 13, 1, "SHFT", 0xA1, false], // RSHIFT

  // Row 4: Bottom row
  [4, 0, 1, "SHFT", 0xA0, false], // LSHIFT
  [4, 1, 1, "Z", 0x5A, false], [4, 2, 1, "X", 0x58, false],
  [4, 3, 1, "C", 0x43, false], [4, 4, 1, "V", 0x56, false],
  [4, 5, 1, "B", 0x42, false], [4, 6, 1, "N", 0x4E, false],
  [4, 7, 1, "M", 0x4D, false], [4, 8, 1, ",", 0xBC, false],
  [4, 9, 1, ".", 0xBE, false], [4, 10, 1, "/", 0xBF, false],
  [4, 11, 1, "LEFT", 0x25, true],
  [4, 12, 1, "DOWN", 0x28, true],
  [4, 13, 1, "RIGHT", 0x27, true],

  // Row 5: Modifier row
  [5, 0, 1, "CTRL", 0xA2, false], // LCTRL
  [5, 1, 1, "WIN", 0x5B, false],
  [5, 2, 1, "ALT", 0xA4, false], // LALT
  [5, 3, 8, "SPACE", 0x20, false],
  [5, 11, 1, "ALT", 0xA5, false], // RALT
  [5, 12, 1, "WIN", 0x5C, false],
  [5, 13, 1, "CTRL", 0xA3, false], // RCTRL
];

// Map generic VK to both L/R versions
export const VK_ALIASES = {
  0x10: [0xA0, 0xA1], // SHIFT -> LSHIFT, RSHIFT
  0x11: [0xA2, 0xA3], // CTRL -> LCTRL, RCTRL
  0x12: [0xA4, 0xA5], // ALT -> LALT, RALT
};

