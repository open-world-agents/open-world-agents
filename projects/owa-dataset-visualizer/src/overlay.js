/**
 * Overlay rendering for keyboard and mouse visualization
 */
import { KEY_SIZE, KEY_MARGIN, COLORS } from "./constants.js";

// Keyboard layout: [row, col, width, label, vkCode, isArrow]
const KEYBOARD_LAYOUT = [
  [0, 0, 1, "ESC", 0x1b, false],
  [0, 1, 1, "F1", 0x70, false],
  [0, 2, 1, "F2", 0x71, false],
  [0, 3, 1, "F3", 0x72, false],
  [0, 4, 1, "F4", 0x73, false],
  [0, 5, 1, "F5", 0x74, false],
  [0, 6, 1, "F6", 0x75, false],
  [0, 7, 1, "F7", 0x76, false],
  [0, 8, 1, "F8", 0x77, false],
  [0, 9, 1, "F9", 0x78, false],
  [0, 10, 1, "F10", 0x79, false],
  [0, 11, 1, "F11", 0x7a, false],
  [0, 12, 1, "F12", 0x7b, false],
  [0, 13, 1, "BACK", 0x08, false],
  [1, 0, 1, "~", 0xc0, false],
  [1, 1, 1, "1", 0x31, false],
  [1, 2, 1, "2", 0x32, false],
  [1, 3, 1, "3", 0x33, false],
  [1, 4, 1, "4", 0x34, false],
  [1, 5, 1, "5", 0x35, false],
  [1, 6, 1, "6", 0x36, false],
  [1, 7, 1, "7", 0x37, false],
  [1, 8, 1, "8", 0x38, false],
  [1, 9, 1, "9", 0x39, false],
  [1, 10, 1, "0", 0x30, false],
  [1, 11, 1, "-", 0xbd, false],
  [1, 12, 1, "=", 0xbb, false],
  [1, 13, 1, "\\", 0xdc, false],
  [2, 0, 1, "TAB", 0x09, false],
  [2, 1, 1, "Q", 0x51, false],
  [2, 2, 1, "W", 0x57, false],
  [2, 3, 1, "E", 0x45, false],
  [2, 4, 1, "R", 0x52, false],
  [2, 5, 1, "T", 0x54, false],
  [2, 6, 1, "Y", 0x59, false],
  [2, 7, 1, "U", 0x55, false],
  [2, 8, 1, "I", 0x49, false],
  [2, 9, 1, "O", 0x4f, false],
  [2, 10, 1, "P", 0x50, false],
  [2, 11, 1, "[", 0xdb, false],
  [2, 12, 1, "]", 0xdd, false],
  [2, 13, 1, "ENT", 0x0d, false],
  [3, 0, 1, "CAPS", 0x14, false],
  [3, 1, 1, "A", 0x41, false],
  [3, 2, 1, "S", 0x53, false],
  [3, 3, 1, "D", 0x44, false],
  [3, 4, 1, "F", 0x46, false],
  [3, 5, 1, "G", 0x47, false],
  [3, 6, 1, "H", 0x48, false],
  [3, 7, 1, "J", 0x4a, false],
  [3, 8, 1, "K", 0x4b, false],
  [3, 9, 1, "L", 0x4c, false],
  [3, 10, 1, ";", 0xba, false],
  [3, 11, 1, "'", 0xde, false],
  [3, 12, 1, "UP", 0x26, true],
  [3, 13, 1, "SHFT", 0xa1, false],
  [4, 0, 1, "SHFT", 0xa0, false],
  [4, 1, 1, "Z", 0x5a, false],
  [4, 2, 1, "X", 0x58, false],
  [4, 3, 1, "C", 0x43, false],
  [4, 4, 1, "V", 0x56, false],
  [4, 5, 1, "B", 0x42, false],
  [4, 6, 1, "N", 0x4e, false],
  [4, 7, 1, "M", 0x4d, false],
  [4, 8, 1, ",", 0xbc, false],
  [4, 9, 1, ".", 0xbe, false],
  [4, 10, 1, "/", 0xbf, false],
  [4, 11, 1, "LEFT", 0x25, true],
  [4, 12, 1, "DOWN", 0x28, true],
  [4, 13, 1, "RIGHT", 0x27, true],
  [5, 0, 1, "CTRL", 0xa2, false],
  [5, 1, 1, "WIN", 0x5b, false],
  [5, 2, 1, "ALT", 0xa4, false],
  [5, 3, 8, "SPACE", 0x20, false],
  [5, 11, 1, "ALT", 0xa5, false],
  [5, 12, 1, "WIN", 0x5c, false],
  [5, 13, 1, "CTRL", 0xa3, false],
];

const VK_ALIASES = {
  0x10: [0xa0, 0xa1], // SHIFT -> LSHIFT, RSHIFT
  0x11: [0xa2, 0xa3], // CTRL -> LCTRL, RCTRL
  0x12: [0xa4, 0xa5], // ALT -> LALT, RALT
};

function drawArrow(ctx, cx, cy, dir) {
  const s = 8;
  ctx.beginPath();
  if (dir === "UP") {
    ctx.moveTo(cx, cy - s);
    ctx.lineTo(cx - s, cy + s / 2);
    ctx.lineTo(cx + s, cy + s / 2);
  } else if (dir === "DOWN") {
    ctx.moveTo(cx, cy + s);
    ctx.lineTo(cx - s, cy - s / 2);
    ctx.lineTo(cx + s, cy - s / 2);
  } else if (dir === "LEFT") {
    ctx.moveTo(cx - s, cy);
    ctx.lineTo(cx + s / 2, cy - s);
    ctx.lineTo(cx + s / 2, cy + s);
  } else if (dir === "RIGHT") {
    ctx.moveTo(cx + s, cy);
    ctx.lineTo(cx - s / 2, cy - s);
    ctx.lineTo(cx - s / 2, cy + s);
  }
  ctx.closePath();
  ctx.fill();
}

export function drawKeyboard(ctx, x, y, pressedKeys) {
  const expanded = new Set(pressedKeys);
  for (const [generic, specifics] of Object.entries(VK_ALIASES)) {
    if (pressedKeys.has(Number(generic))) specifics.forEach((vk) => expanded.add(vk));
  }

  for (const [row, col, width, label, vk, isArrow] of KEYBOARD_LAYOUT) {
    const kx = x + col * (KEY_SIZE + KEY_MARGIN);
    const ky = y + row * (KEY_SIZE + KEY_MARGIN);
    const kw = width * (KEY_SIZE + KEY_MARGIN) - KEY_MARGIN;
    const isPressed = expanded.has(vk);

    ctx.fillStyle = isPressed ? COLORS.keyPressed : COLORS.keyBackground;
    ctx.fillRect(kx, ky, kw, KEY_SIZE);
    ctx.strokeStyle = COLORS.keyBorder;
    ctx.lineWidth = 1;
    ctx.strokeRect(kx, ky, kw, KEY_SIZE);

    ctx.fillStyle = COLORS.keyText;
    if (isArrow) {
      drawArrow(ctx, kx + kw / 2, ky + KEY_SIZE / 2, label);
    } else {
      ctx.font = `bold ${label.length <= 1 ? 14 : label.length <= 3 ? 10 : 8}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, kx + kw / 2, ky + KEY_SIZE / 2);
    }
  }
}

export function drawMouse(ctx, x, y, activeButtons, wheelDir = 0) {
  const w = 60,
    h = 80,
    cx = x + w / 2,
    cy = y + h / 2,
    rx = w / 2,
    ry = h / 2;

  // Body
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.mouseBody;
  ctx.fill();
  ctx.strokeStyle = COLORS.mouseBorder;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Left button
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.ellipse(cx, cy, rx, ry, 0, Math.PI, Math.PI * 1.5);
  ctx.closePath();
  ctx.fillStyle = activeButtons.has("left") ? COLORS.mouseLeft : COLORS.mouseInactive;
  ctx.fill();
  ctx.stroke();

  // Right button
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.ellipse(cx, cy, rx, ry, 0, Math.PI * 1.5, Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = activeButtons.has("right") ? COLORS.mouseRight : COLORS.mouseInactive;
  ctx.fill();
  ctx.stroke();

  // Wheel
  const mw = 10,
    mh = 24,
    mx = cx - mw / 2,
    my = y + 8;
  ctx.fillStyle =
    wheelDir > 0 ? COLORS.mouseWheel : activeButtons.has("middle") ? COLORS.mouseMiddle : COLORS.mouseInactive;
  ctx.fillRect(mx, my, mw, mh / 2);
  ctx.fillStyle =
    wheelDir < 0 ? COLORS.mouseWheel : activeButtons.has("middle") ? COLORS.mouseMiddle : COLORS.mouseInactive;
  ctx.fillRect(mx, my + mh / 2, mw, mh / 2);
  ctx.strokeStyle = COLORS.mouseBorder;
  ctx.lineWidth = 1;
  ctx.strokeRect(mx, my, mw, mh);
  ctx.beginPath();
  ctx.moveTo(mx, my + mh / 2);
  ctx.lineTo(mx + mw, my + mh / 2);
  ctx.stroke();
}

export function drawMinimap(ctx, x, y, w, h, mouseX, mouseY, screenW, screenH, activeButtons) {
  ctx.strokeStyle = COLORS.minimapBorder;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  const pad = 4;
  const px = x + pad + (mouseX / screenW) * (w - 2 * pad);
  const py = y + pad + (mouseY / screenH) * (h - 2 * pad);

  ctx.beginPath();
  ctx.arc(px, py, 4, 0, Math.PI * 2);
  ctx.strokeStyle = COLORS.minimapCursor;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (activeButtons.size > 0) {
    const color = activeButtons.has("left")
      ? COLORS.mouseLeft
      : activeButtons.has("right")
        ? COLORS.mouseRight
        : COLORS.mouseMiddle;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
