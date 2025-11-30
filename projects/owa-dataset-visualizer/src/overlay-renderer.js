import { KEYBOARD_LAYOUT, VK_ALIASES } from "./keyboard-layout.js";

const KEY_SIZE = 32;
const KEY_MARGIN = 3;
const COLORS = {
  bg: "#333",
  pressed: "#50b0ab",
  border: "#555",
  text: "#fff",
  mouseBody: "#282828",
  mouseBorder: "#888",
  mouseInactive: "#444",
  mouseLeft: "#e74c3c",
  mouseRight: "#3498db",
  mouseMiddle: "#f1c40f",
};

/**
 * Draw keyboard overlay on canvas
 */
export function drawKeyboard(ctx, x, y, pressedKeys) {
  // Expand pressedKeys with aliases (SHIFT -> LSHIFT/RSHIFT)
  const expanded = new Set(pressedKeys);
  for (const [generic, specifics] of Object.entries(VK_ALIASES)) {
    if (pressedKeys.has(Number(generic))) {
      specifics.forEach((vk) => expanded.add(vk));
    }
  }

  for (const [row, col, width, label, vk, isArrow] of KEYBOARD_LAYOUT) {
    const kx = x + col * (KEY_SIZE + KEY_MARGIN);
    const ky = y + row * (KEY_SIZE + KEY_MARGIN);
    const kw = width * (KEY_SIZE + KEY_MARGIN) - KEY_MARGIN;
    const kh = KEY_SIZE;

    const isPressed = expanded.has(vk);

    // Key background
    ctx.fillStyle = isPressed ? COLORS.pressed : COLORS.bg;
    ctx.fillRect(kx, ky, kw, kh);

    // Key border
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(kx, ky, kw, kh);

    // Label
    ctx.fillStyle = COLORS.text;
    if (isArrow) {
      drawArrow(ctx, kx + kw / 2, ky + kh / 2, label);
    } else {
      const fontSize = label.length <= 1 ? 14 : label.length <= 3 ? 10 : 8;
      ctx.font = `bold ${fontSize}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, kx + kw / 2, ky + kh / 2);
    }
  }
}

/**
 * Draw arrow symbol
 */
function drawArrow(ctx, cx, cy, direction) {
  const size = 8;
  ctx.beginPath();
  switch (direction) {
    case "UP":
      ctx.moveTo(cx, cy - size);
      ctx.lineTo(cx - size, cy + size / 2);
      ctx.lineTo(cx + size, cy + size / 2);
      break;
    case "DOWN":
      ctx.moveTo(cx, cy + size);
      ctx.lineTo(cx - size, cy - size / 2);
      ctx.lineTo(cx + size, cy - size / 2);
      break;
    case "LEFT":
      ctx.moveTo(cx - size, cy);
      ctx.lineTo(cx + size / 2, cy - size);
      ctx.lineTo(cx + size / 2, cy + size);
      break;
    case "RIGHT":
      ctx.moveTo(cx + size, cy);
      ctx.lineTo(cx - size / 2, cy - size);
      ctx.lineTo(cx - size / 2, cy + size);
      break;
  }
  ctx.closePath();
  ctx.fill();
}

/**
 * Draw mouse figure
 */
export function drawMouse(ctx, x, y, activeButtons) {
  const w = 60, h = 80;
  const cx = x + w / 2, cy = y + h / 2;
  const rx = w / 2, ry = h / 2;

  // Body
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.mouseBody;
  ctx.fill();
  ctx.strokeStyle = COLORS.mouseBorder;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Left button (top-left quadrant)
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.ellipse(cx, cy, rx, ry, 0, Math.PI, Math.PI * 1.5);
  ctx.closePath();
  ctx.fillStyle = activeButtons.has("left") ? COLORS.mouseLeft : COLORS.mouseInactive;
  ctx.fill();
  ctx.stroke();

  // Right button (top-right quadrant)
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.ellipse(cx, cy, rx, ry, 0, Math.PI * 1.5, Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = activeButtons.has("right") ? COLORS.mouseRight : COLORS.mouseInactive;
  ctx.fill();
  ctx.stroke();

  // Middle button (scroll wheel)
  const mw = 10, mh = 24;
  const mx = cx - mw / 2, my = y + 8;
  ctx.fillStyle = activeButtons.has("middle") ? COLORS.mouseMiddle : COLORS.mouseInactive;
  ctx.fillRect(mx, my, mw, mh);
  ctx.strokeRect(mx, my, mw, mh);

  // Scroll lines
  ctx.strokeStyle = COLORS.mouseBorder;
  ctx.lineWidth = 1;
  for (let i = 1; i <= 2; i++) {
    const ly = my + (i * mh) / 3;
    ctx.beginPath();
    ctx.moveTo(mx + 2, ly);
    ctx.lineTo(mx + mw - 2, ly);
    ctx.stroke();
  }
}

/**
 * Draw mouse position minimap
 */
export function drawMinimap(ctx, x, y, w, h, mouseX, mouseY, screenW, screenH, activeButtons) {
  // Border
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  // Cursor position
  const pad = 4;
  const px = x + pad + ((mouseX / screenW) * (w - 2 * pad));
  const py = y + pad + ((mouseY / screenH) * (h - 2 * pad));

  // Cursor dot
  ctx.beginPath();
  ctx.arc(px, py, 4, 0, Math.PI * 2);
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Click indicator
  if (activeButtons.size > 0) {
    const color = activeButtons.has("left") ? COLORS.mouseLeft :
                  activeButtons.has("right") ? COLORS.mouseRight : COLORS.mouseMiddle;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

