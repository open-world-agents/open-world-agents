import { loadMcap, loadMcapFromUrl } from "./mcap-loader.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay-renderer.js";

// Constants
const SCREEN_WIDTH = 1920;
const SCREEN_HEIGHT = 1080;
const OVERLAY_HEIGHT = 220;
const MOUSE_VK_MAP = { 1: "left", 2: "right", 4: "middle", 5: "x1", 6: "x2" };

// Mouse button flags
const BUTTON_PRESS_FLAGS = { 0x0001: "left", 0x0004: "right", 0x0010: "middle" };
const BUTTON_RELEASE_FLAGS = { 0x0002: "left", 0x0008: "right", 0x0020: "middle" };

// DOM Elements
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const timeInfo = document.querySelector("#time-info span");
const recenterInput = document.getElementById("recenter-interval");

// State
let mcapReader = null;
let basePtsTime = null;
let mouseMode = "raw";
let recenterIntervalMs = 0;
let lastRecenterTime = 0n;
let lastProcessedTime = 0n;
let isLoading = false;
let userWantsToPlay = false;
let currentState = {
  keyboard: new Set(),
  mouse: { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 2, buttons: new Set() },
};

// Convert video time to MCAP timestamp
function videoTimeToMcap(videoTimeSec) {
  if (basePtsTime === null) return 0n;
  return basePtsTime + BigInt(Math.floor(videoTimeSec * 1e9));
}

// Loading indicator
const loadingIndicator = document.getElementById("loading-indicator");

function showLoading() {
  loadingIndicator?.classList.remove("hidden");
}

function hideLoading() {
  loadingIndicator?.classList.add("hidden");
}

// Load state at a specific time (finds last keyboard/state and mouse before targetTime)
async function loadStateAt(targetTime) {
  if (!mcapReader) return;

  isLoading = true;
  video.pause();
  showLoading();

  // Reset state
  currentState = {
    keyboard: new Set(),
    mouse: { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 2, buttons: new Set() },
  };
  lastRecenterTime = targetTime;

  // Step 1: Find last keyboard/state before targetTime (reverse iteration)
  let keyboardStateTime = 0n;
  for await (const msg of mcapReader.readMessages({
    endTime: targetTime,
    topics: ["keyboard/state"],
    reverse: true,
  })) {
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    // Extract keyboard keys (excluding mouse VKs)
    const keyVks = (data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]);
    currentState.keyboard = new Set(keyVks);
    keyboardStateTime = msg.logTime;
    break; // Only need the last one
  }

  // Step 2: Process all keyboard events from keyboardStateTime to targetTime
  if (keyboardStateTime > 0n) {
    for await (const msg of mcapReader.readMessages({
      startTime: keyboardStateTime + 1n, // Exclude the state message itself
      endTime: targetTime,
      topics: ["keyboard"],
    })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      if (MOUSE_VK_MAP[data.vk]) continue; // Skip mouse buttons
      if (data.pressed) {
        currentState.keyboard.add(data.vk);
      } else {
        currentState.keyboard.delete(data.vk);
      }
    }
  }

  // Step 3: Find last mouse/state before targetTime
  let mouseStateTime = 0n;
  for await (const msg of mcapReader.readMessages({
    endTime: targetTime,
    topics: ["mouse/state"],
    reverse: true,
  })) {
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    // mouse/state has x, y (absolute position) and buttons (Set of strings like "left", "right")
    currentState.mouse.x = data.x ?? SCREEN_WIDTH / 2;
    currentState.mouse.y = data.y ?? SCREEN_HEIGHT / 2;
    currentState.mouse.buttons = new Set(data.buttons || []);
    mouseStateTime = msg.logTime;
    break; // Only need the last one
  }

  // Step 4: Process mouse events from mouseStateTime to targetTime
  const mouseTopic = mouseMode === "raw" ? "mouse/raw" : "mouse";
  if (mouseStateTime > 0n) {
    for await (const msg of mcapReader.readMessages({
      startTime: mouseStateTime + 1n,
      endTime: targetTime,
      topics: [mouseTopic],
    })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      processMessage(mouseTopic, data, msg.logTime);
    }
  } else if (mouseMode === "absolute") {
    // Fallback: find last mouse position if no mouse/state found
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime,
      topics: ["mouse"],
      reverse: true,
    })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      currentState.mouse.x = data.x ?? SCREEN_WIDTH / 2;
      currentState.mouse.y = data.y ?? SCREEN_HEIGHT / 2;
      break;
    }
  }

  lastProcessedTime = targetTime;
  isLoading = false;
  hideLoading();

  // Resume if user wants to play
  if (userWantsToPlay) {
    video.play();
  }
}

// Process a single message and update state
function processMessage(topic, data, time) {
  if (topic === "keyboard/state") {
    const keyVks = (data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]);
    currentState.keyboard = new Set(keyVks);
  } else if (topic === "mouse/raw") {
    // Recenter check
    if (recenterIntervalMs > 0) {
      const intervalNs = BigInt(recenterIntervalMs) * 1000000n;
      if (time - lastRecenterTime >= intervalNs) {
        currentState.mouse.x = SCREEN_WIDTH / 2;
        currentState.mouse.y = SCREEN_HEIGHT / 2;
        lastRecenterTime = time;
      }
    }
    // Accumulate relative movement
    currentState.mouse.x = Math.max(0, Math.min(SCREEN_WIDTH - 1, currentState.mouse.x + (data.last_x ?? 0)));
    currentState.mouse.y = Math.max(0, Math.min(SCREEN_HEIGHT - 1, currentState.mouse.y + (data.last_y ?? 0)));
    // Button flags
    const flags = data.button_flags ?? 0;
    for (const [flag, btn] of Object.entries(BUTTON_PRESS_FLAGS)) {
      if (flags & Number(flag)) currentState.mouse.buttons.add(btn);
    }
    for (const [flag, btn] of Object.entries(BUTTON_RELEASE_FLAGS)) {
      if (flags & Number(flag)) currentState.mouse.buttons.delete(btn);
    }
  } else if (topic === "mouse") {
    currentState.mouse.x = data.x ?? currentState.mouse.x;
    currentState.mouse.y = data.y ?? currentState.mouse.y;
    if (data.event_type === "click" && data.button) {
      if (data.pressed) currentState.mouse.buttons.add(data.button);
      else currentState.mouse.buttons.delete(data.button);
    }
  }
}

// Load and process messages from lastProcessedTime to targetTime
async function updateStateUpTo(targetTime) {
  // Skip if loading (loadStateAt is handling state) or no new messages needed
  if (!mcapReader || isLoading || targetTime <= lastProcessedTime) return;

  const mouseTopic = mouseMode === "raw" ? "mouse/raw" : "mouse";

  for await (const msg of mcapReader.readMessages({
    startTime: lastProcessedTime,
    endTime: targetTime,
    topics: ["keyboard/state", mouseTopic],
  })) {
    // Double-check isLoading in case loadStateAt started during iteration
    if (isLoading) return;
    const channel = mcapReader.channelsById.get(msg.channelId);
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    processMessage(channel.topic, data, msg.logTime);
  }

  if (!isLoading) {
    lastProcessedTime = targetTime;
  }
}

// Render loop
function startRenderLoop() {
  const ctx = overlay.getContext("2d");

  async function render() {
    const mcapTime = videoTimeToMcap(video.currentTime);

    // Update state up to current time
    await updateStateUpTo(mcapTime);

    // Draw
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    drawKeyboard(ctx, 10, 10, currentState.keyboard);
    const mouseX = 10 + 14 * 35 + 20;
    drawMouse(ctx, mouseX, 10, currentState.mouse.buttons);
    drawMinimap(ctx, mouseX + 70, 10, 160, 100, currentState.mouse.x, currentState.mouse.y, SCREEN_WIDTH, SCREEN_HEIGHT, currentState.mouse.buttons);

    if (timeInfo) timeInfo.textContent = `${video.currentTime.toFixed(2)}s`;

    requestAnimationFrame(render);
  }

  render();
}

// Setup function for initializing with mcapReader
async function setup(reader) {
  mcapReader = reader;

  // Find basePtsTime from first screen event
  for await (const msg of reader.readMessages({ topics: ["screen"] })) {
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    basePtsTime = msg.logTime - BigInt(data?.media_ref?.pts_ns || 0);
    break;
  }

  lastProcessedTime = basePtsTime || 0n;
  lastRecenterTime = lastProcessedTime;

  // Video event handlers
  let pendingSeek = null;

  video.addEventListener("seeked", async () => {
    const targetTime = videoTimeToMcap(video.currentTime);

    // Cancel any pending seek and start a new one
    pendingSeek = targetTime;

    // If already loading, let it finish - it will check pendingSeek
    if (isLoading) return;

    // Load state at the new seek position
    await loadStateAt(targetTime);

    // If another seek happened while we were loading, handle it
    while (pendingSeek !== null && pendingSeek !== lastProcessedTime) {
      const nextTarget = pendingSeek;
      pendingSeek = null;
      await loadStateAt(nextTarget);
    }
    pendingSeek = null;
  });

  video.addEventListener("play", () => {
    userWantsToPlay = true;
    // If still loading, pause immediately (loadStateAt will resume when done)
    if (isLoading) {
      video.pause();
    }
  });

  video.addEventListener("pause", () => {
    // Only clear userWantsToPlay if user actually paused (not our loading pause)
    if (!isLoading) {
      userWantsToPlay = false;
    }
  });
}

// Event handlers
recenterInput?.addEventListener("change", (e) => {
  recenterIntervalMs = Math.max(0, parseInt(e.target.value, 10) || 0);
});

document.querySelectorAll('input[name="mouse-mode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    mouseMode = e.target.value;
    recenterInput.disabled = mouseMode !== "raw";
    loadStateAt(videoTimeToMcap(video.currentTime)); // Reload with new mode
  });
});

function updateLoadButton() {
  const mcapInput = document.getElementById("mcap-input");
  const mkvInput = document.getElementById("mkv-input");
  const loadBtn = document.getElementById("load-btn");
  loadBtn.disabled = !(mcapInput.files?.length && mkvInput.files?.length);
}
document.getElementById("mcap-input")?.addEventListener("change", updateLoadButton);
document.getElementById("mkv-input")?.addEventListener("change", updateLoadButton);

// Common initialization after loading
function initViewer(channelCount) {
  document.getElementById("file-select").classList.add("hidden");
  document.getElementById("viewer").classList.remove("hidden");

  video.onloadedmetadata = () => {
    const w = video.offsetWidth || 800;
    overlay.width = w;
    overlay.height = OVERLAY_HEIGHT;
    overlay.style.width = w + "px";
    startRenderLoop();
  };

  document.getElementById("status").textContent = `Ready: ${channelCount} channels`;
}

// Auto-load from URL params
const params = new URLSearchParams(location.search);
if (params.has("mcap") && params.has("mkv")) {
  (async () => {
    document.getElementById("status").textContent = "Loading...";
    try {
      const { reader, channels } = await loadMcapFromUrl(params.get("mcap"));
      await setup(reader);
      video.src = params.get("mkv");
      initViewer(channels.length);
    } catch (e) {
      document.getElementById("status").textContent = `Error: ${e.message}`;
    }
  })();
}

// Manual file load
document.getElementById("load-btn")?.addEventListener("click", async () => {
  document.getElementById("status").textContent = "Loading...";
  try {
    const mcapFile = document.getElementById("mcap-input").files[0];
    const mkvFile = document.getElementById("mkv-input").files[0];

    const { reader, channels } = await loadMcap(mcapFile);
    await setup(reader);
    video.src = URL.createObjectURL(mkvFile);
    initViewer(channels.length);
  } catch (e) {
    document.getElementById("status").textContent = `Error: ${e.message}`;
    console.error(e);
  }
});

