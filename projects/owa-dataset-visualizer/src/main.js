import { loadMcap, loadMcapFromUrl } from "./mcap-loader.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay-renderer.js";

// DOM Elements
const mcapInput = document.getElementById("mcap-input");
const mkvInput = document.getElementById("mkv-input");
const loadBtn = document.getElementById("load-btn");
const status = document.getElementById("status");
const fileSelect = document.getElementById("file-select");
const viewer = document.getElementById("viewer");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const timeInfo = document.querySelector("#time-info span");

// State
let mcapReader = null;
let basePtsTime = null;
let cachedState = {
  keyboard: new Set(),
  mouse: { x: 960, y: 540, buttons: new Set() }, // Start at screen center
};
let lastLoadedTime = 0n;
let mouseMode = "raw"; // "raw" (relative/3D) or "absolute" (2D)
let recenterIntervalMs = 0; // 0 = off, else recenter every N ms
let lastRecenterTime = 0n;

// Mouse button VK to name
const MOUSE_VK_MAP = { 1: "left", 2: "right", 4: "middle", 5: "x1", 6: "x2" };

// Screen dimensions for mouse position clamping
const SCREEN_WIDTH = 1920;
const SCREEN_HEIGHT = 1080;

// Overlay dimensions
const OVERLAY_HEIGHT = 220;

// Mouse mode selection
document.querySelectorAll('input[name="mouse-mode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    mouseMode = e.target.value;
    resetState();
  });
});

// Recenter interval input
const recenterInput = document.getElementById("recenter-interval");
recenterInput?.addEventListener("change", (e) => {
  recenterIntervalMs = Math.max(0, parseInt(e.target.value, 10) || 0);
});

// Enable load button when both files selected
function updateLoadButton() {
  loadBtn.disabled = !(mcapInput.files?.length && mkvInput.files?.length);
}
mcapInput.addEventListener("change", updateLoadButton);
mkvInput.addEventListener("change", updateLoadButton);

// Auto-load from URL params (for testing): ?mcap=/test.mcap&mkv=/test.mkv
const params = new URLSearchParams(location.search);
if (params.has("mcap") && params.has("mkv")) {
  (async () => {
    status.textContent = "Loading...";
    try {
      const { reader, channels } = await loadMcapFromUrl(params.get("mcap"));
      mcapReader = reader;
      for await (const msg of reader.readMessages({ topics: ["screen"] })) {
        const data = JSON.parse(new TextDecoder().decode(msg.data));
        basePtsTime = msg.logTime - BigInt(data?.media_ref?.pts_ns || 0);
        break;
      }
      video.src = params.get("mkv");
      fileSelect.classList.add("hidden");
      viewer.classList.remove("hidden");
      video.onloadedmetadata = () => {
        // Match canvas to video's displayed width
        const displayWidth = video.offsetWidth || video.clientWidth || 800;
        overlay.width = displayWidth;
        overlay.height = OVERLAY_HEIGHT;
        overlay.style.width = displayWidth + "px";
        startRenderLoop();
      };
      video.onseeking = resetState;
      status.textContent = `Ready: ${channels.length} channels`;
    } catch (e) {
      status.textContent = `Error: ${e.message}`;
    }
  })();
}

// Load files
loadBtn.addEventListener("click", async () => {
  status.textContent = "Loading MCAP index...";

  try {
    // Load MCAP (index only, no messages yet)
    const mcapFile = mcapInput.files[0];
    const { reader, channels } = await loadMcap(mcapFile);
    mcapReader = reader;

    // Find basePtsTime from first screen event
    for await (const msg of reader.readMessages({ topics: ["screen"] })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      basePtsTime = msg.logTime - BigInt(data?.media_ref?.pts_ns || 0);
      break;
    }

    // Load video
    const mkvFile = mkvInput.files[0];
    video.src = URL.createObjectURL(mkvFile);

    // Show viewer
    fileSelect.classList.add("hidden");
    viewer.classList.remove("hidden");

    // Setup canvas
    video.addEventListener("loadedmetadata", () => {
      const displayWidth = video.offsetWidth || video.clientWidth || 800;
      overlay.width = displayWidth;
      overlay.height = OVERLAY_HEIGHT;
      overlay.style.width = displayWidth + "px";
      startRenderLoop();
    });
    video.addEventListener("seeking", resetState);

    status.textContent = `Ready: ${channels.length} channels`;
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
  }
});

// Convert video time to MCAP timestamp
function videoTimeToMcap(videoTimeSec) {
  if (basePtsTime === null) return 0n;
  return basePtsTime + BigInt(Math.floor(videoTimeSec * 1e9));
}

// Reset state when seeking
function resetState() {
  lastLoadedTime = 0n;
  lastRecenterTime = 0n;
  cachedState.keyboard.clear();
  cachedState.mouse = { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 2, buttons: new Set() };
}

// Mouse button flags from RawMouseEvent
const BUTTON_PRESS_FLAGS = {
  0x0001: "left",   // RI_MOUSE_LEFT_BUTTON_DOWN
  0x0004: "right",  // RI_MOUSE_RIGHT_BUTTON_DOWN
  0x0010: "middle", // RI_MOUSE_MIDDLE_BUTTON_DOWN
};
const BUTTON_RELEASE_FLAGS = {
  0x0002: "left",   // RI_MOUSE_LEFT_BUTTON_UP
  0x0008: "right",  // RI_MOUSE_RIGHT_BUTTON_UP
  0x0020: "middle", // RI_MOUSE_MIDDLE_BUTTON_UP
};

// Load state up to timestamp (incremental)
async function loadStateUpTo(mcapTime) {
  if (!mcapReader || mcapTime <= lastLoadedTime) return;

  const mouseTopic = mouseMode === "raw" ? "mouse/raw" : "mouse";
  const topics = ["keyboard/state", mouseTopic];

  for await (const msg of mcapReader.readMessages({
    startTime: lastLoadedTime,
    endTime: mcapTime,
    topics,
  })) {
    const channel = mcapReader.channelsById.get(msg.channelId);
    const data = JSON.parse(new TextDecoder().decode(msg.data));

    if (channel.topic === "keyboard/state") {
      // Filter out mouse VK codes (1, 2, 4, 5, 6)
      const keyVks = (data.buttons || []).filter((vk) => !MOUSE_VK_MAP[vk]);
      cachedState.keyboard = new Set(keyVks);
    } else if (channel.topic === "mouse/raw") {
      // Check if we need to recenter
      if (recenterIntervalMs > 0) {
        const intervalNs = BigInt(recenterIntervalMs) * 1000000n;
        if (msg.logTime - lastRecenterTime >= intervalNs) {
          cachedState.mouse.x = SCREEN_WIDTH / 2;
          cachedState.mouse.y = SCREEN_HEIGHT / 2;
          lastRecenterTime = msg.logTime;
        }
      }

      // Relative mode: accumulate mouse movement
      const dx = data.last_x ?? 0;
      const dy = data.last_y ?? 0;
      cachedState.mouse.x = Math.max(0, Math.min(SCREEN_WIDTH - 1, cachedState.mouse.x + dx));
      cachedState.mouse.y = Math.max(0, Math.min(SCREEN_HEIGHT - 1, cachedState.mouse.y + dy));

      // Process button flags
      const flags = data.button_flags ?? 0;
      for (const [flag, btn] of Object.entries(BUTTON_PRESS_FLAGS)) {
        if (flags & Number(flag)) cachedState.mouse.buttons.add(btn);
      }
      for (const [flag, btn] of Object.entries(BUTTON_RELEASE_FLAGS)) {
        if (flags & Number(flag)) cachedState.mouse.buttons.delete(btn);
      }
    } else if (channel.topic === "mouse") {
      // Absolute mode: use direct coordinates
      cachedState.mouse.x = data.x ?? cachedState.mouse.x;
      cachedState.mouse.y = data.y ?? cachedState.mouse.y;

      // Handle click events
      if (data.event_type === "click" && data.button) {
        if (data.pressed) {
          cachedState.mouse.buttons.add(data.button);
        } else {
          cachedState.mouse.buttons.delete(data.button);
        }
      }
    }
  }
  lastLoadedTime = mcapTime;
}

// Render loop
function startRenderLoop() {
  const ctx = overlay.getContext("2d");

  async function render() {
    const videoTime = video.currentTime;
    const mcapTime = videoTimeToMcap(videoTime);

    // Load state incrementally
    await loadStateUpTo(mcapTime);

    // Clear canvas
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw keyboard (left side)
    const kbX = 10;
    const kbY = 10;
    drawKeyboard(ctx, kbX, kbY, cachedState.keyboard);

    // Draw mouse figure (right of keyboard)
    const mouseX = kbX + 14 * 35 + 20; // 14 cols * (32+3) + gap
    drawMouse(ctx, mouseX, kbY, cachedState.mouse.buttons);

    // Draw minimap (right of mouse)
    const minimapX = mouseX + 70;
    drawMinimap(
      ctx, minimapX, kbY, 160, 100,
      cachedState.mouse.x, cachedState.mouse.y,
      1920, 1080, // Assume 1080p screen
      cachedState.mouse.buttons
    );

    // Update time info
    if (timeInfo) {
      timeInfo.textContent = `${videoTime.toFixed(2)}s`;
    }

    requestAnimationFrame(render);
  }

  render();
}

