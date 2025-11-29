import { loadMcap, readMessages, groupByTopic } from "./mcap-loader.js";

// DOM Elements
const mcapInput = document.getElementById("mcap-input");
const mkvInput = document.getElementById("mkv-input");
const loadBtn = document.getElementById("load-btn");
const status = document.getElementById("status");
const fileSelect = document.getElementById("file-select");
const viewer = document.getElementById("viewer");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const keyboardState = document.querySelector("#keyboard-state span");
const mouseState = document.querySelector("#mouse-state span");
const timeInfo = document.querySelector("#time-info span");

// State
let mcapReader = null;
let basePtsTime = null;
let currentEvents = { keyboard: new Set(), mouse: { x: 0, y: 0, buttons: 0 } };
let allMessages = [];

// Enable load button when both files selected
function updateLoadButton() {
  loadBtn.disabled = !(mcapInput.files?.length && mkvInput.files?.length);
}
mcapInput.addEventListener("change", updateLoadButton);
mkvInput.addEventListener("change", updateLoadButton);

// Load files
loadBtn.addEventListener("click", async () => {
  status.textContent = "Loading...";

  try {
    // Load MCAP
    const mcapFile = mcapInput.files[0];
    const { reader, startTime, endTime, channels } = await loadMcap(mcapFile);
    mcapReader = reader;

    status.textContent = `MCAP loaded: ${channels.length} channels`;

    // Read all messages (for POC - in production use windowed loading)
    allMessages = await readMessages(reader, startTime, endTime);
    const byTopic = groupByTopic(allMessages);

    // Find basePtsTime from first screen event
    const screenEvents = byTopic["screen/frame"] || [];
    if (screenEvents.length > 0) {
      const first = screenEvents[0];
      const ptsNs = BigInt(first.data?.media_ref?.pts_ns || 0);
      basePtsTime = first.timestamp - ptsNs;
    }

    // Load video
    const mkvFile = mkvInput.files[0];
    video.src = URL.createObjectURL(mkvFile);

    // Show viewer
    fileSelect.classList.add("hidden");
    viewer.classList.remove("hidden");

    // Setup canvas
    video.addEventListener("loadedmetadata", () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      startRenderLoop();
    });

    status.textContent = `Loaded ${allMessages.length} events`;
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

// Get events active at timestamp
function getActiveEvents(mcapTime) {
  const keyboard = new Set();
  const mouse = { x: 0, y: 0, buttons: 0 };

  for (const msg of allMessages) {
    if (msg.timestamp > mcapTime) break;

    if (msg.topic === "keyboard/state" && msg.data) {
      // Keyboard state contains pressed keys
      if (msg.data.pressed_keys) {
        keyboard.clear();
        for (const vk of msg.data.pressed_keys) {
          keyboard.add(vk);
        }
      }
    } else if (msg.topic === "mouse/state" && msg.data) {
      mouse.x = msg.data.x ?? 0;
      mouse.y = msg.data.y ?? 0;
      mouse.buttons = msg.data.button_flags ?? 0;
    }
  }

  return { keyboard, mouse };
}

// Render loop
function startRenderLoop() {
  const ctx = overlay.getContext("2d");

  function render() {
    const videoTime = video.currentTime;
    const mcapTime = videoTimeToMcap(videoTime);

    // Get active events
    const events = getActiveEvents(mcapTime);
    currentEvents = events;

    // Clear canvas
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw mouse cursor
    const scaleX = overlay.width / (video.videoWidth || 1920);
    const scaleY = overlay.height / (video.videoHeight || 1080);
    const mx = events.mouse.x * scaleX;
    const my = events.mouse.y * scaleY;

    ctx.beginPath();
    ctx.arc(mx, my, 10, 0, Math.PI * 2);
    if (events.mouse.buttons & 1) {
      ctx.fillStyle = "rgba(255, 0, 0, 0.7)";
      ctx.fill();
    } else if (events.mouse.buttons & 2) {
      ctx.fillStyle = "rgba(0, 0, 255, 0.7)";
      ctx.fill();
    } else {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Update info panel
    const keys = Array.from(events.keyboard).map((vk) => vkToName(vk)).join(", ");
    keyboardState.textContent = keys || "(none)";
    mouseState.textContent = `(${events.mouse.x}, ${events.mouse.y}) btn=${events.mouse.buttons}`;
    timeInfo.textContent = `${videoTime.toFixed(2)}s`;

    requestAnimationFrame(render);
  }

  render();
}

// VK code to name (minimal set)
function vkToName(vk) {
  const names = {
    8: "Back", 9: "Tab", 13: "Enter", 16: "Shift", 17: "Ctrl", 18: "Alt",
    20: "Caps", 27: "Esc", 32: "Space", 37: "←", 38: "↑", 39: "→", 40: "↓",
    65: "A", 66: "B", 67: "C", 68: "D", 69: "E", 70: "F", 71: "G", 72: "H",
    73: "I", 74: "J", 75: "K", 76: "L", 77: "M", 78: "N", 79: "O", 80: "P",
    81: "Q", 82: "R", 83: "S", 84: "T", 85: "U", 86: "V", 87: "W", 88: "X",
    89: "Y", 90: "Z",
  };
  return names[vk] || `VK${vk}`;
}

