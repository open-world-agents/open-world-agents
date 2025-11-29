import { loadMcap } from "./mcap-loader.js";

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
let cachedState = { keyboard: new Set(), mouse: { x: 0, y: 0, buttons: 0 } };
let lastLoadedTime = 0n;

// Enable load button when both files selected
function updateLoadButton() {
  loadBtn.disabled = !(mcapInput.files?.length && mkvInput.files?.length);
}
mcapInput.addEventListener("change", updateLoadButton);
mkvInput.addEventListener("change", updateLoadButton);

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
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      startRenderLoop();
    });

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

// Load state up to timestamp (incremental)
async function loadStateUpTo(mcapTime) {
  if (!mcapReader || mcapTime <= lastLoadedTime) return;

  const topics = ["keyboard/state", "mouse/state"];
  for await (const msg of mcapReader.readMessages({
    startTime: lastLoadedTime,
    endTime: mcapTime,
    topics,
  })) {
    const channel = mcapReader.channelsById.get(msg.channelId);
    const data = JSON.parse(new TextDecoder().decode(msg.data));

    if (channel.topic === "keyboard/state") {
      // data.buttons contains VK codes of pressed keys
      cachedState.keyboard = new Set(data.buttons || []);
    } else if (channel.topic === "mouse/state") {
      cachedState.mouse.x = data.x ?? 0;
      cachedState.mouse.y = data.y ?? 0;
      // data.buttons is an array of pressed button indices
      cachedState.mouse.buttons = (data.buttons || []).length > 0 ? 1 : 0;
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

    // Draw mouse cursor
    const scaleX = overlay.width / (video.videoWidth || 1920);
    const scaleY = overlay.height / (video.videoHeight || 1080);
    const mx = cachedState.mouse.x * scaleX;
    const my = cachedState.mouse.y * scaleY;

    ctx.beginPath();
    ctx.arc(mx, my, 10, 0, Math.PI * 2);
    if (cachedState.mouse.buttons & 1) {
      ctx.fillStyle = "rgba(255, 0, 0, 0.7)";
      ctx.fill();
    } else if (cachedState.mouse.buttons & 2) {
      ctx.fillStyle = "rgba(0, 0, 255, 0.7)";
      ctx.fill();
    } else {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Update info panel
    const keys = Array.from(cachedState.keyboard).map((vk) => vkToName(vk)).join(", ");
    keyboardState.textContent = keys || "(none)";
    mouseState.textContent = `(${cachedState.mouse.x}, ${cachedState.mouse.y}) btn=${cachedState.mouse.buttons}`;
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

