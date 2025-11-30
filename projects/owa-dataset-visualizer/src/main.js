/**
 * OWA Dataset Visualizer - Main Entry Point
 */
import { loadMcap, loadMcapFromUrl, TimeSync } from "./mcap.js";
import { StateManager } from "./state.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay.js";
import { updateWindowInfo, displayMcapInfo, LoadingIndicator, updateStatus } from "./ui.js";
import { SCREEN_WIDTH, SCREEN_HEIGHT, OVERLAY_HEIGHT, KEYBOARD_COLUMNS, KEY_SIZE, KEY_MARGIN, TOPICS } from "./constants.js";

// DOM Elements
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const timeInfo = document.querySelector("#time-info span");
const recenterInput = document.getElementById("recenter-interval");
const windowInfoEl = document.getElementById("window-info");
const mcapInfoEl = document.getElementById("mcap-info");

// Application State
let mcapReader = null;
const timeSync = new TimeSync();
const stateManager = new StateManager();
const loading = new LoadingIndicator();
let userWantsToPlay = false;

// Load state at a specific MCAP timestamp
async function loadStateAt(targetTime) {
  if (!mcapReader) return;

  const t0 = performance.now();
  console.log(`[loadStateAt] Start, targetTime=${targetTime}`);

  stateManager.isLoading = true;
  video.pause();
  loading.show();

  try {
    stateManager.reset(targetTime);

    // Step 1: Find last keyboard/state
    let keyboardStateTime = 0n;
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime, topics: [TOPICS.KEYBOARD_STATE], reverse: true,
    })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      stateManager.applyKeyboardState(data);
      keyboardStateTime = msg.logTime;
      break;
    }

    // Step 2: Process keyboard events
    if (keyboardStateTime > 0n) {
      for await (const msg of mcapReader.readMessages({
        startTime: keyboardStateTime + 1n, endTime: targetTime, topics: [TOPICS.KEYBOARD],
      })) {
        const data = JSON.parse(new TextDecoder().decode(msg.data));
        stateManager.processMessage(TOPICS.KEYBOARD, data, msg.logTime);
      }
    }

    // Step 3: Find last mouse/state
    let mouseStateTime = 0n;
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime, topics: [TOPICS.MOUSE_STATE], reverse: true,
    })) {
      const data = JSON.parse(new TextDecoder().decode(msg.data));
      stateManager.applyMouseState(data);
      mouseStateTime = msg.logTime;
      break;
    }

    // Step 4: Process mouse events
    const mouseTopic = stateManager.getMouseTopic();
    if (mouseStateTime > 0n) {
      for await (const msg of mcapReader.readMessages({
        startTime: mouseStateTime + 1n, endTime: targetTime, topics: [mouseTopic],
      })) {
        const data = JSON.parse(new TextDecoder().decode(msg.data));
        stateManager.processMessage(mouseTopic, data, msg.logTime);
      }
    }

    // Step 5: Find last window info
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime, topics: [TOPICS.WINDOW], reverse: true,
    })) {
      stateManager.applyWindowState(JSON.parse(new TextDecoder().decode(msg.data)));
      break;
    }

    stateManager.lastProcessedTime = targetTime;
  } finally {
    stateManager.isLoading = false;
    loading.hide();
  }

  console.log(`[loadStateAt] Complete (+${(performance.now() - t0).toFixed(1)}ms)`);
  if (userWantsToPlay) video.play();
}

// Incremental state update during playback
async function updateStateUpTo(targetTime) {
  if (!mcapReader || stateManager.isLoading || targetTime <= stateManager.lastProcessedTime) {
    return;
  }

  for await (const msg of mcapReader.readMessages({
    startTime: stateManager.lastProcessedTime,
    endTime: targetTime,
    topics: stateManager.getUpdateTopics(),
  })) {
    if (stateManager.isLoading) return;

    const channel = mcapReader.channelsById.get(msg.channelId);
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    stateManager.processMessage(channel.topic, data, msg.logTime);
  }

  if (!stateManager.isLoading) {
    stateManager.lastProcessedTime = targetTime;
  }
}

// Render loop
function startRenderLoop() {
  const ctx = overlay.getContext("2d");
  const keyboardWidth = KEYBOARD_COLUMNS * (KEY_SIZE + KEY_MARGIN);
  const mouseX = 10 + keyboardWidth + 20;

  function render() {
    const mcapTime = timeSync.videoTimeToMcap(video.currentTime);
    updateStateUpTo(mcapTime).catch(err => console.error("State update error:", err));
    stateManager.decayWheel();

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const state = stateManager.state;
    drawKeyboard(ctx, 10, 10, state.keyboard);
    drawMouse(ctx, mouseX, 10, state.mouse.buttons, state.mouse.wheel);
    drawMinimap(ctx, mouseX + 70, 10, 160, 100, state.mouse.x, state.mouse.y, SCREEN_WIDTH, SCREEN_HEIGHT, state.mouse.buttons);
    updateWindowInfo(windowInfoEl, state.window);
    if (timeInfo) timeInfo.textContent = `${video.currentTime.toFixed(2)}s`;

    requestAnimationFrame(render);
  }
  render();
}

// Setup
async function setup(reader) {
  mcapReader = reader;
  await displayMcapInfo(mcapInfoEl, reader);

  // Find basePtsTime from first screen event
  for await (const msg of reader.readMessages({ topics: [TOPICS.SCREEN] })) {
    timeSync.initFromScreenMessage(msg.logTime, JSON.parse(new TextDecoder().decode(msg.data)));
    break;
  }

  stateManager.lastProcessedTime = timeSync.getBasePtsTime();
  stateManager.lastRecenterTime = stateManager.lastProcessedTime;

  let pendingSeek = null;
  video.addEventListener("seeked", async () => {
    const targetTime = timeSync.videoTimeToMcap(video.currentTime);
    pendingSeek = targetTime;
    if (stateManager.isLoading) return;
    await loadStateAt(targetTime);
    while (pendingSeek !== null && pendingSeek !== stateManager.lastProcessedTime) {
      const nextTarget = pendingSeek;
      pendingSeek = null;
      await loadStateAt(nextTarget);
    }
    pendingSeek = null;
  });

  video.addEventListener("play", () => {
    userWantsToPlay = true;
    if (stateManager.isLoading) video.pause();
  });

  video.addEventListener("pause", () => {
    if (!stateManager.isLoading) userWantsToPlay = false;
  });
}

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
  updateStatus(`Ready: ${channelCount} channels`);
}

// Event Handlers
recenterInput?.addEventListener("change", e => {
  stateManager.recenterIntervalMs = Math.max(0, parseInt(e.target.value, 10) || 0);
});

document.querySelectorAll('input[name="mouse-mode"]').forEach(radio => {
  radio.addEventListener("change", e => {
    stateManager.mouseMode = e.target.value;
    recenterInput.disabled = stateManager.mouseMode !== "raw";
    loadStateAt(timeSync.videoTimeToMcap(video.currentTime));
  });
});

function updateLoadButton() {
  const mcap = document.getElementById("mcap-input");
  const mkv = document.getElementById("mkv-input");
  document.getElementById("load-btn").disabled = !(mcap.files?.length && mkv.files?.length);
}
document.getElementById("mcap-input")?.addEventListener("change", updateLoadButton);
document.getElementById("mkv-input")?.addEventListener("change", updateLoadButton);

// Initialization
const params = new URLSearchParams(location.search);
if (params.has("mcap") && params.has("mkv")) {
  (async () => {
    updateStatus("Loading...");
    try {
      const { reader, channels } = await loadMcapFromUrl(params.get("mcap"));
      await setup(reader);
      video.src = params.get("mkv");
      initViewer(channels.length);
    } catch (e) {
      updateStatus(`Error: ${e.message}`);
      console.error(e);
    }
  })();
}

document.getElementById("load-btn")?.addEventListener("click", async () => {
  updateStatus("Loading...");
  try {
    const mcapFile = document.getElementById("mcap-input").files[0];
    const mkvFile = document.getElementById("mkv-input").files[0];
    const { reader, channels } = await loadMcap(mcapFile);
    await setup(reader);
    video.src = URL.createObjectURL(mkvFile);
    initViewer(channels.length);
  } catch (e) {
    updateStatus(`Error: ${e.message}`);
    console.error(e);
  }
});
