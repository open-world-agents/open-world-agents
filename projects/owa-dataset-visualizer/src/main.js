/**
 * OWA Dataset Visualizer - Main Entry Point
 * Synchronizes MCAP input data with MKV video playback
 * @module main
 */

import { loadMcap, loadMcapFromUrl } from "./mcap/loader.js";
import { TimeSync } from "./mcap/time-utils.js";
import { StateManager } from "./state/index.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay/renderer.js";
import { updateWindowInfo, displayMcapInfo } from "./ui/side-panel.js";
import { LoadingIndicator, updateStatus } from "./ui/loading.js";
import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  OVERLAY_HEIGHT,
  KEYBOARD_COLUMNS,
  KEY_SIZE,
  KEY_MARGIN,
  TOPICS,
} from "./constants.js";

// ============================================================================
// DOM Elements
// ============================================================================

/** @type {HTMLVideoElement} */
const video = document.getElementById("video");
/** @type {HTMLCanvasElement} */
const overlay = document.getElementById("overlay");
/** @type {HTMLElement|null} */
const timeInfo = document.querySelector("#time-info span");
/** @type {HTMLInputElement|null} */
const recenterInput = document.getElementById("recenter-interval");
/** @type {HTMLElement|null} */
const windowInfoEl = document.getElementById("window-info");
/** @type {HTMLElement|null} */
const mcapInfoEl = document.getElementById("mcap-info");

// ============================================================================
// Application State
// ============================================================================

/** @type {McapIndexedReader|null} */
let mcapReader = null;

/** Time synchronization manager */
const timeSync = new TimeSync();

/** Input state manager */
const stateManager = new StateManager();

/** Loading indicator */
const loading = new LoadingIndicator();

/** Whether user wants video to play (tracks intent during loading) */
let userWantsToPlay = false;

// ============================================================================
// State Loading
// ============================================================================

/**
 * Load state at a specific MCAP timestamp
 * Finds the nearest state snapshots and replays events to reach exact time
 * @param {bigint} targetTime - Target MCAP timestamp
 */
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
      stateManager.state.window = JSON.parse(new TextDecoder().decode(msg.data));
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

// ============================================================================
// Incremental State Updates
// ============================================================================

/**
 * Update state incrementally from lastProcessedTime to targetTime
 * Used during normal playback (not seeking)
 * @param {bigint} targetTime - Target MCAP timestamp
 */
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

// ============================================================================
// Render Loop
// ============================================================================

/**
 * Start the render loop for overlay updates
 */
function startRenderLoop() {
  const ctx = overlay.getContext("2d");
  const keyboardWidth = KEYBOARD_COLUMNS * (KEY_SIZE + KEY_MARGIN);
  const mouseX = 10 + keyboardWidth + 20;

  async function render() {
    const mcapTime = timeSync.videoTimeToMcap(video.currentTime);

    // Update state (fire-and-forget for smooth rendering)
    updateStateUpTo(mcapTime).catch((err) => {
      console.error("Error during state update:", err);
    });

    // Decay wheel indicator
    stateManager.decayWheel();

    // Draw overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const state = stateManager.state;
    drawKeyboard(ctx, 10, 10, state.keyboard);
    drawMouse(ctx, mouseX, 10, state.mouse.buttons, state.mouse.wheel);
    drawMinimap(
      ctx, mouseX + 70, 10, 160, 100,
      state.mouse.x, state.mouse.y,
      SCREEN_WIDTH, SCREEN_HEIGHT,
      state.mouse.buttons
    );

    // Update side panel
    updateWindowInfo(windowInfoEl, state.window);

    // Update time display
    if (timeInfo) {
      timeInfo.textContent = `${video.currentTime.toFixed(2)}s`;
    }

    requestAnimationFrame(render);
  }

  render();
}

// ============================================================================
// Setup
// ============================================================================

/**
 * Initialize the visualizer with an MCAP reader
 * @param {McapIndexedReader} reader - MCAP reader instance
 */
async function setup(reader) {
  mcapReader = reader;

  // Display MCAP info
  await displayMcapInfo(mcapInfoEl, reader);

  // Find basePtsTime from first screen event
  for await (const msg of reader.readMessages({ topics: [TOPICS.SCREEN] })) {
    const data = JSON.parse(new TextDecoder().decode(msg.data));
    timeSync.initFromScreenMessage(msg.logTime, data);
    break;
  }

  stateManager.lastProcessedTime = timeSync.getBasePtsTime();
  stateManager.lastRecenterTime = stateManager.lastProcessedTime;

  // Video event handlers
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
    if (stateManager.isLoading) {
      video.pause();
    }
  });

  video.addEventListener("pause", () => {
    if (!stateManager.isLoading) {
      userWantsToPlay = false;
    }
  });
}

/**
 * Common initialization after loading files
 * @param {number} channelCount - Number of MCAP channels
 */
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

// ============================================================================
// Event Handlers
// ============================================================================

// Recenter interval
recenterInput?.addEventListener("change", (e) => {
  stateManager.recenterIntervalMs = Math.max(0, parseInt(e.target.value, 10) || 0);
});

// Mouse mode toggle
document.querySelectorAll('input[name="mouse-mode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    stateManager.mouseMode = e.target.value;
    recenterInput.disabled = stateManager.mouseMode !== "raw";
    loadStateAt(timeSync.videoTimeToMcap(video.currentTime));
  });
});

// File input validation
function updateLoadButton() {
  const mcapInput = document.getElementById("mcap-input");
  const mkvInput = document.getElementById("mkv-input");
  const loadBtn = document.getElementById("load-btn");
  loadBtn.disabled = !(mcapInput.files?.length && mkvInput.files?.length);
}

document.getElementById("mcap-input")?.addEventListener("change", updateLoadButton);
document.getElementById("mkv-input")?.addEventListener("change", updateLoadButton);

// ============================================================================
// Initialization
// ============================================================================

// Auto-load from URL params
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

// Manual file load
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
