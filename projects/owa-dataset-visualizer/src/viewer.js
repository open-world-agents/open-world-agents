import { loadMcap, loadMcapFromUrl, TimeSync } from "./mcap.js";
import { StateManager } from "./state.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay.js";
import { updateWindowInfo, displayMcapInfo, LoadingIndicator, updateStatus } from "./ui.js";
import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  OVERLAY_HEIGHT,
  KEYBOARD_COLUMNS,
  KEY_SIZE,
  KEY_MARGIN,
  TOPICS,
} from "./constants.js";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const timeInfo = document.querySelector("#time-info span");
const recenterInput = document.getElementById("recenter-interval");
const windowInfoEl = document.getElementById("window-info");
const mcapInfoEl = document.getElementById("mcap-info");

let mcapReader = null;
const timeSync = new TimeSync();
const stateManager = new StateManager();
const loading = new LoadingIndicator();
let userWantsToPlay = false;

async function loadStateAt(targetTime) {
  if (!mcapReader) return;
  stateManager.isLoading = true;
  video.pause();
  loading.show();

  try {
    stateManager.reset(targetTime);

    let keyboardStateTime = 0n;
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime,
      topics: [TOPICS.KEYBOARD_STATE],
      reverse: true,
    })) {
      stateManager.applyKeyboardState(JSON.parse(new TextDecoder().decode(msg.data)));
      keyboardStateTime = msg.logTime;
      break;
    }

    if (keyboardStateTime > 0n) {
      for await (const msg of mcapReader.readMessages({
        startTime: keyboardStateTime + 1n,
        endTime: targetTime,
        topics: [TOPICS.KEYBOARD],
      })) {
        stateManager.processMessage(TOPICS.KEYBOARD, JSON.parse(new TextDecoder().decode(msg.data)), msg.logTime);
      }
    }

    let mouseStateTime = 0n;
    for await (const msg of mcapReader.readMessages({
      endTime: targetTime,
      topics: [TOPICS.MOUSE_STATE],
      reverse: true,
    })) {
      stateManager.applyMouseState(JSON.parse(new TextDecoder().decode(msg.data)));
      mouseStateTime = msg.logTime;
      break;
    }

    const mouseTopic = stateManager.getMouseTopic();
    if (mouseStateTime > 0n) {
      for await (const msg of mcapReader.readMessages({
        startTime: mouseStateTime + 1n,
        endTime: targetTime,
        topics: [mouseTopic],
      })) {
        stateManager.processMessage(mouseTopic, JSON.parse(new TextDecoder().decode(msg.data)), msg.logTime);
      }
    }

    for await (const msg of mcapReader.readMessages({ endTime: targetTime, topics: [TOPICS.WINDOW], reverse: true })) {
      stateManager.applyWindowState(JSON.parse(new TextDecoder().decode(msg.data)));
      break;
    }

    stateManager.lastProcessedTime = targetTime;
  } finally {
    stateManager.isLoading = false;
    loading.hide();
  }
  if (userWantsToPlay) video.play();
}

async function updateStateUpTo(targetTime) {
  if (!mcapReader || stateManager.isLoading || targetTime <= stateManager.lastProcessedTime) return;

  for await (const msg of mcapReader.readMessages({
    startTime: stateManager.lastProcessedTime,
    endTime: targetTime,
    topics: stateManager.getUpdateTopics(),
  })) {
    if (stateManager.isLoading) return;
    const channel = mcapReader.channelsById.get(msg.channelId);
    stateManager.processMessage(channel.topic, JSON.parse(new TextDecoder().decode(msg.data)), msg.logTime);
  }

  if (!stateManager.isLoading) stateManager.lastProcessedTime = targetTime;
}

function startRenderLoop() {
  const ctx = overlay.getContext("2d");
  const keyboardWidth = KEYBOARD_COLUMNS * (KEY_SIZE + KEY_MARGIN);
  const mouseX = 10 + keyboardWidth + 20;

  (function render() {
    const mcapTime = timeSync.videoTimeToMcap(video.currentTime);
    updateStateUpTo(mcapTime).catch(console.error);
    stateManager.decayWheel();

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const { keyboard, mouse, window: win } = stateManager.state;
    drawKeyboard(ctx, 10, 10, keyboard);
    drawMouse(ctx, mouseX, 10, mouse.buttons, mouse.wheel);
    drawMinimap(ctx, mouseX + 70, 10, 160, 100, mouse.x, mouse.y, SCREEN_WIDTH, SCREEN_HEIGHT, mouse.buttons);
    updateWindowInfo(windowInfoEl, win);
    if (timeInfo) timeInfo.textContent = `${video.currentTime.toFixed(2)}s`;

    requestAnimationFrame(render);
  })();
}

async function setup(reader) {
  mcapReader = reader;
  await displayMcapInfo(mcapInfoEl, reader);

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
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select")?.classList.add("hidden");
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

// Event handlers
recenterInput?.addEventListener("change", (e) => {
  stateManager.recenterIntervalMs = Math.max(0, parseInt(e.target.value, 10) || 0);
});

document.querySelectorAll('input[name="mouse-mode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    stateManager.mouseMode = e.target.value;
    recenterInput.disabled = stateManager.mouseMode !== "raw";
    loadStateAt(timeSync.videoTimeToMcap(video.currentTime));
  });
});

// Public API
export async function loadFromFiles(mcapFile, mkvFile, statusEl) {
  updateStatus("Loading...");
  try {
    const { reader, channels } = await loadMcap(mcapFile);
    await setup(reader);
    video.src = URL.createObjectURL(mkvFile);
    initViewer(channels.length);
  } catch (e) {
    const msg = `Error: ${e.message}`;
    updateStatus(msg);
    if (statusEl) statusEl.textContent = msg;
  }
}

export async function loadFromUrls(mcapUrl, mkvUrl) {
  updateStatus("Loading...");
  try {
    const { reader, channels } = await loadMcapFromUrl(mcapUrl);
    await setup(reader);
    video.src = mkvUrl;
    initViewer(channels.length);
  } catch (e) {
    updateStatus(`Error: ${e.message}`);
    console.error(e);
  }
}
