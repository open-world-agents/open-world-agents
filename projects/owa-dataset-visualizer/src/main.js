/**
 * OWA Dataset Visualizer - Main Entry Point
 */
import { loadMcap, loadMcapFromUrl, TimeSync } from "./mcap.js";
import { StateManager } from "./state.js";
import { drawKeyboard, drawMouse, drawMinimap } from "./overlay.js";
import { updateWindowInfo, displayMcapInfo, LoadingIndicator, updateStatus } from "./ui.js";
import { SCREEN_WIDTH, SCREEN_HEIGHT, OVERLAY_HEIGHT, KEYBOARD_COLUMNS, KEY_SIZE, KEY_MARGIN, TOPICS } from "./constants.js";

// Config
const FEATURED_DATASETS = [
  "open-world-agents/D2E-480p",
  "open-world-agents/example_dataset",
  "open-world-agents/example-djmax",
  "open-world-agents/example-aimlab",
  "open-world-agents/example-pubg-battleground",
];

const MORE_DATASETS = [
  "open-world-agents/example_dataset2",
];

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

async function loadFiles(mcapFile, mkvFile, statusEl) {
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

// HuggingFace dataset fetching
async function fetchHfTree(repoId, path = "") {
  const apiUrl = `https://huggingface.co/api/datasets/${repoId}/tree/main${path ? "/" + path : ""}`;
  const res = await fetch(apiUrl);
  if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
  return res.json();
}

async function fetchHfFileList(repoId) {
  const baseUrl = `https://huggingface.co/datasets/${repoId}/resolve/main`;
  const tree = { folders: {}, files: [] };

  async function scanDir(path, node) {
    const items = await fetchHfTree(repoId, path);
    const dirs = items.filter(i => i.type === "directory");
    const mcaps = items.filter(i => i.path.endsWith(".mcap"));

    for (const mcap of mcaps) {
      const basename = mcap.path.replace(/\.mcap$/, "");
      node.files.push({
        name: basename.split("/").pop(),
        path: basename,
        url_mcap: `${baseUrl}/${basename}.mcap`,
        url_mkv: `${baseUrl}/${basename}.mkv`,
      });
    }

    for (const dir of dirs) {
      const folderName = dir.path.split("/").pop();
      node.folders[folderName] = { folders: {}, files: [] };
      await scanDir(dir.path, node.folders[folderName]);
    }
  }

  await scanDir("", tree);
  return tree;
}

async function loadHfFilePair(filePair) {
  updateStatus("Loading from HuggingFace...");
  try {
    const { reader, channels } = await loadMcapFromUrl(filePair.url_mcap);
    await setup(reader);
    video.src = filePair.url_mkv;
    initViewer(channels.length);
  } catch (e) {
    updateStatus(`Error: ${e.message}`);
    console.error(e);
  }
}

function showFileTree(tree) {
  const section = document.getElementById("file-section");
  const container = document.getElementById("hf-file-list");
  if (!section || !container) return;

  section.classList.remove("hidden");
  container.innerHTML = "";
  let firstLi = null;

  function renderNode(node, parent) {
    for (const [name, subNode] of Object.entries(node.folders).sort((a, b) => a[0].localeCompare(b[0]))) {
      const details = document.createElement("details");
      details.innerHTML = `<summary>${name}</summary>`;
      const ul = document.createElement("ul");
      renderNode(subNode, ul);
      details.appendChild(ul);
      parent.appendChild(details);
    }

    for (const f of node.files.sort((a, b) => a.name.localeCompare(b.name))) {
      const li = document.createElement("li");
      li.textContent = f.name;
      li.onclick = () => {
        container.querySelectorAll("li").forEach(el => el.classList.remove("active"));
        li.classList.add("active");
        loadHfFilePair(f);
      };
      parent.appendChild(li);
      if (!firstLi) firstLi = li;
    }
  }

  renderNode(tree, container);
  firstLi?.click();
}

function initLanding() {
  const featured = document.getElementById("featured-datasets");
  const more = document.getElementById("more-datasets");

  FEATURED_DATASETS.forEach(ds => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `?repo_id=${ds}`;
    a.textContent = ds;
    li.appendChild(a);
    featured.appendChild(li);
  });

  MORE_DATASETS.forEach(ds => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `?repo_id=${ds}`;
    a.textContent = ds;
    li.appendChild(a);
    more.appendChild(li);
  });

  // Search box
  const input = document.getElementById("dataset-input");
  const goBtn = document.getElementById("go-btn");
  const navigateToDataset = () => {
    const val = input.value.trim();
    if (val) location.href = `?repo_id=${val}`;
  };
  goBtn?.addEventListener("click", navigateToDataset);
  input?.addEventListener("keyup", e => { if (e.key === "Enter") navigateToDataset(); });

  // Landing file selection
  const dropZone = document.getElementById("drop-zone");
  const mcapInput = document.getElementById("mcap-input-landing");
  const mkvInput = document.getElementById("mkv-input-landing");
  const fileStatus = document.getElementById("file-status");
  let selectedMcap = null, selectedMkv = null;

  function updateFileStatus() {
    const parts = [];
    if (selectedMcap) parts.push(`✓ ${selectedMcap.name}`);
    if (selectedMkv) parts.push(`✓ ${selectedMkv.name}`);
    fileStatus.textContent = parts.join("  ");

    mcapInput?.parentElement.classList.toggle("selected", !!selectedMcap);
    mkvInput?.parentElement.classList.toggle("selected", !!selectedMkv);

    if (selectedMcap && selectedMkv) {
      loadFiles(selectedMcap, selectedMkv, fileStatus);
    }
  }

  mcapInput?.addEventListener("change", e => {
    selectedMcap = e.target.files[0] || null;
    updateFileStatus();
  });
  mkvInput?.addEventListener("change", e => {
    selectedMkv = e.target.files[0] || null;
    updateFileStatus();
  });

  // Drag and drop
  dropZone?.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone?.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });
  dropZone?.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    for (const file of e.dataTransfer.files) {
      if (file.name.endsWith(".mcap")) selectedMcap = file;
      else if (/\.(mkv|mp4|webm)$/i.test(file.name)) selectedMkv = file;
    }
    updateFileStatus();
  });
}

// Initialization
const params = new URLSearchParams(location.search);
const repoId = params.get("repo_id");

function hasFiles(tree) {
  if (tree.files.length > 0) return true;
  return Object.values(tree.folders).some(hasFiles);
}

if (repoId) {
  // Viewer mode with HuggingFace dataset
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select")?.classList.add("hidden");
  document.getElementById("viewer").classList.remove("hidden");
  updateStatus("Fetching file list...");

  fetchHfFileList(repoId)
    .then(tree => {
      if (!hasFiles(tree)) {
        updateStatus("No MCAP files found in dataset");
        return;
      }
      showFileTree(tree);
    })
    .catch(e => {
      updateStatus(`Error: ${e.message}`);
      console.error(e);
    });

} else if (params.has("mcap") && params.has("mkv")) {
  // Direct URL mode
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select").classList.remove("hidden");
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

} else {
  // Landing page mode
  document.getElementById("landing").classList.remove("hidden");
  initLanding();
}