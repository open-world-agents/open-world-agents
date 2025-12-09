import { FEATURED_DATASETS, MORE_DATASETS } from "./config.js";
import { fetchFileList, fetchLocalFileList, hasFiles, renderFileTree } from "./hf.js";
import { loadFromFiles, loadFromUrls } from "./viewer.js";
import { updateStatus } from "./ui.js";

// Landing page initialization
function initLanding() {
  const featured = document.getElementById("featured-datasets");
  const more = document.getElementById("more-datasets");

  for (const ds of FEATURED_DATASETS) {
    const li = document.createElement("li");
    li.innerHTML = `<a href="?repo_id=${ds}">${ds}</a>`;
    featured.appendChild(li);
  }

  for (const ds of MORE_DATASETS) {
    const li = document.createElement("li");
    li.innerHTML = `<a href="?repo_id=${ds}">${ds}</a>`;
    more.appendChild(li);
  }

  // Search box
  const input = document.getElementById("dataset-input");
  const goBtn = document.getElementById("go-btn");
  const go = () => {
    const v = input.value.trim();
    if (v) location.href = `?repo_id=${v}`;
  };
  goBtn?.addEventListener("click", go);
  input?.addEventListener("keyup", (e) => {
    if (e.key === "Enter") go();
  });

  // File drop zone
  const dropZone = document.getElementById("drop-zone");
  const mcapInput = document.getElementById("mcap-input-landing");
  const mkvInput = document.getElementById("mkv-input-landing");
  const fileStatus = document.getElementById("file-status");
  let mcap = null,
    mkv = null;

  function update() {
    const parts = [];
    if (mcap) parts.push(`✓ ${mcap.name}`);
    if (mkv) parts.push(`✓ ${mkv.name}`);
    fileStatus.textContent = parts.join("  ");
    mcapInput?.parentElement.classList.toggle("selected", !!mcap);
    mkvInput?.parentElement.classList.toggle("selected", !!mkv);
    if (mcap && mkv) loadFromFiles(mcap, mkv, fileStatus);
  }

  mcapInput?.addEventListener("change", (e) => {
    mcap = e.target.files[0] || null;
    update();
  });
  mkvInput?.addEventListener("change", (e) => {
    mkv = e.target.files[0] || null;
    update();
  });

  dropZone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    for (const f of e.dataTransfer.files) {
      if (f.name.endsWith(".mcap")) mcap = f;
      else if (/\.(mkv|mp4|webm)$/i.test(f.name)) mkv = f;
    }
    update();
  });
}

// HuggingFace dataset viewer
async function initHfViewer(repoId) {
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select")?.classList.add("hidden");
  document.getElementById("viewer").classList.remove("hidden");
  updateStatus("Fetching file list...");

  try {
    const tree = await fetchFileList(repoId);
    if (!hasFiles(tree)) {
      updateStatus("No MCAP files found in dataset");
      return;
    }

    const section = document.getElementById("file-section");
    const container = document.getElementById("hf-file-list");
    section?.classList.remove("hidden");

    const firstLi = renderFileTree(tree, container, (f) => loadFromUrls(f.mcap, f.mkv));
    firstLi?.click();
  } catch (e) {
    updateStatus(`Error: ${e.message}`);
  }
}

// Direct URL viewer
async function initUrlViewer(mcapUrl, mkvUrl) {
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select").classList.remove("hidden");
  await loadFromUrls(mcapUrl, mkvUrl);
}

// Local server viewer (base_url parameter)
async function initLocalViewer(baseUrl) {
  document.getElementById("landing")?.classList.add("hidden");
  document.getElementById("file-select")?.classList.add("hidden");
  document.getElementById("viewer").classList.remove("hidden");
  updateStatus("Fetching file list...");

  try {
    const files = await fetchLocalFileList(baseUrl);
    if (files.length === 0) {
      updateStatus("No MCAP files found");
      return;
    }

    const section = document.getElementById("file-section");
    const container = document.getElementById("hf-file-list");
    section?.classList.remove("hidden");

    // Convert flat list to tree format
    const tree = {
      folders: {},
      files: files.map((f) => ({ ...f, mcap: `${baseUrl}/${f.mcap}`, mkv: `${baseUrl}/${f.mkv}` })),
    };
    const firstLi = renderFileTree(tree, container, (f) => loadFromUrls(f.mcap, f.mkv));
    firstLi?.click();
  } catch (e) {
    updateStatus(`Error: ${e.message}`);
  }
}

// Router
const params = new URLSearchParams(location.search);
const repoId = params.get("repo_id");
const baseUrl = params.get("base_url");

if (repoId) {
  initHfViewer(repoId);
} else if (baseUrl) {
  initLocalViewer(baseUrl);
} else if (params.has("mcap") && params.has("mkv")) {
  initUrlViewer(params.get("mcap"), params.get("mkv"));
} else {
  document.getElementById("landing").classList.remove("hidden");
  initLanding();
}
