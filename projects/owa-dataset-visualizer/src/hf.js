async function fetchTree(repoId, path = "") {
  const url = `https://huggingface.co/api/datasets/${repoId}/tree/main${path ? "/" + path : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
  return res.json();
}

export async function fetchFileList(repoId) {
  const baseUrl = `https://huggingface.co/datasets/${repoId}/resolve/main`;
  const tree = { folders: {}, files: [] };

  async function scanDir(path, node) {
    const items = await fetchTree(repoId, path);
    const dirs = items.filter((i) => i.type === "directory");
    const files = items.filter((i) => i.type !== "directory");

    // Group files by basename and pair mcap with video
    const pairs = new Map();
    for (const file of files) {
      const isMcap = file.path.endsWith(".mcap");
      const isVideo = /\.(mkv|mp4|webm)$/i.test(file.path);
      if (!isMcap && !isVideo) continue;

      const basename = file.path.replace(/\.(mcap|mkv|mp4|webm)$/i, "");
      const pair = pairs.get(basename) || {};
      if (isMcap) pair.mcap = file.path;
      if (isVideo) pair.video = file.path;
      pairs.set(basename, pair);
    }

    for (const [basename, pair] of pairs) {
      if (pair.mcap && pair.video) {
        node.files.push({
          name: basename.split("/").pop(),
          path: basename,
          mcap: `${baseUrl}/${pair.mcap}`,
          mkv: `${baseUrl}/${pair.video}`,
        });
      }
    }

    // Scan subdirectories in parallel
    await Promise.all(
      dirs.map(async (dir) => {
        const folderName = dir.path.split("/").pop();
        node.folders[folderName] = { folders: {}, files: [] };
        await scanDir(dir.path, node.folders[folderName]);
      }),
    );
  }

  await scanDir("", tree);
  return tree;
}

export function hasFiles(tree) {
  if (tree.files.length > 0) return true;
  return Object.values(tree.folders).some(hasFiles);
}

export async function fetchLocalFileList(baseUrl) {
  const res = await fetch(`${baseUrl}/files.json`);
  if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
  const files = await res.json();
  return buildTreeFromFiles(files, baseUrl);
}

function buildTreeFromFiles(files, baseUrl) {
  const tree = { folders: {}, files: [] };

  for (const f of files) {
    const parts = f.path.split("/").filter((p) => p);
    let node = tree;

    // Navigate/create folder structure
    for (let i = 0; i < parts.length - 1; i++) {
      const folder = parts[i];
      if (!node.folders[folder]) {
        node.folders[folder] = { folders: {}, files: [] };
      }
      node = node.folders[folder];
    }

    node.files.push({
      name: f.name,
      path: f.path,
      mcap: `${baseUrl}/${f.mcap}`,
      mkv: `${baseUrl}/${f.mkv}`,
    });
  }

  return tree;
}

export function renderFileTree(tree, container, onSelect) {
  container.innerHTML = "";
  let firstLi = null;
  let activeLi = null;

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
      li.onclick = async () => {
        activeLi?.classList.remove("active", "loading");
        li.classList.add("active", "loading");
        activeLi = li;
        try {
          await onSelect(f);
        } finally {
          li.classList.remove("loading");
        }
      };
      parent.appendChild(li);
      if (!firstLi) firstLi = li;
    }
  }

  renderNode(tree, container);
  return firstLi;
}
