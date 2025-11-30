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
    const dirs = items.filter(i => i.type === "directory");
    const mcaps = items.filter(i => i.path.endsWith(".mcap"));

    for (const mcap of mcaps) {
      const basename = mcap.path.replace(/\.mcap$/, "");
      node.files.push({
        name: basename.split("/").pop(),
        path: basename,
        mcap: `${baseUrl}/${basename}.mcap`,
        mkv: `${baseUrl}/${basename}.mkv`,
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

export function hasFiles(tree) {
  if (tree.files.length > 0) return true;
  return Object.values(tree.folders).some(hasFiles);
}

export function renderFileTree(tree, container, onSelect) {
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
        onSelect(f);
      };
      parent.appendChild(li);
      if (!firstLi) firstLi = li;
    }
  }

  renderNode(tree, container);
  return firstLi;
}

