/**
 * UI components: loading indicator, window info, MCAP info
 */

export class LoadingIndicator {
  constructor(elementId = "loading-indicator") {
    this.element = document.getElementById(elementId);
  }
  show() {
    this.element?.classList.remove("hidden");
  }
  hide() {
    this.element?.classList.add("hidden");
  }
}

export function updateStatus(message, elementId = "status") {
  const el = document.getElementById(elementId);
  if (el) el.textContent = message;
}

export function updateWindowInfo(container, windowData) {
  if (!container) return;
  container.innerHTML = "";

  if (!windowData) {
    container.innerHTML = '<p class="placeholder">No window data</p>';
    return;
  }

  const rect = windowData.rect || [0, 0, 0, 0];
  container.innerHTML = `
    <p class="title">${windowData.title || "Unknown"}</p>
    <p class="coords">Position: ${rect[0]}, ${rect[1]}</p>
    <p class="coords">Size: ${rect[2] - rect[0]} × ${rect[3] - rect[1]}</p>
  `;
}

export async function displayMcapInfo(container, reader) {
  if (!container) return;

  const topicStats = new Map();
  for (const ch of reader.channelsById.values()) {
    topicStats.set(ch.topic, { count: 0n });
  }

  const stats = reader.statistics;
  if (stats?.channelMessageCounts) {
    for (const [chId, count] of stats.channelMessageCounts) {
      const ch = reader.channelsById.get(chId);
      if (ch && topicStats.has(ch.topic)) topicStats.get(ch.topic).count = count;
    }
  }

  const durationSec = stats ? Number(stats.messageEndTime - stats.messageStartTime) / 1e9 : 0;

  let html = '<div class="section"><div class="section-title">Topics</div>';
  for (const [topic, info] of topicStats) {
    const count = info.count > 0n ? Number(info.count).toLocaleString() : "—";
    html += `<div class="topic-row"><span class="topic-name">${topic}</span><span class="topic-count">${count}</span></div>`;
  }
  html += "</div>";
  if (durationSec > 0) html += `<div class="time-range">Duration: ${durationSec.toFixed(1)}s</div>`;
  if (stats) html += `<div class="time-range">Messages: ${Number(stats.messageCount).toLocaleString()}</div>`;

  container.innerHTML = html;
}
