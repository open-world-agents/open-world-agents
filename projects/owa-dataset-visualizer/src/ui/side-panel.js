/**
 * Side panel UI components (Window Info, MCAP Info)
 * @module ui/side-panel
 */

/**
 * Update window info display with current window state
 * @param {HTMLElement} container - Container element for window info
 * @param {Object|null} windowData - Window state data
 */
export function updateWindowInfo(container, windowData) {
  if (!container) return;

  if (!windowData) {
    container.innerHTML = '<p class="placeholder">No window data</p>';
    return;
  }

  const rect = windowData.rect || [0, 0, 0, 0];
  const width = rect[2] - rect[0];
  const height = rect[3] - rect[1];

  container.innerHTML = `
    <p class="title">${windowData.title || "Unknown"}</p>
    <p class="coords">Position: ${rect[0]}, ${rect[1]}</p>
    <p class="coords">Size: ${width} × ${height}</p>
  `;
}

/**
 * Display MCAP file information
 * @param {HTMLElement} container - Container element for MCAP info
 * @param {McapIndexedReader} reader - MCAP reader instance
 */
export async function displayMcapInfo(container, reader) {
  if (!container) return;

  // Collect topic stats
  const topicStats = new Map();
  for (const channel of reader.channelsById.values()) {
    topicStats.set(channel.topic, { count: 0n, channelId: channel.id });
  }

  // Get message counts from statistics
  const stats = reader.statistics;
  if (stats && stats.channelMessageCounts) {
    for (const [channelId, count] of stats.channelMessageCounts) {
      const channel = reader.channelsById.get(channelId);
      if (channel && topicStats.has(channel.topic)) {
        topicStats.get(channel.topic).count = count;
      }
    }
  }

  // Get time range from statistics
  const startTime = stats?.messageStartTime || 0n;
  const endTime = stats?.messageEndTime || 0n;

  // Format duration
  const durationNs = endTime - startTime;
  const durationSec = Number(durationNs) / 1e9;

  // Build HTML
  let html = '<div class="section">';
  html += '<div class="section-title">Topics</div>';

  for (const [topic, info] of topicStats) {
    const countStr = info.count > 0n ? Number(info.count).toLocaleString() : "—";
    html += `<div class="topic-row">
      <span class="topic-name">${topic}</span>
      <span class="topic-count">${countStr}</span>
    </div>`;
  }

  html += "</div>";

  if (durationSec > 0) {
    html += `<div class="time-range">Duration: ${durationSec.toFixed(1)}s</div>`;
  }

  if (stats) {
    html += `<div class="time-range">Messages: ${Number(stats.messageCount).toLocaleString()}</div>`;
  }

  container.innerHTML = html;
}

