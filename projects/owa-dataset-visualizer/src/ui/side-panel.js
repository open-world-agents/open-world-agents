/**
 * Side panel UI components (Window Info, MCAP Info)
 * @module ui/side-panel
 */

/**
 * Create a paragraph element with class and text
 * @param {string} className - CSS class name
 * @param {string} text - Text content
 * @returns {HTMLParagraphElement}
 */
function createParagraph(className, text) {
  const p = document.createElement("p");
  p.className = className;
  p.textContent = text;
  return p;
}

/**
 * Update window info display with current window state
 * @param {HTMLElement} container - Container element for window info
 * @param {Object|null} windowData - Window state data
 */
export function updateWindowInfo(container, windowData) {
  if (!container) return;

  container.innerHTML = "";

  if (!windowData) {
    container.append(createParagraph("placeholder", "No window data"));
    return;
  }

  const rect = windowData.rect || [0, 0, 0, 0];
  const width = rect[2] - rect[0];
  const height = rect[3] - rect[1];

  container.append(
    createParagraph("title", windowData.title || "Unknown"),
    createParagraph("coords", `Position: ${rect[0]}, ${rect[1]}`),
    createParagraph("coords", `Size: ${width} × ${height}`)
  );
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
  const durationNs = endTime - startTime;
  const durationSec = Number(durationNs) / 1e9;

  // Build DOM
  container.innerHTML = "";

  const topicsSection = document.createElement("div");
  topicsSection.className = "section";

  const topicsTitle = document.createElement("div");
  topicsTitle.className = "section-title";
  topicsTitle.textContent = "Topics";
  topicsSection.append(topicsTitle);

  for (const [topic, info] of topicStats) {
    const row = document.createElement("div");
    row.className = "topic-row";

    const nameSpan = document.createElement("span");
    nameSpan.className = "topic-name";
    nameSpan.textContent = topic;

    const countSpan = document.createElement("span");
    countSpan.className = "topic-count";
    countSpan.textContent = info.count > 0n ? Number(info.count).toLocaleString() : "—";

    row.append(nameSpan, countSpan);
    topicsSection.append(row);
  }
  container.append(topicsSection);

  if (durationSec > 0) {
    const durationEl = document.createElement("div");
    durationEl.className = "time-range";
    durationEl.textContent = `Duration: ${durationSec.toFixed(1)}s`;
    container.append(durationEl);
  }

  if (stats) {
    const messagesEl = document.createElement("div");
    messagesEl.className = "time-range";
    messagesEl.textContent = `Messages: ${Number(stats.messageCount).toLocaleString()}`;
    container.append(messagesEl);
  }
}

