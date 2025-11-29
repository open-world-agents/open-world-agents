import { McapIndexedReader } from "@mcap/core";
import { decompress } from "fzstd";

/**
 * Simple IReadable implementation for browser using ArrayBuffer
 */
class BlobReadable {
  constructor(buffer) {
    this.buffer = buffer;
    this.view = new Uint8Array(buffer);
  }

  async size() {
    return BigInt(this.buffer.byteLength);
  }

  async read(offset, length) {
    const start = Number(offset);
    const end = start + Number(length);
    return new Uint8Array(this.buffer.slice(start, end));
  }
}

/**
 * Load and parse MCAP file in browser
 */
export async function loadMcap(file) {
  const buffer = await file.arrayBuffer();
  const readable = new BlobReadable(buffer);

  const decompressHandlers = {
    zstd: (buffer, decompressedSize) => {
      const out = new Uint8Array(Number(decompressedSize));
      return decompress(buffer, out);
    },
  };

  const reader = await McapIndexedReader.Initialize({ readable, decompressHandlers });

  return {
    reader,
    startTime: reader.statistics?.messageStartTime,
    endTime: reader.statistics?.messageEndTime,
    channels: Array.from(reader.channelsById.values()),
    schemas: Array.from(reader.schemasById.values()),
  };
}

/**
 * Read messages in a time range
 */
export async function readMessages(reader, startTime, endTime) {
  const messages = [];
  for await (const msg of reader.readMessages({ startTime, endTime })) {
    const channel = reader.channelsById.get(msg.channelId);
    const schema = channel ? reader.schemasById.get(channel.schemaId) : null;

    // Decode JSON message data
    let data = null;
    if (schema?.encoding === "json" || schema?.encoding === "jsonschema") {
      try {
        data = JSON.parse(new TextDecoder().decode(msg.data));
      } catch {
        data = null;
      }
    }

    messages.push({
      topic: channel?.topic ?? "unknown",
      timestamp: msg.logTime,
      data,
      raw: msg.data,
    });
  }
  return messages;
}

/**
 * Group messages by topic
 */
export function groupByTopic(messages) {
  const groups = {};
  for (const msg of messages) {
    if (!groups[msg.topic]) groups[msg.topic] = [];
    groups[msg.topic].push(msg);
  }
  return groups;
}

