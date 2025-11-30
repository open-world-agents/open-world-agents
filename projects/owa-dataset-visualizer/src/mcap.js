/**
 * MCAP loading and time synchronization
 */
import { McapIndexedReader } from "@mcap/core";
import { decompress } from "fzstd";

// Blob-based readable for McapIndexedReader
class BlobReadable {
  constructor(blob) {
    this.blob = blob;
  }
  async size() {
    return BigInt(this.blob.size);
  }
  async read(offset, length) {
    const slice = this.blob.slice(Number(offset), Number(offset) + Number(length));
    return new Uint8Array(await slice.arrayBuffer());
  }
}

const decompressHandlers = {
  zstd: (data, size) => decompress(data, new Uint8Array(Number(size))),
};

export async function loadMcap(file) {
  const reader = await McapIndexedReader.Initialize({
    readable: new BlobReadable(file),
    decompressHandlers,
  });
  return { reader, channels: Array.from(reader.channelsById.values()) };
}

export async function loadMcapFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch MCAP: ${response.status}`);
  return loadMcap(await response.blob());
}

// Time synchronization between video and MCAP
export class TimeSync {
  constructor() {
    this.basePtsTime = null;
  }

  /** @param {bigint} logTime */
  initFromScreenMessage(logTime, data) {
    this.basePtsTime = logTime - BigInt(data?.media_ref?.pts_ns || 0);
  }

  /** @param {number} videoTimeSec @returns {bigint} */
  videoTimeToMcap(videoTimeSec) {
    if (this.basePtsTime === null) return 0n;
    return this.basePtsTime + BigInt(Math.floor(videoTimeSec * 1e9));
  }

  /** @returns {bigint} */
  getBasePtsTime() {
    return this.basePtsTime ?? 0n;
  }
}
