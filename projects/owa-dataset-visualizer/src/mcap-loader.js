import { McapIndexedReader } from "@mcap/core";
import { decompress } from "fzstd";

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

/**
 * Load and parse MCAP file in browser
 */
export async function loadMcap(file) {
  const reader = await McapIndexedReader.Initialize({
    readable: new BlobReadable(file),
    decompressHandlers,
  });
  return { reader, channels: Array.from(reader.channelsById.values()) };
}

/**
 * Load MCAP from URL
 */
export async function loadMcapFromUrl(url) {
  const res = await fetch(url);
  return loadMcap(await res.blob());
}
