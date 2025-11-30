/**
 * MCAP file loading utilities for browser environment
 * @module mcap/loader
 */

import { McapIndexedReader } from "@mcap/core";
import { decompress } from "fzstd";

/**
 * Blob-based readable interface for McapIndexedReader
 * Allows reading MCAP files from browser Blob/File objects
 */
class BlobReadable {
  /**
   * @param {Blob} blob - The blob to read from
   */
  constructor(blob) {
    this.blob = blob;
  }

  /**
   * Get the size of the blob
   * @returns {Promise<bigint>} Size in bytes
   */
  async size() {
    return BigInt(this.blob.size);
  }

  /**
   * Read a range of bytes from the blob
   * @param {bigint} offset - Start offset
   * @param {bigint} length - Number of bytes to read
   * @returns {Promise<Uint8Array>} The read bytes
   */
  async read(offset, length) {
    const slice = this.blob.slice(Number(offset), Number(offset) + Number(length));
    return new Uint8Array(await slice.arrayBuffer());
  }
}

/**
 * Decompression handlers for MCAP chunks
 * @type {Object<string, function>}
 */
const decompressHandlers = {
  /**
   * Decompress zstd-compressed data
   * @param {Uint8Array} data - Compressed data
   * @param {bigint} size - Expected uncompressed size
   * @returns {Uint8Array} Decompressed data
   */
  zstd: (data, size) => decompress(data, new Uint8Array(Number(size))),
};

/**
 * Load and parse MCAP file from a File/Blob object
 * @param {Blob|File} file - The MCAP file to load
 * @returns {Promise<{reader: McapIndexedReader, channels: Array}>} Reader and channel list
 */
export async function loadMcap(file) {
  const reader = await McapIndexedReader.Initialize({
    readable: new BlobReadable(file),
    decompressHandlers,
  });
  return {
    reader,
    channels: Array.from(reader.channelsById.values()),
  };
}

/**
 * Load MCAP file from a URL
 * @param {string} url - URL to fetch the MCAP file from
 * @returns {Promise<{reader: McapIndexedReader, channels: Array}>} Reader and channel list
 */
export async function loadMcapFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch MCAP: ${response.status} ${response.statusText}`);
  }
  const blob = await response.blob();
  return loadMcap(blob);
}

