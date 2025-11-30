/**
 * Time conversion utilities for MCAP/video synchronization
 * @module mcap/time-utils
 */

/**
 * Manages time synchronization between video playback and MCAP timestamps
 */
export class TimeSync {
  constructor() {
    /**
     * Base PTS time - the MCAP timestamp corresponding to video time 0
     * Calculated from the first screen message: logTime - media_ref.pts_ns
     * @type {bigint|null}
     */
    this.basePtsTime = null;
  }

  /**
   * Initialize base PTS time from a screen message
   * @param {bigint} logTime - Message log time
   * @param {Object} data - Screen message data containing media_ref
   */
  initFromScreenMessage(logTime, data) {
    const ptsNs = data?.media_ref?.pts_ns || 0;
    this.basePtsTime = logTime - BigInt(ptsNs);
  }

  /**
   * Convert video time (seconds) to MCAP timestamp (nanoseconds)
   * @param {number} videoTimeSec - Video playback time in seconds
   * @returns {bigint} MCAP timestamp in nanoseconds
   */
  videoTimeToMcap(videoTimeSec) {
    if (this.basePtsTime === null) return 0n;
    return this.basePtsTime + BigInt(Math.floor(videoTimeSec * 1e9));
  }

  /**
   * Convert MCAP timestamp (nanoseconds) to video time (seconds)
   * @param {bigint} mcapTime - MCAP timestamp in nanoseconds
   * @returns {number} Video playback time in seconds
   */
  mcapTimeToVideo(mcapTime) {
    if (this.basePtsTime === null) return 0;
    return Number(mcapTime - this.basePtsTime) / 1e9;
  }

  /**
   * Get the base PTS time
   * @returns {bigint} Base PTS time or 0n if not initialized
   */
  getBasePtsTime() {
    return this.basePtsTime ?? 0n;
  }

  /**
   * Check if time sync is initialized
   * @returns {boolean} True if basePtsTime is set
   */
  isInitialized() {
    return this.basePtsTime !== null;
  }
}

