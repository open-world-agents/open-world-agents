/**
 * Loading indicator management
 * @module ui/loading
 */

/**
 * Manages loading indicator visibility
 */
export class LoadingIndicator {
  /**
   * @param {string} elementId - ID of the loading indicator element
   */
  constructor(elementId = "loading-indicator") {
    /** @type {HTMLElement|null} */
    this.element = document.getElementById(elementId);
  }

  /**
   * Show the loading indicator
   */
  show() {
    this.element?.classList.remove("hidden");
  }

  /**
   * Hide the loading indicator
   */
  hide() {
    this.element?.classList.add("hidden");
  }

  /**
   * Check if the loading indicator is visible
   * @returns {boolean} True if visible
   */
  isVisible() {
    return this.element ? !this.element.classList.contains("hidden") : false;
  }
}

/**
 * Update status text in the UI
 * @param {string} message - Status message to display
 * @param {string} elementId - ID of the status element
 */
export function updateStatus(message, elementId = "status") {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = message;
  }
}

