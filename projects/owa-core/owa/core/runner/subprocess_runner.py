"""
Subprocess management utility for running external processes as threads.

This module provides tools to run and manage external processes in a thread-safe manner,
with proper handling of termination signals and cleanup. The implementation supports
both Windows and Unix-like systems with platform-specific optimizations.
"""

import os
import signal
import subprocess
from typing import List, Optional, Union

from loguru import logger

from ..runnable import RunnableThread


def disable_ctrl_c_once():  # pragma: no cover
    """
    DO NOT USE THIS FUNCTION.
    Instead, pass CREATE_NEW_PROCESS_GROUP to the subprocess.Popen() function
    and send CTRL_BREAK_EVENT signal instead of CTRL_C_EVENT. CTRL_C_EVENT does NOT work.

    ===
    Disable Ctrl+C once. This function must be called before sending Ctrl+C to a process.
    Utilizing this function is not preferred, as this function must be called in main thread, which is not guaranteed in common.

    Related issues:
        - https://github.com/robotframework/robotframework/issues/3924
        - https://stackoverflow.com/a/60795888
    """
    # Store the original SIGINT handler to restore it later
    original_handler = signal.getsignal(signal.SIGINT)

    def enable_ctrl_c(signum, frame):
        """
        Inner function that restores the original SIGINT handler after
        intercepting the first Ctrl+C signal.
        """
        logger.info("Ctrl+C intercepted.")
        signal.signal(signal.SIGINT, original_handler)

    # Replace the SIGINT handler temporarily
    signal.signal(signal.SIGINT, enable_ctrl_c)


class SubprocessRunner(RunnableThread):
    """
    Thread-based runner for managing subprocesses with proper termination handling.

    This class extends RunnableThread to run external processes in a controlled manner,
    allowing clean shutdown and resource cleanup when the parent application terminates.
    It's particularly designed to handle Windows-specific process group behavior.

    Example usage:
    ```python
    import time

    from owa.core.runner import SubprocessRunner


    def main():
        # Create a runner for a GStreamer pipeline
        cmd = (
            "gst-launch-1.0 -e videotestsrc is-live=true ! videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4"
        ).split()
        runner = SubprocessRunner()
        try:
            # Configure and start the subprocess
            runner.configure(cmd)
            with runner.session:
                time.sleep(5)
            print("Subprocess completed.")
        except KeyboardInterrupt:
            print("Interrupted by user")
            runner.stop()
        finally:
            runner.join()


    if __name__ == "__main__":
        main()
    ```
    """

    def on_configure(self, subprocess_args: Union[List[str], str], stop_signal: Optional[int] = None) -> None:
        """
        Configure the subprocess runner with command arguments.

        Args:
            subprocess_args: List or string containing the command and arguments
                             to be executed as a subprocess.
            stop_signal: Signal to send when stopping the process. Defaults to
                        CTRL_BREAK_EVENT on Windows, SIGINT on Unix-like systems.
        """
        self._process: Optional[subprocess.Popen] = None
        self.subprocess_args = subprocess_args

        # Set default stop signal based on platform if not provided
        self._stop_signal = stop_signal or (signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)

    def loop(self):
        """
        Main execution loop that runs the subprocess and monitors its state.
        Ensures cleanup occurs even if exceptions are raised.
        """
        try:
            self._loop()
        finally:
            self.cleanup()

    def _loop(self) -> None:
        """
        Internal implementation of the monitoring loop.

        Starts the subprocess with proper flags and continuously checks:
        1. If the subprocess is still running
        2. If a stop event has been triggered

        When a stop is requested, sends the configured signal to allow clean termination.
        """
        # Start the subprocess with platform-specific flags
        if os.name == "nt":  # Windows
            self._process = subprocess.Popen(self.subprocess_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:  # Unix-like systems
            self._process = subprocess.Popen(self.subprocess_args)

        # Monitor the process and check for stop event
        while self._process.poll() is None:  # None indicates the process is still running
            if self._stop_event.is_set():
                # Stop event is set, send the configured signal for clean shutdown
                self._process.send_signal(self._stop_signal)
                break
            # Sleep briefly to avoid busy waiting while checking status
            self._stop_event.wait(0.5)

        # Grant time for the process to handle the signal gracefully
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # Process didn't terminate gracefully, cleanup will handle it

    def cleanup(self) -> None:
        """
        Clean up resources and ensure subprocess termination.

        This method is called after loop() exits and handles:
        1. Terminating the process if still running
        2. Forceful killing if termination times out
        3. Error logging for any issues during cleanup
        """
        if self._process is None:
            return
        rt = self._process.poll()
        if rt is None:
            try:
                # Try graceful termination first
                self._process.terminate()
                rt = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If graceful termination fails, forcefully kill the process
                self._process.kill()
                logger.error("SubprocessRunner was killed forcefully because of timeout.")
            except Exception:
                # Catch and log any unexpected exceptions during cleanup
                import traceback

                traceback.print_exc()
                pass  # Continue cleanup despite errors

        # Inform about the termination status
        if rt == 0:
            logger.info("SubprocessRunner terminated successfully.")
        else:
            logger.error(f"SubprocessRunner terminated with return code {rt}")

        # Clear the process reference
        self._process = None
