import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from owa import Runnable
from owa.registry import RUNNABLES

from .gst_factory import recorder_pipeline


def disable_ctrl_c_once():
    """Disable Ctrl+C once. This function must be called before sending Ctrl+C to a process.
    Utilizing this function is not preferred, as this function must be called in main thread, which is not guaranteed in common.

    Otherwise, you may pass CREATE_NEW_PROCESS_GROUP to the subprocess.Popen() function. After that, you must pass CTRL_BREAK_EVENT instead of CTRL_C_EVENT(it does NOT work).
    Related issues:
        - https://github.com/robotframework/robotframework/issues/3924
        - https://stackoverflow.com/a/60795888
    """
    original_handler = signal.getsignal(signal.SIGINT)

    def enable_ctrl_c(signum, frame):
        logger.info("Ctrl+C intercepted.")
        signal.signal(signal.SIGINT, original_handler)

    signal.signal(signal.SIGINT, enable_ctrl_c)


@RUNNABLES.register("screen/recorder")
class ScreenRecorder(Runnable):
    """A ScreenRecorder Runnable that records video and/or audio using a GStreamer pipeline."""

    def configure(
        self,
        filesink_location: str,
        record_audio: bool = True,
        record_video: bool = True,
        record_timestamp: bool = True,
        enable_appsink: bool = False,
        enable_fpsdisplaysink: bool = True,
        show_cursor: bool = True,
        fps: float = 60,
        window_name: Optional[str] = None,
        monitor_idx: Optional[int] = None,
        additional_args: Optional[str] = None,
    ):
        """Prepare the GStreamer pipeline command."""
        self._process = None  # This will hold the subprocess running the pipeline
        self._pipeline_cmd = None  # The pipeline command to execute

        # if filesink_location does not exist, create it and warn the user
        if not Path(filesink_location).parent.exists():
            Path(filesink_location).parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Output directory {filesink_location} does not exist. Creating it.")

        pipeline_description = recorder_pipeline(
            filesink_location=filesink_location,
            record_audio=record_audio,
            record_video=record_video,
            record_timestamp=record_timestamp,
            enable_appsink=enable_appsink,
            enable_fpsdisplaysink=enable_fpsdisplaysink,
            show_cursor=show_cursor,
            fps=fps,
            window_name=window_name,
            monitor_idx=monitor_idx,
            additional_args=additional_args,
        )
        self._pipeline_cmd = f"gst-launch-1.0.exe -e -v {pipeline_description}"

    def loop(self):
        """Main loop. This method must be interruptible by calling stop(), which sets the self._stop_event."""
        if self._pipeline_cmd is None:
            self.configure()
        # Start the GStreamer pipeline process with the modified environment
        self._process = subprocess.Popen(self._pipeline_cmd.split(), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        # Monitor the process and check for stop event
        while self._process.poll() is None:
            if self._stop_event.is_set():
                # Stop event is set, send eos to the pipeline
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
                break
            # Sleep briefly to avoid busy waiting
            time.sleep(0.5)
        # Wait for the process to terminate fully
        rt = self._process.wait(5)
        if rt == 0:
            logger.info("ScreenRecorder process terminated successfully.")
        else:
            logger.error(f"ScreenRecorder process terminated with return code {rt}")

    def cleanup(self):
        """Clean up resources. This method is called after loop() exits."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If the process doesn't terminate, kill it forcefully
                self._process.kill()
                logger.error("ScreenRecorder process killed forcefully.")
            except Exception:
                import traceback

                traceback.print_exc()
                pass  # Handle any other exceptions silently
        self._process = None
