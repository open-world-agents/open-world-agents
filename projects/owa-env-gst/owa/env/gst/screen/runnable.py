import threading
from collections import deque

from owa.core.registry import RUNNABLES

from ..msg import FrameStamped
from .listeners import ScreenListener


@RUNNABLES.register("screen_capture")
class ScreenCapture(ScreenListener):
    """
    Screen capture thread using GStreamer pipeline.

    Captures screen frames continuously and makes the latest frame
    available through a thread-safe interface.

    Example:
    ```python
    from owa.core.registry import RUNNABLES, activate_module

    activate_module("owa.env.gst")
    screen_capture = RUNNABLES["screen_capture"]().configure(fps=60)

    with screen_capture.session:
        for _ in range(10):
            frame = screen_capture.grab()
            print(f"Shape: {frame.frame_arr.shape}")
    ```
    """

    def on_configure(self, *args, **kwargs):
        """
        Configure and start the screen listener.

        Args:
            fps (float): Frames per second for capture.
            window_name (str, optional): Window to capture. If None, captures entire screen.
            monitor_idx (int, optional): Monitor index to capture.
        """
        self.queue = deque(maxlen=1)  # Holds the most recent frame
        self._event = threading.Event()

        def on_frame(frame):
            self.queue.append(frame)
            self._event.set()

        super().on_configure(callback=on_frame, *args, **kwargs)
        return self

    def grab(self) -> FrameStamped:
        """
        Get the most recent frame (blocks until frame is available).

        Returns:
            FrameStamped: Latest captured frame with timestamp.

        Raises:
            TimeoutError: If no frame is received within 1 second.
        """
        if not self._event.wait(timeout=1.0):
            raise TimeoutError("Timeout waiting for frame")
        self._event.clear()
        return self.queue[0]
