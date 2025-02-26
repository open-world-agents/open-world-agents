import time
from queue import Queue

from owa.registry import RUNNABLES
from owa.runnable import RunnableThread

from .listeners import ScreenListener
from .msg import FrameStamped


@RUNNABLES.register("screen_capture")
class ScreenCapture(RunnableThread):
    """
    A runnable thread for capturing the screen using a GStreamer-based pipeline.
    It continuously listens for screen frames and stores them in a queue.

    Example usage:
    ```python
    from owa.registry import RUNNABLES, activate_module
    activate_module("owa_env_gst")
    screen_capture = RUNNABLES["screen_capture"]()
    screen_capture.configure()
    screen_capture.start()
    for _ in range(10):
        timestamp, frame = screen_capture.grab()
        print(timestamp, frame.shape)
    ```
    """

    def __init__(self):
        """Initialize the screen capture runnable with a frame queue and listener."""
        super().__init__()
        self.queue = Queue(maxsize=1)  # Holds the most recent frame
        self.listener = None  # Listener for capturing screen frames

    def on_configure(self, *, fps: float = 60, window_name: str | None = None, monitor_idx: int | None = None):
        """
        Configure and start the screen listener.

        Args:
            fps (float): Frames per second for capture.
            window_name (str | None): Name of the window to capture (optional).
            monitor_idx (int | None): Index of the monitor to capture (optional).
        """
        self.listener = ScreenListener()
        self.listener.configure(fps=fps, window_name=window_name, monitor_idx=monitor_idx, callback=self.on_frame)
        self.listener.start()

    def on_frame(self, frame):
        """
        Callback function for handling incoming frames.
        Stores the most recent frame in the queue, removing older ones if necessary.

        Args:
            frame: Captured frame from the listener.
        """
        if self.queue.full():
            self.queue.get()  # Remove the oldest frame to make space
        self.queue.put(frame)

    def loop(self):
        """
        Keep the thread running until stopped.
        This loop does not process frames but keeps the thread alive.
        """
        while not self._stop_event.is_set():
            time.sleep(1)  # Prevent high CPU usage by sleeping

    def cleanup(self):
        """
        Stop and clean up the screen listener before shutting down the thread.
        """
        if self.listener:
            self.listener.stop()
            self.listener.join()

    def grab(self) -> FrameStamped:
        """
        Retrieve the most recent frame from the queue.

        Returns:
            FrameStamped: The latest captured frame with a timestamp.
        """
        return self.queue.get()
