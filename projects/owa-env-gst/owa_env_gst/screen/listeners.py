import gi

gi.require_version("Gst", "1.0")

# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import threading
import time

import numpy as np
from gi.repository import GLib, Gst
from owa import Callable, Listener
from owa.registry import LISTENERS
from pydantic import BaseModel

from ..gst_factory import screen_capture_pipeline

Gst.init(None)

"""
(function) def screen_capture_pipeline(
    fps: float = 60,
    window_name: str | None = None,
    monitor_idx: int | None = None
) -> str
"""


class FrameStamped(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    timestamp_ns: int
    frame_arr: np.ndarray  # [W, H, BGRA]


@LISTENERS.register("screen")
class ScreenListener(Listener):
    """
    A self-contained screen listener that captures the screen using a GStreamer pipeline.
    When a frame is captured (via the appsink 'new-sample' signal), it is converted into a numpy
    array and then wrapped with a FrameStamped object. The user-provided callback (passed during
    instantiation) is then called with the FrameStamped object.
    """

    def __init__(self, callback: Callable[[FrameStamped], None]):
        super().__init__(callback=callback)
        self.pipeline = None
        self.appsink = None
        self.loop = None
        self._loop_thread = None

    @property
    def latency(self): ...  # TODO

    @property
    def fps(self): ...  # TODO

    def on_configure(self, *, fps: float = 60, window_name: str | None = None, monitor_idx: int | None = None):
        """
        Configure the GStreamer pipeline for screen capture.

        Keyword Arguments:
            fps (float): Frames per second.
            window_name (str | None): (Optional) specific window to capture.
            monitor_idx (int | None): (Optional) specific monitor index.
        """
        # Construct the pipeline description
        pipeline_description = screen_capture_pipeline(fps, window_name, monitor_idx)
        self.pipeline = Gst.parse_launch(pipeline_description)

        # Get the appsink element by name and set its properties (redundant if already set in pipeline desc.)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", True)
        # Connect the "new-sample" signal to our callback
        self.appsink.connect("new-sample", self.__on_new_sample)

        # Create a GLib mainloop to handle the GStreamer bus events
        self.loop = GLib.MainLoop()
        return True

    def on_activate(self):
        """
        Start the GStreamer pipeline in a dedicated thread.
        """
        # Clear any stop-flag if using one
        self._loop_thread = threading.Thread(target=self._run, daemon=True)
        self._loop_thread.start()
        return True

    def _run(self):
        """Internal run method that sets the pipeline to PLAYING and starts the GLib main loop."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            bus = self.pipeline.get_bus()
            msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR)
            if msg:
                err, debug = msg.parse_error()
                print(f"Failed to set pipeline to PLAYING state: {err} ({debug})")
            return
        self.loop.run()

    def on_deactivate(self):
        """
        Stop the pipeline gracefully.
        """
        # Send End-Of-Stream event to the pipeline.
        self.pipeline.send_event(Gst.Event.new_eos())
        bus = self.pipeline.get_bus()
        while True:
            msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if msg:
                if msg.type == Gst.MessageType.EOS:
                    print("Received EOS signal, shutting down gracefully.")
                    break
                elif msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print("Error received:", err, debug)
                    break
        self.pipeline.set_state(Gst.State.NULL)

        self.loop.quit()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join()
        return True

    def on_cleanup(self):
        """
        Clean up resources.
        """
        self.pipeline = None
        self.appsink = None
        self.loop = None
        self._loop_thread = None
        return True

    def on_shutdown(self):
        """
        Optional shutdown process.
        """
        return True

    def on_error(self):
        """
        Handle errors if needed.
        """
        return True

    def __get_frame_time_ns(self, pts: int) -> int:
        """
        Calculate the frame timestamp in ns adjusted by pipeline latency.
        This mimics the latency correction from the legacy code.

        Parameters:
            pts (int): The presentation timestamp of the buffer.

        Returns:
            int: A corrected timestamp in nanoseconds.
        """
        if pts == Gst.CLOCK_TIME_NONE:
            return time.time_ns()
        clock = self.pipeline.get_clock()
        # Calculate elapsed time since the pipeline went to PLAYING state.
        elapsed = clock.get_time() - self.pipeline.get_base_time()
        latency = elapsed - pts
        # Adjust current system time by the computed latency.
        return time.time_ns() - latency

    def __on_new_sample(self, sink) -> Gst.FlowReturn:
        """
        This callback is connected to the appsink 'new-sample' signal.
        It extracts the data from the sample, converts it into a numpy array,
        and then calls the user-supplied callback with a FrameStamped message.
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            print("Received null sample.")
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        format_ = structure.get_value("format")
        if format_ != "BGRA":
            print(f"Unsupported format: {format_}")
            return Gst.FlowReturn.ERROR

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            # Create a numpy array from the raw data.
            frame_data = mapinfo.data
            # reshape to (height, width, 4), BGRA.
            frame_arr = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
            timestamp_ns = self.__get_frame_time_ns(buf.pts)
            # Create the message and call the user callback.
            message = FrameStamped(timestamp_ns=timestamp_ns, frame_arr=frame_arr)
            self.callback(message)
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK
