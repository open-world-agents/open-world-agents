# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")
import time

import numpy as np
from gi.repository import Gst

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)


def get_frame_time_ns(sample: Gst.Sample, pipeline: Gst.Pipeline) -> dict:
    """
    Calculate frame timestamp in ns adjusted by pipeline latency.

    Args:
        sample: GStreamer sample object
        pipeline: GStreamer pipeline object

    Returns:
        Dictionary containing frame_time_ns and latency
    """
    pts = sample.get_buffer().pts

    if pts == Gst.CLOCK_TIME_NONE:
        return dict(frame_time_ns=time.time_ns(), latency=0)

    clock = pipeline.get_clock()
    # https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html?gi-language=c#clock-runningtime
    # says running-time = absolute-time - base-time
    elapsed = clock.get_time() - pipeline.get_base_time()
    latency = elapsed - pts
    return dict(frame_time_ns=time.time_ns() - latency, latency=latency)


def sample_to_ndarray(sample: Gst.Sample) -> np.ndarray:
    """
    Convert GStreamer sample to numpy array.

    Args:
        sample: GStreamer sample object

    Returns:
        Numpy array containing the frame data
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width, height = structure.get_value("width"), structure.get_value("height")
    format_ = structure.get_value("format")
    assert format_ == "BGRA", f"Unsupported format: {format_}"

    frame_data = buf.extract_dup(0, buf.get_size())
    # baseline: np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
    return np.ndarray((height, width, 4), buffer=frame_data, dtype=np.uint8)


def sample_to_shape(sample: Gst.Sample) -> tuple[int, int]:
    """
    Get the shape of the frame from the GStreamer sample.

    Args:
        sample: GStreamer sample object

    Returns:
        Tuple containing the height and width of the frame
    """
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    return structure.get_value("height"), structure.get_value("width")
