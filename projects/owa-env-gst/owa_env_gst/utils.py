# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")
import time

import numpy as np
from gi.repository import Gst
from loguru import logger

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


# by default, common gstreamer element uses 1 second value. so timoeut must be > 1 seconds.
# The default queue size limits are 200 buffers, 10MB of data, or one second worth of data, whichever is reached first.
# https://gstreamer.freedesktop.org/documentation/coreelements/queue.html?gi-language=c
# videorate's max-closing-segment-duplication-duration is 1 second by default.
# https://gstreamer.freedesktop.org/documentation/videorate/index.html?gi-language=c#videorate:max-closing-segment-duplication-duration
def try_set_state(pipeline: Gst.Pipeline, state: Gst.State, timeout: float = 3.0):
    """
    Attempt to set pipeline state with error handling.

    Args:
        pipeline: GStreamer pipeline object
        state: Desired pipeline state
        timeout: Timeout duration in seconds

    Raises:
        Exception: If state change fails
    """
    bus = pipeline.get_bus()

    ret = pipeline.set_state(state)
    # set_state can return following values:
    # - Gst.StateChangeReturn.SUCCESS
    # - Gst.StateChangeReturn.ASYNC
    # - Gst.StateChangeReturn.FAILURE
    # - Gst.StateChangeReturn.NO_PREROLL: in live-sync's pause state
    # ref: https://gstreamer.freedesktop.org/documentation/additional/design/states.html?gi-language=c
    if ret == Gst.StateChangeReturn.FAILURE:
        msg = bus.timed_pop_filtered(Gst.SECOND * timeout, Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            logger.error(f"Failed to set pipeline to {state} state: {err} ({debug})")
        raise Exception(f"Failed to set pipeline to {state} state")
    elif ret == Gst.StateChangeReturn.ASYNC:
        wait_for_message(pipeline, Gst.MessageType.ASYNC_DONE, timeout=timeout)
    return ret


def wait_for_message(pipeline: Gst.Pipeline, message: Gst.MessageType, timeout: float = 3.0):
    """
    Wait for a specific message on the pipeline bus.

    Args:
        pipeline: GStreamer pipeline object
        message: Message type to wait for
        timeout: Timeout duration in seconds

    Raises:
        Exception: If message is not received within the timeout
    """
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(Gst.SECOND * timeout, message)
    if not msg:
        raise Exception(f"Failed to get {message} message within {timeout} seconds")
    return msg
