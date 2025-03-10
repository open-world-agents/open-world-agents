# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst
from loguru import logger

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)


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
