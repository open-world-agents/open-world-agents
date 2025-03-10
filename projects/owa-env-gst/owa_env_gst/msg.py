import numpy as np

from owa.message import OWAMessage


class FrameStamped(OWAMessage):
    _type = "owa_env_gst.msg.FrameStamped"

    model_config = {"arbitrary_types_allowed": True}

    timestamp_ns: int
    frame_arr: np.ndarray  # [W, H, BGRA]


class ScreenEmitted(OWAMessage):
    _type = "owa_env_gst.msg.ScreenEmitted"

    # Path to the stream, e.g. output.mkv
    path: str
    # Time since stream start as nanoseconds.
    pts: int
    # Time since epoch as nanoseconds.
    utc_ns: int
