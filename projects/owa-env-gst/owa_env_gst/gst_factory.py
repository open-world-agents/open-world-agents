"""
This module provides a set of functions to construct GStreamer pipelines
for screen capturing and recording.

TODO: implement macOS and Linux support, as https://github.com/open-world-agents/desktop-env/blob/31b44e759a22dee20f08a5c61a345e6d76b383a2/src/desktop_env/windows_capture/gst_pipeline.py
"""

from fractions import Fraction
from typing import Optional

from owa.registry import CALLABLES, activate_module

# src, convert, enc, mux


def matroskamux(srcs: list[str]):
    muxer_name = "mux"  # be aware that the name of the muxer is hardcoded
    return " ".join((src + "queue ! mux.") for src in srcs) + f" matroskamux name={muxer_name} ! "


def tee(src: str, sinks: list[str]):
    tee_name = "t"  # be aware that the name of the tee is hardcoded
    return f"{src}tee name={tee_name} " + " ".join((f"t. ! {sink}" for sink in sinks))


def screen_src(*, fps: float = 60, window_name: Optional[str] = None, monitor_idx: Optional[int] = None):
    src_parameter = ["show-cursor=true", "do-timestamp=true"]
    if window_name is not None:
        activate_module("owa_env_desktop")
        window = CALLABLES["window.get_window_by_title"](window_name)
        src_parameter += [f"window-handle={window.hWnd}"]

    if monitor_idx is not None:
        src_parameter += [f"monitor-index={monitor_idx}"]

    src_parameter = " ".join(src_parameter)
    frac = Fraction(fps).limit_denominator()
    framerate = f"framerate=0/1,max-framerate={frac.numerator}/{frac.denominator}"

    return (
        f"d3d11screencapturesrc {src_parameter} ! "
        f"videorate drop-only=true ! video/x-raw(memory:D3D11Memory),{framerate} ! "
    )


def screen_enc():
    # BUG: mfh264enc only takes even-sized input, which causes d3d11convert to resize, which causes a char to be vague
    # return "d3d11convert ! mfh264enc ! h264parse ! "

    # CAUTION: If your desktop suffer with resource consumption, you may try h264 instead of h265.
    return "d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! "


def screen_to_fpsdisplaysink():
    return "d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink"


def screen_to_appsink():
    return "d3d11download ! videoconvert ! video/x-raw,format=BGRA ! appsink name=appsink sync=true max-buffers=1 drop=true emit-signals=true"


def audio_src():
    return "wasapi2src do-timestamp=true loopback=true low-latency=true ! audioconvert ! "


def audio_enc():
    # BUG: using mfaacenc along with nvd3d11h265enc causes a crash
    # return "mfaacenc ! "

    return "avenc_aac ! "


def utctimestampsrc():
    return "utctimestampsrc interval=1 ! subparse ! "


def recorder_pipeline(
    filesink_location: str,
    *,
    record_audio: bool = True,
    record_video: bool = True,
    record_timestamp: bool = True,
    enable_appsink: bool = False,
    enable_fpsdisplaysink: bool = True,
    fps: float = 60,
    window_name: Optional[str] = None,
    monitor_idx: Optional[int] = None,
) -> str:
    """Construct a GStreamer pipeline for screen capturing.
    Args:
        filesink_location: The location of the output file.
        record_audio: Whether to record audio.
        record_video: Whether to record video.
        record_timestamp: Whether to record timestamp.
        enable_appsink: Whether to enable appsink.
        enable_fpsdisplaysink: Whether to enable fpsdisplaysink.
        fps: The frame rate of the video.
        window_name: The name of the window to capture. If None, the entire screen will be captured.
        monitor_idx: The index of the monitor to capture. If None, the primary monitor will be captured.
    """
    assert filesink_location.endswith(".mkv"), "Only Matroska (.mkv) files are supported now."

    srcs = []
    if record_video:
        _screen_src = screen_src(fps=fps, window_name=window_name, monitor_idx=monitor_idx)
        sinks = []
        if enable_appsink:
            sinks.append("queue leaky=downstream ! " + screen_to_appsink())
        if enable_fpsdisplaysink:
            sinks.append("queue leaky=downstream ! " + screen_to_fpsdisplaysink())
        sinks.append("queue ! " + screen_enc())
        srcs.append(tee(_screen_src, sinks))

    if record_audio:
        srcs.append(audio_src() + audio_enc())
    if record_timestamp:
        srcs.append(utctimestampsrc())

    return f"{matroskamux(srcs)}filesink location={filesink_location}"


def screen_capture_pipeline(
    fps: float = 60, window_name: Optional[str] = None, monitor_idx: Optional[int] = None
) -> str:
    """
    Construct a GStreamer pipeline for screen capturing with appsink.
    Args:
        fps: The frame rate of the video.
        window_name: The name of the window to capture. If None, the entire screen will be captured.
        monitor_idx: The index of the monitor to capture. If None, the primary monitor will be captured.
    """

    src = screen_src(fps=fps, window_name=window_name, monitor_idx=monitor_idx)
    sinks = ["queue leaky=downstream ! " + screen_to_appsink()]
    return tee(src, sinks)
