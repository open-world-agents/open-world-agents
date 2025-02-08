from fractions import Fraction
from typing import Optional

from owa.registry import CALLABLES, activate_module

# src, convert, enc, mux


def matroskamux(srcs: list[str]):
    muxer_name = "mux"  # be aware that the name of the muxer is hardcoded
    return " ".join((src + "queue ! mux. ") for src in srcs) + f"matroskamux name={muxer_name} ! "


def tee(src: str, sinks: list[str]):
    tee_name = "t"  # be aware that the name of the tee is hardcoded
    return f"{src}tee name={tee_name} " + " ".join((f"t. ! {sink}" for sink in sinks))


def screen_src(*, fps: float = 60, window_name: Optional[str] = None, monitor_idx: Optional[int] = None):
    src_parameter = ""
    if window_name is not None:
        activate_module("owa_env_desktop")
        window = CALLABLES["window.get_window_by_title"](window_name)
        src_parameter = f" window-handle={window.hWnd}"

    if monitor_idx is not None:
        src_parameter += f" monitor-index={monitor_idx}"

    framerate = f"framerate=0/1,max-framerate={Fraction(fps).limit_denominator()}"

    return (
        f"d3d11screencapturesrc show-cursor=true {src_parameter} do-timestamp=true ! "
        f"videorate drop-only=true ! video/x-raw(memory:D3D11Memory),{framerate} ! "
    )


def screen_enc():
    return "d3d11convert ! mfh264enc ! h264parse ! "


def screen_to_fpsdisplaysink():
    return "d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink"


def screen_to_appsink():
    return "d3d11download ! videoconvert ! video/x-raw,format=BGRA ! appsink name=appsink sync=true max-buffers=1 drop=true emit-signals=true"


def audio_src():
    return "wasapi2src do-timestamp=true loopback=true low-latency=true ! audioconvert ! "


def audio_enc():
    return "mfaacenc ! "


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

    return f"{matroskamux(srcs)} filesink location={filesink_location}"


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
