"""
This module provides a set of functions to construct GStreamer pipelines
for screen capturing and recording.

TODO: implement macOS and Linux support, as https://github.com/open-world-agents/desktop-env/blob/31b44e759a22dee20f08a5c61a345e6d76b383a2/src/desktop_env/windows_capture/gst_pipeline.py

TODO: supports various encoder depending on the platform and hardware
BUG: mfh264enc only takes even-sized input, which causes d3d11convert to resize, which causes a char to be vague
NOTE: If your desktop suffer with resource consumption, you may try h264 instead of h265.

BUG: using mfaacenc along with nvd3d11h265enc causes a crash
"""

from typing import Optional

from .element import Element
from .factory import ElementFactory


# Copied from subprocess_recorder_pipeline
def appsink_recorder_pipeline(
    filesink_location: str,
    *,
    record_audio: bool = True,
    record_video: bool = True,
    record_timestamp: bool = True,
    enable_fpsdisplaysink: bool = True,
    show_cursor: bool = True,
    fps: float = 60,
    window_name: Optional[str] = None,
    monitor_idx: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    additional_properties: Optional[dict] = None,
) -> str:
    """Construct a GStreamer pipeline for screen capturing.
    Args:
        filesink_location: The location of the output file.
        record_audio: Whether to record audio.
        record_video: Whether to record video.
        record_timestamp: Whether to record timestamp.
        enable_fpsdisplaysink: Whether to enable fpsdisplaysink.
        fps: The frame rate of the video.
        window_name: The name of the window to capture. If None, the entire screen will be captured.
        monitor_idx: The index of the monitor to capture. If None, the primary monitor will be captured.
        width: The width of the video. If None, the width will be determined by the source.
        height: The height of the video. If None, the height will be determined by the source.
    """
    assert filesink_location.endswith(".mkv"), "Only Matroska (.mkv) files are supported now."

    src = Element("")
    if record_video:
        screen_src = (
            ElementFactory.d3d11screencapturesrc(
                show_cursor=show_cursor,
                fps=fps,
                window_name=window_name,
                monitor_idx=monitor_idx,
                width=width,
                height=height,
                additional_properties=additional_properties,
            )
            >> "identity name=ts silent=true"
            >> ElementFactory.tee(name="t")
        )
        if enable_fpsdisplaysink:
            screen_src |= (
                "t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink"
            )
        # in here, conversion to NV12 is required for nvd3d11h265enc to prevent alpha channel ignoring.
        # also, usage of mfh264enc is avoided to prevent forceful odd-to-even resize.
        # Related issue: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4124
        screen_src |= "t. ! queue ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! queue ! mux."
        src |= screen_src

    if record_audio:
        src |= ElementFactory.wasapi2src(window_name=window_name) >> "audioconvert ! avenc_aac ! queue ! mux."
    if record_timestamp:
        src |= "utctimestampsrc interval=1 ! subparse ! queue ! mux."

    pipeline = src | ElementFactory.matroskamux(name="mux") >> ElementFactory.filesink(location=filesink_location)
    return str(pipeline)


def subprocess_recorder_pipeline(
    filesink_location: str,
    *,
    record_audio: bool = True,
    record_video: bool = True,
    record_timestamp: bool = True,
    enable_fpsdisplaysink: bool = True,
    show_cursor: bool = True,
    fps: float = 60,
    window_name: Optional[str] = None,
    monitor_idx: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    additional_properties: Optional[dict] = None,
) -> str:
    """Construct a GStreamer pipeline for screen capturing.
    Args:
        filesink_location: The location of the output file.
        record_audio: Whether to record audio.
        record_video: Whether to record video.
        record_timestamp: Whether to record timestamp.
        enable_fpsdisplaysink: Whether to enable fpsdisplaysink.
        fps: The frame rate of the video.
        window_name: The name of the window to capture. If None, the entire screen will be captured.
        monitor_idx: The index of the monitor to capture. If None, the primary monitor will be captured.
        width: The width of the video. If None, the width will be determined by the source.
        height: The height of the video. If None, the height will be determined by the source.
    """
    assert filesink_location.endswith(".mkv"), "Only Matroska (.mkv) files are supported now."

    src = Element("")
    if record_video:
        screen_src = ElementFactory.d3d11screencapturesrc(
            show_cursor=show_cursor,
            fps=fps,
            window_name=window_name,
            monitor_idx=monitor_idx,
            width=width,
            height=height,
            additional_properties=additional_properties,
        ) >> ElementFactory.tee(name="t")
        if enable_fpsdisplaysink:
            screen_src |= (
                "t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink"
            )
        # in here, conversion to NV12 is required for nvd3d11h265enc to prevent alpha channel ignoring.
        # also, usage of mfh264enc is avoided to prevent forceful odd-to-even resize.
        # Related issue: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4124
        screen_src |= "t. ! queue ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! queue ! mux."
        src |= screen_src

    if record_audio:
        src |= ElementFactory.wasapi2src(window_name=window_name) >> "audioconvert ! avenc_aac ! queue ! mux."
    if record_timestamp:
        src |= "utctimestampsrc interval=1 ! subparse ! queue ! mux."

    pipeline = src | ElementFactory.matroskamux(name="mux") >> ElementFactory.filesink(location=filesink_location)
    return str(pipeline)


def screen_capture_pipeline(
    show_cursor: bool = True,
    fps: float = 60,
    window_name: Optional[str] = None,
    monitor_idx: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    additional_properties: Optional[dict] = None,
) -> str:
    """
    Construct a GStreamer pipeline for screen capturing with appsink.
    Args:
        fps: The frame rate of the video.
        window_name: The name of the window to capture. If None, the entire screen will be captured.
        monitor_idx: The index of the monitor to capture. If None, the primary monitor will be captured.
        width: The width of the video. If None, the width will be determined by the source.
        height: The height of the video. If None, the height will be determined by the source.
        additional_properties: Additional properties to pass to the d3d11screencapturesrc element.
    """

    src = ElementFactory.d3d11screencapturesrc(
        show_cursor=show_cursor,
        fps=fps,
        window_name=window_name,
        monitor_idx=monitor_idx,
        width=width,
        height=height,
        additional_properties=additional_properties,
    )
    pipeline = src >> (
        "queue leaky=downstream ! d3d11download ! videoconvert ! "
        "video/x-raw,format=BGRA ! appsink name=appsink sync=false max-buffers=1 "
        "drop=true emit-signals=true wait-on-eos=false"
    )
    return str(pipeline)
