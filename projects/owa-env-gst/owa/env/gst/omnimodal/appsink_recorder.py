# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711

import gi

gi.require_version("Gst", "1.0")


import time
from pathlib import Path
from typing import Any, Callable

from gi.repository import Gst
from loguru import logger

from owa.msgs.desktop.screen import ScreenCaptured

from ..gst_runner import GstPipelineRunner
from ..pipeline_builder import appsink_recorder_pipeline

if not Gst.is_initialized():
    Gst.init(None)


class AppsinkRecorder(GstPipelineRunner):
    """
    High-performance screen recorder using GStreamer appsink for real-time processing.

    This recorder captures screen content and saves it to a file while providing
    real-time frame notifications through a callback mechanism. It supports
    hardware acceleration and various output formats.

    Examples:
        Basic screen recording to file:

        >>> def on_frame(screen_data):
        ...     print(f"Recording frame at {screen_data.utc_ns}")
        >>>
        >>> recorder = AppsinkRecorder()
        >>> recorder.configure(
        ...     filesink_location="output.mkv",
        ...     callback=on_frame
        ... )
        >>> recorder.start()

        Recording with custom resolution:

        >>> recorder.configure(
        ...     filesink_location="recording.mkv",
        ...     callback=my_callback,
        ...     width=1920,
        ...     height=1080
        ... )
    """

    def on_configure(self, filesink_location: str, *args: Any, callback: Callable, **kwargs: Any) -> None:
        """
        Configure the appsink recorder with output location and callback.

        Args:
            filesink_location: Path where the recording will be saved.
            *args: Additional positional arguments for pipeline configuration.
            callback: Function to call for each recorded frame.
            **kwargs: Additional keyword arguments for pipeline configuration.

        Returns:
            None: Configuration is applied to the recorder instance.
        """
        # if filesink_location does not exist, create it and warn the user
        if not Path(filesink_location).parent.exists():
            Path(filesink_location).parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Output directory {filesink_location} does not exist. Creating it.")

        # convert to posix path. this is required for gstreamer executable.
        filesink_location = Path(filesink_location).as_posix()

        pipeline_description = appsink_recorder_pipeline(filesink_location, *args, **kwargs)
        logger.debug(f"Constructed pipeline: {pipeline_description}")
        super().on_configure(pipeline_description)

        identity = self.pipeline.get_by_name("ts")

        notified_shape = None

        def parse_shape_from_scale():
            """Parse the shape from the d3d11scale element."""
            scale = self.pipeline.get_by_name("d3d11scale0")
            # Get the source and sink capabilities
            sink_caps = scale.get_static_pad("sink").get_current_caps()
            src_caps = scale.get_static_pad("src").get_current_caps()
            if sink_caps and src_caps:
                sink_structure = sink_caps.get_structure(0)
                src_structure = src_caps.get_structure(0)
                return (sink_structure.get_value("width"), sink_structure.get_value("height")), (
                    src_structure.get_value("width"),
                    src_structure.get_value("height"),
                )
            logger.warning("Failed to get sink or source capabilities.")
            return None, None

        def buffer_probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo):
            """Callback function to handle buffer probe events."""

            nonlocal notified_shape
            buf = info.get_buffer()
            frame_time_ns = time.time_ns()

            clock = self.pipeline.get_clock()
            elapsed = clock.get_time() - self.pipeline.get_base_time()
            latency = elapsed - buf.pts

            # warn if latency is too high, e.g. > 100ms
            if latency > 100 * Gst.MSECOND:
                logger.warning(f"High latency: {latency / Gst.MSECOND:.2f}ms")

            original_shape, shape = parse_shape_from_scale()
            if notified_shape != (original_shape, shape):
                logger.success(f"Video's original shape: {original_shape}, rescaled shape: {shape}")
                notified_shape = (original_shape, shape)

            # Create ScreenCaptured with external video reference
            screen_captured = ScreenCaptured(
                utc_ns=frame_time_ns,
                source_shape=original_shape,
                shape=shape,
                media_ref=dict(type="external_video", path=filesink_location, pts_ns=buf.pts),
            )
            callback(screen_captured)
            return Gst.PadProbeReturn.OK

        identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, buffer_probe_callback)
        self.enable_fps_display()
