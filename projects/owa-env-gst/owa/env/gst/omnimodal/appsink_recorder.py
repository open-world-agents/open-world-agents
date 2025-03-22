# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711

import gi

gi.require_version("Gst", "1.0")


import time
from pathlib import Path

from gi.repository import Gst
from loguru import logger

from owa.core.registry import LISTENERS

from ..gst_runner import GstPipelineRunner
from ..msg import ScreenEmitted
from ..pipeline_builder import appsink_recorder_pipeline

if not Gst.is_initialized():
    Gst.init(None)


@LISTENERS.register("owa.env.gst/omnimodal/appsink_recorder")
class AppsinkRecorder(GstPipelineRunner):
    def on_configure(self, filesink_location, *args, callback, **kwargs) -> bool:
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

        def buffer_probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo):
            buf = info.get_buffer()
            frame_time_ns = time.time_ns()

            clock = self.pipeline.get_clock()
            elapsed = clock.get_time() - self.pipeline.get_base_time()
            latency = elapsed - buf.pts

            # warn if latency is too high, e.g. > 30ms
            if latency > 30 * Gst.MSECOND:
                logger.warning(f"High latency: {latency / Gst.MSECOND:.2f}ms")

            callback(ScreenEmitted(path=filesink_location, pts=buf.pts, utc_ns=frame_time_ns))
            return Gst.PadProbeReturn.OK

        identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, buffer_probe_callback)
        self.enable_fps_display()
