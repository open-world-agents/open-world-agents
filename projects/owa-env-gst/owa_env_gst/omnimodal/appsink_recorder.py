# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711

import gi

gi.require_version("Gst", "1.0")


import time

from gi.repository import Gst
from loguru import logger

from owa.registry import LISTENERS

from ..gst_runner import GstPipelineRunner
from ..pipeline_builder import appsink_recorder_pipeline

if not Gst.is_initialized():
    Gst.init(None)


@LISTENERS.register("owa_env_gst/omnimodal/appsink_recorder")
class AppsinkRecorder(GstPipelineRunner):
    def on_configure(self, *args, callback, **kwargs) -> bool:
        pipeline_description = appsink_recorder_pipeline(*args, **kwargs)
        logger.debug(f"Constructed pipeline: {pipeline_description}")
        super().on_configure(pipeline_description)

        identity = self.pipeline.get_by_name("ts")

        def buffer_probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo):
            buf = info.get_buffer()
            frame_time_ns = time.time_ns()
            callback((buf.pts, frame_time_ns))
            return Gst.PadProbeReturn.OK

        identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, buffer_probe_callback)
        self.enable_fps_display()
