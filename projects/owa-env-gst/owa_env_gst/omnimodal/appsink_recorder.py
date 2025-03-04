# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711

import gi

gi.require_version("Gst", "1.0")


import time

from gi.repository import Gst
from loguru import logger

from owa.registry import LISTENERS

from ..gst_runner import GstPipelineRunner

if not Gst.is_initialized():
    Gst.init(None)


@LISTENERS.register("owa_env_gst/omnimodal/appsink_recorder")
class AppsinkRecorder(GstPipelineRunner):
    def on_configure(self, *, callback) -> bool:
        pipeline_description = (
            "d3d11screencapturesrc show-cursor=true do-timestamp=true ! "
            "videorate drop-only=true ! "
            "video/x-raw(memory:D3D11Memory),framerate=0/1,max-framerate=60/1 ! "
            "identity name=ts silent=true ! "
            "tee name=t "
            "t. ! queue leaky=downstream ! d3d11download ! videoconvert ! fpsdisplaysink video-sink=fakesink "
            "t. ! queue ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! nvd3d11h265enc ! h265parse ! queue ! mux. "
            "wasapi2src do-timestamp=true loopback=true low-latency=true ! audioconvert ! avenc_aac ! queue ! mux. "
            "utctimestampsrc interval=1 ! subparse ! queue ! mux. "
            "matroskamux name=mux ! filesink location=test.mkv"
        )
        logger.debug(f"Constructed pipeline: {pipeline_description}")
        super().on_configure(pipeline_description)

        identity = self.pipeline.get_by_name("ts")

        def buffer_probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo):
            buf = info.get_buffer()
            frame_time_ns = time.time_ns()
            callback(buf.pts, frame_time_ns)
            return Gst.PadProbeReturn.OK

        identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, buffer_probe_callback)
