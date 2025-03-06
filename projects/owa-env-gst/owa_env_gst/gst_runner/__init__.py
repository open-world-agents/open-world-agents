from owa.registry import RUNNABLES

from .extensions import AppsinkExtension, FPSDisplayExtension, SeekExtension
from .gst_runner import BaseGstPipelineRunner


@RUNNABLES.register("gst_pipeline_runner")
class GstPipelineRunner(BaseGstPipelineRunner, AppsinkExtension, FPSDisplayExtension, SeekExtension): ...
