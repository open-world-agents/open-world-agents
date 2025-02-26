import os
import subprocess

# check if GStreamer is properly installed
try:
    subprocess.run(["gst-inspect-1.0.exe", "d3d11"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except Exception as e:  # noqa: F841
    raise ImportError(
        "GStreamer is not properly installed or not in PATH. "
        "Please install conda packages in `projects/owa-env-gst/environment_detail.yml`"
    )

# set GST_PLUGIN_PATH to the 'gst-plugins' directory in the current working directory
os.environ["GST_PLUGIN_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gst-plugins")

from . import gst_factory
from .gst_runner import GstPipelineRunner


def activate():
    from . import screen  # noqa
    from . import recorder  # noqa


__all__ = ["gst_factory", "activate", "GstPipelineRunner"]
