"""
GStreamer-based environment plugin for Open World Agents.

This module provides high-performance screen capture and recording capabilities
using GStreamer pipelines with the entry points-based discovery system.
"""

import os
import platform
import subprocess

from loguru import logger

# check if GStreamer is properly installed. TODO: multi-os support
try:
    # if os is windows
    if platform.system() == "Windows":
        subprocess.run(["gst-inspect-1.0.exe", "d3d11"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif platform.system() == "Linux":
        subprocess.run(["gst-inspect-1.0", "ximagesrc"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif platform.system() == "Darwin":
        subprocess.run(["gst-inspect-1.0", "avfvideosrc"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except Exception as e:  # noqa: F841
    logger.warning(
        "GStreamer is not properly installed or not in PATH. "
        "Please run `conda install open-world-agents::gstreamer-bundle`"
    )

# set GST_PLUGIN_PATH to the 'gst-plugins' directory in the current working directory
os.environ["GST_PLUGIN_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gst-plugins")


from . import pipeline_builder
from .gst_runner import GstPipelineRunner

__all__ = ["pipeline_builder", "GstPipelineRunner"]
