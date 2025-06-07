"""
Plugin specification for the GStreamer environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points discovery
plugin_spec = PluginSpec(
    namespace="gst",
    version="0.3.9.post1",
    description="High-performance GStreamer-based screen capture and recording plugin",
    author="OWA Development Team",
    components={
        "listeners": {
            "screen": "owa.env.gst.screen.listeners:ScreenListener",
            "omnimodal.appsink_recorder": "owa.env.gst.omnimodal.appsink_recorder:AppsinkRecorder",
        },
        "runnables": {
            "screen_capture": "owa.env.gst.screen.runnable:ScreenCapture",
            "omnimodal.subprocess_recorder": "owa.env.gst.omnimodal.subprocess_recorder:SubprocessRecorder",
        },
    },
)
