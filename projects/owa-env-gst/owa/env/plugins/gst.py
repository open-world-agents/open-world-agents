"""
Plugin specification for the GStreamer environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
"""

from owa.core.plugin_spec import PluginSpec


def _get_package_version() -> str:
    """Get the version of the owa-env-gst package."""
    try:
        from importlib.metadata import version
    except ImportError:  # For Python <3.8
        from importlib_metadata import version

    try:
        return version("owa-env-gst")
    except Exception:
        return "unknown"


# Plugin specification for entry points discovery
plugin_spec = PluginSpec(
    namespace="gst",
    version=_get_package_version(),
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
