"""
Plugin specification for the Standard environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
"""

from owa.core.plugin_spec import PluginSpec


def _get_package_version() -> str:
    """Get the version of the owa-core package."""
    try:
        from importlib.metadata import version
    except ImportError:  # For Python <3.8
        from importlib_metadata import version

    try:
        return version("owa-core")
    except Exception:
        return "unknown"


# Plugin specification for entry points discovery
plugin_spec = PluginSpec(  # pragma: no cover
    namespace="std",
    version=_get_package_version(),
    description="Standard system components for OWA",
    author="OWA Development Team",
    components={
        "callables": {
            "time_ns": "owa.env.std.clock:time_ns",
        },
        "listeners": {
            "tick": "owa.env.std.clock:ClockTickListener",
        },
    },
)
