"""
Plugin specification for the Standard environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points discovery
plugin_spec = PluginSpec(  # pragma: no cover
    namespace="std",
    version="0.1.0",
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
