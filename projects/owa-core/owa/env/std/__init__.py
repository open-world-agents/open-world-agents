"""
Standard environment plugin for Open World Agents.

This plugin provides essential time-related functionality including
clock functions and tick listeners.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points-based discovery
plugin_spec = PluginSpec(
    namespace="std",
    version="0.3.9",
    description="Standard environment plugin with time utilities",
    author="Open World Agents Team",
    components={
        "callables": {
            "time_ns": "owa.env.std.clock:time_ns_func",
        },
        "listeners": {
            "tick": "owa.env.std.clock:ClockTickListener",
        },
    },
)


# Legacy activate() function removed - plugin now uses entry points discovery
