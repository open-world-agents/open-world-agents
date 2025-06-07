"""
Standard environment components for Open World Agents.

This module provides basic system components like clock functionality
using the entry points-based discovery system.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points discovery
plugin_spec = PluginSpec(
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
