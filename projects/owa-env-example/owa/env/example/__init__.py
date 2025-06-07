"""
Example environment plugin for Open World Agents.

This plugin demonstrates how to create a simple environment plugin
following the OwaEnvInterface pattern and the new entry points system.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points-based discovery
plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin for Open World Agents",
    author="Open World Agents Team",
    components={
        "callables": {
            "callable": "owa.env.example.example_callable:example_callable",
            "print": "owa.env.example.example_callable:example_print",
            "add": "owa.env.example.example_callable:example_add",
        },
        "listeners": {
            "listener": "owa.env.example.example_listener:ExampleListener",
            "timer": "owa.env.example.example_listener:ExampleTimerListener",
        },
        "runnables": {
            "runnable": "owa.env.example.example_runnable:ExampleRunnable",
            "counter": "owa.env.example.example_runnable:ExampleCounterRunnable",
        },
    },
)


# Legacy activate() function removed - plugin now uses entry points discovery
