"""
Example environment plugin for Open World Agents.

This plugin demonstrates how to create a plugin using the
entry points-based discovery system.
"""

from owa.core.plugin_spec import PluginSpec

# Plugin specification for entry points discovery
plugin_spec = PluginSpec(
    namespace="example",
    version="0.1.0",
    description="Example environment plugin demonstrating the plugin system",
    author="OWA Development Team",
    components={
        "callables": {
            "callable": "owa.env.example.example_callable:ExampleCallable",
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
