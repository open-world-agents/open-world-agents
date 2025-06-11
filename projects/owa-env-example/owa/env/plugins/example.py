"""
Plugin specification for the Example environment plugin.

This module is kept separate to avoid circular imports during plugin discovery.
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
