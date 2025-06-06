"""
Example environment plugin for Open World Agents.

This plugin demonstrates how to create a simple environment plugin
following the OwaEnvInterface pattern.
"""


def activate():
    """
    Activate the example environment plugin.

    This simple function imports modules that register components with the
    global registries. This is all that's needed for a basic plugin.
    """
    # Import modules to trigger registration decorators
    from . import example_callable  # noqa
    from . import example_listener  # noqa
    from . import example_runnable  # noqa
