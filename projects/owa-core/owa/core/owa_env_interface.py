"""
Environment plugin interface for Open World Agents.

This module defines the minimal interface that environment plugins must implement.
Keep it simple - just an activate() method!
"""

from abc import ABC, abstractmethod
from types import ModuleType


class OwaEnvInterface(ModuleType, ABC):
    """
    Minimal interface for environment plugins.

    Environment plugins must implement this interface, which requires only
    an activate() method. This method should import modules that register
    components with the global registries.
    """

    @abstractmethod
    def activate(self):
        """
        Activate the environment plugin.

        This method should import modules that register callables, listeners,
        and runnables with the global registries. Keep it simple!

        Example:
            def activate():
                from . import my_callables  # noqa
                from . import my_listeners  # noqa
                from . import my_runnables  # noqa
        """
        ...


# Example plugin structure:
#
# def activate():
#     """Activate the plugin by importing modules that register components."""
#     from . import my_callables  # noqa
#     from . import my_listeners  # noqa
#     from . import my_runnables  # noqa
#
# Components should use the registries directly:
#
# from owa.core.registry import CALLABLES, LISTENERS, RUNNABLES
#
# @CALLABLES.register("component.name")
# def my_callable(): pass
#
# @LISTENERS.register("component.name")
# class MyListener(Listener): pass
#
# @RUNNABLES.register("component.name")
# class MyRunnable(Runnable): pass
