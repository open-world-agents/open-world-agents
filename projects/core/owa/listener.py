# ================ Definition of the Callable and Listener classes ================
# To deal with the state and action with regard to environment, we need to define the Callable and Listener classes.
# The Callable class is used to:
#     - define the callable that acquires the state
#     - define the callable that performs the action
# The Listener class is used to:
#     - define the listener that listens to the state
#
# Main differences between the Callable and Listener classes is where/whom the function is called.
#     - the Callable class is called by the user
#     - while the Listener class provides the interface for the environment to call the user-defined function.


from typing import Self

from .callable import Callable
from .runnable import RunnableMixin, RunnableProcess, RunnableThread


class ListenerMixin(RunnableMixin):
    """ListenerMixin provides the interface for the environment to call the user-defined function."""

    def get_callback(self) -> Callable:
        if not hasattr(self, "_callback"):
            raise AttributeError("Callback not set. Please call self.register_callback() to set the callback.")
        return self._callback

    def register_callback(self, callback: Callable) -> Self:
        self._callback = callback
        return self

    callback = property(get_callback, register_callback)

    def configure(self, *args, callback: Callable, **kwargs) -> Self:
        """Configure the listener with a callback function. `callback` keyword argument is reserved for the callback function."""
        self.register_callback(callback)
        self.on_configure(*args, **kwargs)
        return self


class ListenerThread(ListenerMixin, RunnableThread): ...


class ListenerProcess(ListenerMixin, RunnableProcess): ...


Listener = ListenerThread  # Default to ListenerThread

# TODO: Synchronous event listening design, as https://pynput.readthedocs.io/en/latest/keyboard.html#synchronous-event-listening-for-the-keyboard-listener
