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


from .callable import Callable
from .runnable import RunnableMixin, RunnableThread


class ListenerMixin(RunnableMixin):
    """ListenerMixin provides the interface for the environment to call the user-defined function."""

    def __init__(self, callback: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback


class ListenerThread(ListenerMixin, RunnableThread): ...


Listener = ListenerThread  # Default to ListenerThread

# TODO: Synchronous event listening design, as https://pynput.readthedocs.io/en/latest/keyboard.html#synchronous-event-listening-for-the-keyboard-listener
