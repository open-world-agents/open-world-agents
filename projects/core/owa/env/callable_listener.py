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


from typing import Callable

from ..node import Node


class Listener(Node):
    """
    The Listener class is a subclass of the Node class. It is used to define the listener objects that listen to the input.

    Example:
    ```python
    class CustomListener(Listener):
        def __init__(self, callback: Callable):
            super().__init__()
            self.callback = callback
            (add your code here)
    ```
    """


# TODO: Synchronous event listening design, as https://pynput.readthedocs.io/en/latest/keyboard.html#synchronous-event-listening-for-the-keyboard-listener
