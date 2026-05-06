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

from node import Node


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

# ================ Definition of the Registry class ================================
# references:
# - https://github.com/open-mmlab/mmdetection/blob/main/mmdet/registry.py
# - https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls

        return decorator

    def __getitem__(self, name: str):
        return self._registry[name]


CALLABLES = Registry()
LISTENERS = Registry()

# ================ Example of registering the CALLABLES and LISTENERS ================================

# === in owa/listeners/screen.py ===
from owa.core.registry import LISTENERS


@LISTENERS.register("screen")
class ScreenListener(Listener): ...  # TODO: implement single-image grab Callable


# === in owa/listeners/keyboard_mouse.py ===
# Register the Listener for keyboard and mouse

from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener

from owa.core.registry import LISTENERS


@LISTENERS.register("keyboard")
class KeyboardListenerWrapper(Listener):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        self.callback("keyboard.press", key)

    def on_release(self, key):
        self.callback("keyboard.release", key)


# === in owa/listeners/minecraft-specific-something.py ===
# Register the Listener for Minecraft


@LISTENERS.register("minecraft")
class YourCustomListener(Listener): ...


# === in owa/callables/keyboard_mouse.py ===
# Register the callable functions for mouse

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController

from owa.core.registry import CALLABLES

mouse_controller = MouseController()

CALLABLES.register("mouse.click")(mouse_controller.click)
CALLABLES.register("mouse.move")(mouse_controller.move)
CALLABLES.register("mouse.position")(lambda: mouse_controller.position)
CALLABLES.register("mouse.press")(mouse_controller.press)
CALLABLES.register("mouse.release")(mouse_controller.release)
CALLABLES.register("mouse.scroll")(mouse_controller.scroll)


# === in owa/callables/minecraft-specific-somethings.py ===
# Register the callable functions for minecraft

from owa.core.registry import CALLABLES


@CALLABLES.register("minecraft.get_inventory")
def get_inventory(player):
    pass


# ================ Example of using the CALLABLES and LISTENERS ================================

from owa.core.registry import CALLABLES, LISTENERS

# Get the callable function for mouse click
mouse_click = CALLABLES["mouse.click"]
mouse_click(1, 2)

inventory = CALLABLES["minecraft.get_inventory"](player="Steve")


# Get the listener for keyboard
def on_keyboard_event(event_type, key):
    print(f"Keyboard event: {event_type}, {key}")


keyboard_listener = LISTENERS["keyboard"](on_keyboard_event)
keyboard_listener.start()
