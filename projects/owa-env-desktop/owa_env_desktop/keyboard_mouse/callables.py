import time

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from owa.registry import CALLABLES

mouse_controller = MouseController()


@CALLABLES.register("mouse.click")
def click(button, count):
    if button in ("left", "middle", "right"):
        button = getattr(Button, button)
    return mouse_controller.click(button, count)


CALLABLES.register("mouse.move")(mouse_controller.move)
CALLABLES.register("mouse.position")(lambda: mouse_controller.position)
CALLABLES.register("mouse.press")(mouse_controller.press)
CALLABLES.register("mouse.release")(mouse_controller.release)
CALLABLES.register("mouse.scroll")(mouse_controller.scroll)

keyboard_controller = KeyboardController()

CALLABLES.register("keyboard.press")(keyboard_controller.press)
CALLABLES.register("keyboard.release")(keyboard_controller.release)
CALLABLES.register("keyboard.type")(keyboard_controller.type)


@CALLABLES.register("keyboard.press_repeat")
def press_repeat_key(key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033):
    """Mocks the behavior of holding a key down, with a delay between presses."""
    repeat_time = max(0, (press_time - initial_delay) // repeat_delay - 1)

    keyboard_controller.press(key)
    time.sleep(initial_delay)
    for _ in range(int(repeat_time)):
        keyboard_controller.press(key)
        time.sleep(repeat_delay)
    keyboard_controller.release(key)
