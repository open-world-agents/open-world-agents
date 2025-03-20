import time

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from owa.core.registry import CALLABLES

from ..msg import KeyboardState, MouseState
from ..utils import get_vk_state, vk_to_keycode

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


@CALLABLES.register("keyboard.press")
def press(key):
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.press(key)


@CALLABLES.register("keyboard.release")
def release(key):
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.release(key)


CALLABLES.register("keyboard.type")(keyboard_controller.type)


@CALLABLES.register("keyboard.press_repeat")
def press_repeat_key(key, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033):
    """Mocks the behavior of holding a key down, with a delay between presses."""
    key = vk_to_keycode(key) if isinstance(key, int) else key
    repeat_time = max(0, (press_time - initial_delay) // repeat_delay - 1)

    keyboard_controller.press(key)
    time.sleep(initial_delay)
    for _ in range(int(repeat_time)):
        keyboard_controller.press(key)
        time.sleep(repeat_delay)
    keyboard_controller.release(key)


@CALLABLES.register("mouse.get_state")
def get_mouse_state():
    x, y = mouse_controller.position
    mouse_buttons = set()
    buttons = get_vk_state()
    for button, vk in {"left": 1, "right": 2, "middle": 4}.items():
        if vk in buttons:
            mouse_buttons.add(button)
    return MouseState(x=x, y=y, buttons=mouse_buttons)


@CALLABLES.register("keyboard.get_state")
def get_keyboard_state():
    return KeyboardState(buttons=get_vk_state())
