import time

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from ..msg import KeyboardState, MouseState
from ..utils import get_vk_state, vk_to_keycode

mouse_controller = MouseController()


def click(button, count):
    if button in ("left", "middle", "right"):
        button = getattr(Button, button)
    return mouse_controller.click(button, count)


def mouse_move(x, y):
    return mouse_controller.move(x, y)


def mouse_position():
    return mouse_controller.position


def mouse_press(button):
    return mouse_controller.press(button)


def mouse_release(button):
    return mouse_controller.release(button)


def mouse_scroll(x, y, dx, dy):
    return mouse_controller.scroll(x, y, dx, dy)


keyboard_controller = KeyboardController()


def press(key):
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.press(key)


def release(key):
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.release(key)


def keyboard_type(text):
    return keyboard_controller.type(text)


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


def get_mouse_state() -> MouseState:
    """Get the current mouse state including position and pressed buttons."""
    position = mouse_controller.position
    if position is None:
        position = (-1, -1)  # Fallback if position cannot be retrieved
    mouse_buttons = set()
    buttons = get_vk_state()
    for button, vk in {"left": 1, "right": 2, "middle": 4}.items():
        if vk in buttons:
            mouse_buttons.add(button)
    return MouseState(x=position[0], y=position[1], buttons=mouse_buttons)


def get_keyboard_state() -> KeyboardState:
    """Get the current keyboard state including pressed keys."""
    return KeyboardState(buttons=get_vk_state())


def release_all_keys():
    """Release all currently pressed keys on the keyboard."""
    keyboard_state: KeyboardState = get_keyboard_state()
    for key in keyboard_state.buttons:
        release(key)
