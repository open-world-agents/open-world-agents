import time

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from owa.msgs.desktop.keyboard import KeyboardState
from owa.msgs.desktop.mouse import MouseState

from ..utils import get_vk_state, vk_to_keycode

mouse_controller = MouseController()


def click(button: str | Button, count: int) -> None:
    """
    Simulate a mouse click.

    Args:
        button: Mouse button to click. Can be "left", "middle", "right" or a Button enum.
        count: Number of clicks to perform.

    Examples:
        >>> click("left", 1)  # Single left click
        >>> click("right", 2)  # Double right click
    """
    if button in ("left", "middle", "right"):
        button = getattr(Button, button)
    return mouse_controller.click(button, count)


def mouse_move(x: int, y: int) -> None:
    """
    Move the mouse cursor to specified coordinates.

    Args:
        x: X coordinate to move to.
        y: Y coordinate to move to.

    Examples:
        >>> mouse_move(100, 200)  # Move mouse to position (100, 200)
    """
    return mouse_controller.move(x, y)


def mouse_position() -> tuple[int, int]:
    """
    Get the current mouse cursor position.

    Returns:
        Tuple of (x, y) coordinates of the mouse cursor.

    Examples:
        >>> x, y = mouse_position()
        >>> print(f"Mouse is at ({x}, {y})")
    """
    return mouse_controller.position


def mouse_press(button: str | Button) -> None:
    """
    Press and hold a mouse button.

    Args:
        button: Mouse button to press. Can be "left", "middle", "right" or a Button enum.

    Examples:
        >>> mouse_press("left")  # Press and hold left mouse button
    """
    return mouse_controller.press(button)


def mouse_release(button: str | Button) -> None:
    """
    Release a previously pressed mouse button.

    Args:
        button: Mouse button to release. Can be "left", "middle", "right" or a Button enum.

    Examples:
        >>> mouse_release("left")  # Release left mouse button
    """
    return mouse_controller.release(button)


def mouse_scroll(x: int, y: int, dx: int, dy: int) -> None:
    """
    Simulate mouse wheel scrolling.

    Args:
        x: X coordinate where scrolling occurs.
        y: Y coordinate where scrolling occurs.
        dx: Horizontal scroll amount.
        dy: Vertical scroll amount.

    Examples:
        >>> mouse_scroll(100, 100, 0, 3)  # Scroll up 3 units at position (100, 100)
        >>> mouse_scroll(100, 100, 0, -3)  # Scroll down 3 units
    """
    return mouse_controller.scroll(x, y, dx, dy)


keyboard_controller = KeyboardController()


def press(key: str | int) -> None:
    """
    Press and hold a keyboard key.

    Args:
        key: Key to press. Can be a string (e.g., 'a', 'enter') or virtual key code.

    Examples:
        >>> press('a')  # Press and hold the 'a' key
        >>> press(65)  # Press and hold the 'a' key using virtual key code
    """
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.press(key)


def release(key: str | int) -> None:
    """
    Release a previously pressed keyboard key.

    Args:
        key: Key to release. Can be a string (e.g., 'a', 'enter') or virtual key code.

    Examples:
        >>> release('a')  # Release the 'a' key
        >>> release(65)  # Release the 'a' key using virtual key code
    """
    key = vk_to_keycode(key) if isinstance(key, int) else key
    return keyboard_controller.release(key)


def keyboard_type(text: str) -> None:
    """
    Type a string of characters.

    Args:
        text: Text string to type.

    Examples:
        >>> keyboard_type("Hello, World!")  # Types the text
        >>> keyboard_type("user@example.com")  # Types an email address
    """
    return keyboard_controller.type(text)


def press_repeat_key(
    key: str | int, press_time: float, initial_delay: float = 0.5, repeat_delay: float = 0.033
) -> None:
    """
    Simulate the behavior of holding a key down with auto-repeat.

    Args:
        key: Key to press repeatedly. Can be a string or virtual key code.
        press_time: Total time to hold the key down in seconds.
        initial_delay: Initial delay before auto-repeat starts (default: 0.5s).
        repeat_delay: Delay between repeated key presses (default: 0.033s).

    Examples:
        >>> press_repeat_key('a', 2.0)  # Hold 'a' key for 2 seconds with auto-repeat
        >>> press_repeat_key('space', 1.5, 0.3, 0.05)  # Custom timing for space key
    """
    key = vk_to_keycode(key) if isinstance(key, int) else key
    repeat_time = max(0, (press_time - initial_delay) // repeat_delay - 1)

    keyboard_controller.press(key)
    time.sleep(initial_delay)
    for _ in range(int(repeat_time)):
        keyboard_controller.press(key)
        time.sleep(repeat_delay)
    keyboard_controller.release(key)


def get_mouse_state() -> MouseState:
    """
    Get the current mouse state including position and pressed buttons.

    Returns:
        MouseState object containing current mouse position and pressed buttons.

    Examples:
        >>> state = get_mouse_state()
        >>> print(f"Mouse at ({state.x}, {state.y}), buttons: {state.buttons}")
    """
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
    """
    Get the current keyboard state including pressed keys.

    Returns:
        KeyboardState object containing currently pressed keys.

    Examples:
        >>> state = get_keyboard_state()
        >>> print(f"Pressed keys: {state.buttons}")
    """
    return KeyboardState(buttons=get_vk_state())


def release_all_keys() -> None:
    """
    Release all currently pressed keys on the keyboard.

    Examples:
        >>> release_all_keys()  # Release all pressed keys
    """
    keyboard_state: KeyboardState = get_keyboard_state()
    for key in keyboard_state.buttons:
        release(key)
