import time

from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Listener as MouseListener

from owa.core.listener import Listener
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent

from ..utils import key_to_vk
from .callables import get_keyboard_state, get_mouse_state


class KeyboardListenerWrapper(Listener):
    """
    Keyboard event listener that captures key press and release events.

    This listener wraps pynput's KeyboardListener to provide keyboard event
    monitoring with OWA's listener interface.

    Examples:
        >>> def on_key_event(event):
        ...     print(f"Key {event.vk} was {event.event_type}")
        >>> listener = KeyboardListenerWrapper().configure(callback=on_key_event)
        >>> listener.start()
    """

    def on_configure(self):
        self.listener = KeyboardListener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        vk = key_to_vk(key)
        self.callback(KeyboardEvent(event_type="press", vk=vk))

    def on_release(self, key):
        vk = key_to_vk(key)
        self.callback(KeyboardEvent(event_type="release", vk=vk))

    def loop(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()


class MouseListenerWrapper(Listener):
    """
    Mouse event listener that captures mouse movement, clicks, and scroll events.

    This listener wraps pynput's MouseListener to provide mouse event
    monitoring with OWA's listener interface.

    Examples:
        >>> def on_mouse_event(event):
        ...     print(f"Mouse {event.event_type} at ({event.x}, {event.y})")
        >>> listener = MouseListenerWrapper().configure(callback=on_mouse_event)
        >>> listener.start()
    """

    def on_configure(self):
        self.listener = MouseListener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)

    def on_move(self, x, y):
        self.callback(MouseEvent(event_type="move", x=x, y=y))

    def on_click(self, x, y, button: Button, pressed):
        self.callback(MouseEvent(event_type="click", x=x, y=y, button=button.name, pressed=pressed))

    def on_scroll(self, x, y, dx, dy):
        self.callback(MouseEvent(event_type="scroll", x=x, y=y, dx=dx, dy=dy))

    def loop(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()


class KeyboardStateListener(Listener):
    """
    Periodically reports the current keyboard state.

    This listener calls the callback function every second with the current
    keyboard state, including which keys are currently pressed.

    Examples:
        >>> def on_keyboard_state(state):
        ...     if state.buttons:
        ...         print(f"Keys pressed: {state.buttons}")
        >>> listener = KeyboardStateListener().configure(callback=on_keyboard_state)
        >>> listener.start()
    """

    def loop(self, stop_event):
        while not stop_event.is_set():
            state = get_keyboard_state()
            self.callback(state)
            time.sleep(1)


class MouseStateListener(Listener):
    """
    Periodically reports the current mouse state.

    This listener calls the callback function every second with the current
    mouse state, including position and pressed buttons.

    Examples:
        >>> def on_mouse_state(state):
        ...     print(f"Mouse at ({state.x}, {state.y}), buttons: {state.buttons}")
        >>> listener = MouseStateListener().configure(callback=on_mouse_state)
        >>> listener.start()
    """

    def loop(self, stop_event):
        while not stop_event.is_set():
            state = get_mouse_state()
            self.callback(state)
            time.sleep(1)
