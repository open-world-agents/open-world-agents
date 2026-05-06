import time

from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Listener as MouseListener

from owa.core.listener import Listener
from owa.msgs.desktop.keyboard import KeyboardEvent
from owa.msgs.desktop.mouse import MouseEvent, RawMouseEvent

from ..utils import key_to_vk
from .callables import get_keyboard_state, get_mouse_state
from .raw_input import RawInputCapture


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
        # this line must be present to ensure is_alive is True while the listener is running
        self.listener.join()

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
        # this line must be present to ensure is_alive is True while the listener is running
        self.listener.join()

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


class RawMouseListener(Listener):
    """
    Raw mouse input listener using Windows WM_INPUT messages.

    This listener captures high-definition mouse movement data directly from the HID stack,
    bypassing Windows pointer acceleration and screen resolution limits. Provides sub-pixel
    precision and unfiltered input data essential for gaming and precision applications.

    Examples:
        >>> def on_raw_mouse_event(event):
        ...     print(f"Raw mouse: dx={event.dx}, dy={event.dy}, flags={event.button_flags}")
        >>> listener = RawMouseListener().configure(callback=on_raw_mouse_event)
        >>> listener.start()
    """

    def on_configure(self):
        """Initialize the raw input capture system."""
        self.raw_input_capture = RawInputCapture()
        self.raw_input_capture.register_callback(self._on_raw_mouse_event)

    def _on_raw_mouse_event(self, event: RawMouseEvent) -> None:
        """Internal callback to forward raw mouse events to the registered callback."""
        if hasattr(self, "_current_callback") and self._current_callback:
            self._current_callback(event)

    def loop(self, stop_event, callback):
        """Start the raw input capture loop."""
        # Store the callback for use in the raw input callback
        self._current_callback = callback

        if not self.raw_input_capture.start():
            raise RuntimeError("Failed to start raw input capture")

        # Keep the loop running while the capture is active
        try:
            # The Windows message loop in raw_input_capture handles events efficiently
            # We just need to wait for the stop event without artificial delays
            stop_event.wait()
        finally:
            self.raw_input_capture.stop()
            self._current_callback = None
