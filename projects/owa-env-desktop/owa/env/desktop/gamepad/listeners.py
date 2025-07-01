import time
import threading

from owa.core.listener import Listener
from .callables import get_gamepad_events, get_gamepad_state


class GamepadListenerWrapper(Listener):
    """
    Gamepad event listener that captures button press, button release, and axis motion events.

    This listener uses the gamepad callables to provide gamepad event monitoring with OWA's
    listener interface. It captures events from all connected gamepads and converts them
    to GamepadEvent messages.

    Examples:
        >>> def on_gamepad_event(event):
        ...     print(f"Gamepad {event.event_type}: {event.button or event.axis}")
        >>> listener = GamepadListenerWrapper().configure(callback=on_gamepad_event)
        >>> listener.start()

    Note:
        Requires the 'inputs' library to be installed. If not available, the listener
        will raise an ImportError when used.
    """

    def loop(self, stop_event: threading.Event):
        """
        Main gamepad event listening loop.

        Args:
            stop_event: Threading event to signal when to stop listening
        """
        while not stop_event.is_set():
            try:
                # Get events from all gamepads using the callable
                events = get_gamepad_events()

                for event in events:
                    if stop_event.is_set():
                        break
                    self.callback(event)

                # Small sleep to prevent excessive CPU usage when no events
                if not events:
                    time.sleep(0.01)

            except ImportError:
                # inputs library not available
                raise
            except Exception:
                # Handle other exceptions gracefully
                if not stop_event.is_set():
                    time.sleep(0.1)


class GamepadStateListener(Listener):
    """
    Periodically reports the current state of all connected gamepads.

    This listener calls the callback function every second with the current
    gamepad state, including which buttons are pressed and current axis values.

    Examples:
        >>> def on_gamepad_state(state):
        ...     if state.buttons:
        ...         print(f"Buttons pressed: {state.buttons}")
        ...     print(f"Axes: {state.axes}")
        >>> listener = GamepadStateListener().configure(callback=on_gamepad_state)
        >>> listener.start()

    Note:
        Requires the 'inputs' library to be installed. If not available, the listener
        will raise an ImportError when configured.
    """

    def loop(self, stop_event: threading.Event):
        """
        Main gamepad state monitoring loop.

        Args:
            stop_event: Threading event to signal when to stop listening
        """
        while not stop_event.is_set():
            # Get state for the first gamepad (index 0)
            # In the future, this could be extended to report all connected gamepads
            state = get_gamepad_state(0)
            if state:  # Only call callback if gamepad is connected
                self.callback(state)

            # Wait for 1 second or until stop is requested
            stop_event.wait(1.0)