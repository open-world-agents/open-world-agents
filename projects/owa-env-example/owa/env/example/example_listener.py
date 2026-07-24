"""
Example listener components for the example environment plugin.

This module demonstrates how to create listener components that can
respond to events and call user-provided callbacks.

Components are discovered via entry points and loaded lazily.
No decorators needed - registration happens via plugin_spec.
"""

import threading
from typing import Callable

from loguru import logger

from owa.core import Listener


class ExampleListener(Listener):
    """
    Example listener that periodically calls a callback with a message.

    This listener demonstrates the basic pattern for creating event-driven
    components that can notify user code when something happens.
    """

    def on_configure(self, *, interval: float = 1.0, message: str = "Example event"):
        """
        Configure the listener.

        Args:
            interval: Time between events in seconds
            message: Message to send with each event
        """
        self.interval = interval
        self.message = message

    def loop(self, *, stop_event: threading.Event, callback: Callable[[str], None]):
        """
        Main listener loop that generates periodic events.

        Args:
            stop_event(threading.Event): Event to signal when to stop
            callback: Function to call with event data
        """
        counter = 0
        while not stop_event.is_set():
            # Create event data
            event_data = f"{self.message} #{counter}"

            # Call the user's callback
            try:
                callback(event_data)
            except Exception as e:
                # Log error but continue running
                logger.error(f"Error in callback: {e}")

            counter += 1

            # Wait for the specified interval or until stop is requested
            stop_event.wait(self.interval)


class ExampleTimerListener(Listener):
    """
    Example timer listener that calls a callback after a specified delay.

    This demonstrates a one-shot listener that stops after triggering once.
    """

    def on_configure(self, *, delay: float = 5.0, callback: Callable[[], None] = None):
        """
        Configure the timer.

        Args:
            delay: Delay in seconds before triggering
            callback: Function to call when timer expires (optional, can be provided in loop)
        """
        self.delay = delay
        self.callback = callback

    def loop(self, *, stop_event: threading.Event, callback: Callable[[], None]):
        """
        Timer loop that waits and then triggers once.

        Args:
            stop_event(threading.Event): Event to signal when to stop
            callback: Function to call when timer expires
        """
        # Wait for the delay or until stop is requested
        if not stop_event.wait(self.delay):
            # Timer expired (not stopped), call the callback
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in timer callback: {e}")
