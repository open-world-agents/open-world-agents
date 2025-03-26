"""
This module provides classes for handling environment state acquisition and action execution.

Two main concepts are introduced:
1. Callable - Defines functions that acquire state or perform actions, called by the user.
2. Listener - Provides interfaces for the environment to call user-defined functions.

The key difference between these two is who initiates the call:
- Callable is actively called by the user
- Listener is passively waiting for events and then calls user-provided callbacks
"""

import inspect
import threading
from abc import abstractmethod
from multiprocessing.synchronize import Event as mpEvent
from typing import Self

from .callable import Callable
from .runnable import RunnableMixin, RunnableProcess, RunnableThread


class ListenerMixin(RunnableMixin):
    """
    A mixin class that provides callback registration and invocation functionality.

    ListenerMixin extends RunnableMixin to add the ability to register a callback
    function that will be called when the listener detects an event. This provides
    a standard interface for environment-triggered calls to user-defined functions.

    Example:
        ```python
        # Define a callback function to be called when events are detected
        def on_event(data):
            print(f"Event detected: {data}")

        # Create a custom listener by inheriting from Listener
        class KeyboardListener(Listener):
            def loop(self, stop_event, callback):
                print("Listening for keyboard events...")
                while not stop_event.is_set():
                    # Simulate detecting an event
                    key_pressed = input("Press a key: ")
                    if key_pressed:
                        # Call the callback with the detected event data
                        callback({"key": key_pressed, "timestamp": time.time()})

                    # Check stop_event periodically
                    if stop_event.is_set():
                        break

        # Create and configure the listener with our callback
        listener = KeyboardListener().configure(callback=on_event)

        # Use the listener in a context manager
        with listener.session as active_listener:
            # The listener is now running in a separate thread
            # and will call on_event when keys are pressed
            time.sleep(30)  # Keep listening for 30 seconds

        # After exiting the context, the listener is stopped and joined
        ```
    """

    def get_callback(self) -> Callable:
        """
        Get the registered callback function.

        Returns:
            Callable: The registered callback function.

        Raises:
            AttributeError: If no callback has been registered.
        """
        if not hasattr(self, "_callback"):
            raise AttributeError("Callback not set. Please call self.register_callback() to set the callback.")
        return self._callback

    def register_callback(self, callback: Callable) -> Self:
        """
        Register a callback function to be called when an event is detected.

        Args:
            callback (Callable): The function to call when an event is detected.

        Returns:
            Self: The listener instance for method chaining.
        """
        self._callback = callback
        return self

    # Property to access and set the callback using attribute notation
    callback = property(get_callback, register_callback)

    def configure(self, *args, callback: Callable, **kwargs) -> Self:
        """
        Configure the listener with a callback function and other parameters.

        Args:
            callback (Callable): The function to call when an event is detected.
                                This is a required keyword argument.
            *args: Positional arguments to pass to the parent's configure method.
            **kwargs: Keyword arguments to pass to the parent's configure method.

        Returns:
            Self: The configured listener instance.
        """
        self.register_callback(callback)
        super().configure(*args, **kwargs)
        return self

    @abstractmethod
    def loop(self):
        """
        Main execution loop. Must be implemented by subclasses.

        This method contains the main logic that runs while the runnable is active.

        Requirements:
            - It MUST respect the self.stop to allow clean termination.
            - It MUST respect the self.callback to call the registered function.
        """


class ListenerThread(ListenerMixin, RunnableThread):
    """
    A thread-based implementation of the ListenerMixin.

    This class runs the listener loop in a separate thread, making it suitable for
    I/O-bound event listening that shouldn't block the main program execution.
    """

    def run(self):
        """
        Thread execution method. Do not call this directly; use start() instead.

        This method ensures the runnable is configured before executing the loop.
        """
        if not getattr(self, "_configured", False):
            raise RuntimeError(
                "RunnableThread is not configured. Call configure() before start(). Or you may have overriden the configure method, not on_configure."
            )

        kwargs = {}
        if "stop_event" in inspect.signature(self.loop).parameters:
            kwargs["stop_event"] = self._stop_event
        if "callback" in inspect.signature(self.loop).parameters:
            kwargs["callback"] = self.callback
        self.loop(**kwargs)

    @abstractmethod
    def loop(self, stop_event: threading.Event, callback: Callable):
        """
        Main thread execution loop. Must be implemented by subclasses.

        Args:
            stop_event (threading.Event): An event that will be set when the thread should stop.
                                        Check this event regularly and exit when it's set.
                                        If this argument is not present, the loop will be called without it.
            callback (Callable): The function to call when an event is detected.
                                If this argument is not present, the loop will be called without it.
        """
        pass


class ListenerProcess(ListenerMixin, RunnableProcess):
    """
    A process-based implementation of the ListenerMixin.

    This class runs the listener loop in a separate process, making it suitable for
    CPU-bound event processing or when isolation from the main process is desired.
    """

    def run(self):
        """
        Process execution method. Do not call this directly; use start() instead.

        This method ensures the runnable is configured before executing the loop.
        """
        if not getattr(self, "_configured", False):
            raise RuntimeError(
                "RunnableProcess is not configured. Call configure() before start(). Or you may have overriden the configure method, not on_configure."
            )

        kwargs = {}
        if "stop_event" in inspect.signature(self.loop).parameters:
            kwargs["stop_event"] = self._stop_event
        self.loop(**kwargs)

    @abstractmethod
    def loop(self, stop_event: mpEvent, callback: Callable):
        """
        Main process execution loop. Must be implemented by subclasses.

        Args:
            stop_event (multiprocessing.Event): An event that will be set when the process should stop.
                                        Check this event regularly and exit when it's set.
                                        If this argument is not present, the loop will be called without it.
            callback (Callable): The function to call when an event is detected.
                                If this argument is not present, the loop will be called without it.
        """
        pass


# Default implementation is thread-based for better compatibility and easier use
Listener = ListenerThread

# TODO: Implement synchronous event listening design, similar to:
# https://pynput.readthedocs.io/en/latest/keyboard.html#synchronous-event-listening-for-the-keyboard-listener
