import time

from owa.core import Listener


def time_ns() -> int:
    """
    Return the current time in nanoseconds since the Unix epoch.

    This function provides high-precision timing for OWA components,
    useful for performance measurement and precise scheduling.

    Returns:
        int: Current time in nanoseconds since Unix epoch (January 1, 1970)

    Examples:
        Get current timestamp:

        >>> current_time = time_ns()
        >>> print(f"Current time: {current_time}")

        Measure execution time:

        >>> start = time_ns()
        >>> # ... some operation ...
        >>> duration = time_ns() - start
        >>> print(f"Operation took {duration} nanoseconds")
    """
    return time.time_ns()


S_TO_NS = 1_000_000_000


class ClockTickListener(Listener):
    """
    A listener that triggers callbacks at regular intervals.

    This listener provides precise timing for periodic tasks in OWA,
    supporting configurable intervals and automatic callback execution.

    Examples:
        Basic usage with 1-second interval:

        >>> def on_tick():
        ...     print(f"Tick at {time_ns()}")
        >>>
        >>> listener = ClockTickListener()
        >>> listener.configure(callback=on_tick, interval=1)
        >>> listener.start()
        >>> # ... listener runs in background ...
        >>> listener.stop()
        >>> listener.join()

        Custom interval timing:

        >>> listener = ClockTickListener()
        >>> listener.configure(callback=my_callback, interval=0.5)  # 500ms
        >>> listener.start()
    """

    def on_configure(self, *, interval: float = 1):
        """
        Configure the tick interval for the listener.

        Args:
            interval (float): Time between ticks in seconds. Defaults to 1.0.

        Examples:
            Configure for 100ms intervals:

            >>> listener.configure(callback=my_func, interval=0.1)
        """
        self.interval = interval * S_TO_NS

    def loop(self, *, stop_event, callback):
        """
        Main loop that executes callbacks at configured intervals.

        Args:
            stop_event: Threading event to signal when to stop
            callback: Function to call at each tick

        Note:
            This method runs in a separate thread and maintains precise timing
            by accounting for callback execution time.
        """
        self._last_called = time.time()
        while not stop_event.is_set():
            callback()
            to_sleep = self.interval - (time.time() - self._last_called)
            if to_sleep > 0:
                stop_event.wait(to_sleep / S_TO_NS)
