import time

from .clock import Clock, get_default_clock


class Rate:
    def __init__(self, rate: float, clock: Clock = None):
        self.rate = rate  # Rate in Hz
        self._interval = 1.0 / rate  # Interval in seconds
        self.clock = clock if clock is not None else get_default_clock()
        self._last_time = self.clock.get_time()

    def sleep(self):
        """
        Sleeps for the duration of the rate interval.
        """
        # Calculate the time to sleep based on the rate interval
        elapsed = self.clock.get_time() - self._last_time
        to_sleep = max(0, self._interval - elapsed)

        time.sleep(to_sleep)

        # Update the last time to the current time
        self._last_time = self.clock.get_time()
