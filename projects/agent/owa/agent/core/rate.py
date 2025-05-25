from typing import Optional

from .clock import Clock, get_default_clock


class Rate:
    """
    Utility class to maintain a fixed execution rate (Hz).
    """

    def __init__(self, rate: float, clock: Optional[Clock] = None):
        """
        Args:
            rate (float): Frequency in Hz. Must be > 0.
            clock (Clock, optional): Clock instance to use. Defaults to system clock.
        Raises:
            ValueError: If rate is not positive.
        """
        if rate <= 0:
            raise ValueError("Rate must be positive and non-zero.")
        self.rate = rate
        self._interval = 1.0 / rate
        self.clock = clock if clock is not None else get_default_clock()
        self.reset()

    def sleep(self) -> None:
        """
        Sleeps for the duration required to maintain the set rate.
        """
        now = self.clock.get_time()
        elapsed = now - self._last_time
        to_sleep = max(0.0, self._interval - elapsed)
        self.clock.sleep(to_sleep)
        self._last_time = self.clock.get_time()

    def reset(self) -> None:
        """
        Resets the timer to the current time.
        """
        self._last_time = self.clock.get_time()

    def __repr__(self) -> str:
        return f"<Rate(rate={self.rate}Hz, interval={self._interval:.4f}s, last_time={self._last_time:.4f})>"
