import time


class Clock:
    def __init__(self, start_time=0, scale=1.0):
        # Initialize the clock with the current time and paused state
        self._start_time = time.time_ns()
        self._start_offset = start_time  # Initial time offset for the simulation
        self._paused_time = 0
        self._is_paused = False
        self._pause_start = None  # Tracks when pause started
        self._scale = scale  # Time scaling factor (1.0 = normal speed, 2.0 = 2x speed, 0.5 = half speed)

    def get_time(self, unit="s") -> int | float:
        """
        Returns the elapsed time since the clock started, excluding any time spent in the paused state.
        """
        time_ns = self.get_time_ns()
        if unit == "ns":
            return time_ns
        elif unit == "s":
            return time_ns / 1_000_000_000

    def get_time_ns(self) -> int:
        """
        Returns the elapsed time in nanoseconds since the clock started,
        excluding any time spent in the paused state.
        """
        if self._is_paused:
            # If paused, return the time when it was paused
            scaled_time = self._pause_start
        else:
            # Calculate elapsed time excluding paused duration
            elapsed_time = time.time_ns() - self._start_time
            scaled_time = elapsed_time * self._scale - self._paused_time + self._start_offset

        return int(scaled_time)

    def pause(self):
        """
        Pauses the clock. If already paused, this has no effect.
        """
        if not self._is_paused:
            # Record the time when the clock was paused
            self._pause_start = self.get_time_ns()
            # Set the paused state. This must be done after recording the pause start
            self._is_paused = True

    def resume(self):
        """
        Resumes the clock if it is paused. If not paused, this has no effect.
        """
        if self._is_paused and self._pause_start is not None:
            self._is_paused = False
            # Add the duration of the pause to the paused time
            self._paused_time += self.get_time_ns() - self._pause_start
            self._pause_start = None  # Reset pause start

    def sleep(self, duration: float):
        """
        Sleeps until the specified duration (in seconds) has passed according to the clock's own time,
        i.e., simulation time, not wall-clock time. If paused, waits until resumed.
        """
        start = self.get_time()
        while True:
            if not self._is_paused:
                now = self.get_time()
                if now - start >= duration:
                    break
            # TODO: more efficient "sleep_until" implementation, without busy waiting
            time.sleep(0.01)


def get_default_clock():
    return Clock()


if __name__ == "__main__":
    clock = Clock(start_time=10**9, scale=2.0)  # Start at 1 second, 2x speed
    print("Clock started with 2x speed")
    time.sleep(1)  # Simulate some elapsed time
    print("Elapsed time (s):", clock.get_time())
    clock.pause()
    print("Clock paused")
    time.sleep(1)  # Simulate some elapsed time while paused
    print("Elapsed time (s):", clock.get_time())
    clock.resume()
    print("Clock resumed")
    time.sleep(1)  # Simulate some elapsed time after resuming
    print("Elapsed time (s):", clock.get_time())
