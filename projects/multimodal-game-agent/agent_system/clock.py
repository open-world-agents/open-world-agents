import time


class Clock:
    def __init__(self, start_time=time.time_ns()):
        # Initialize the clock with the current time and paused state
        self._start_time = time.time_ns()
        self._bias = start_time
        self._paused_time = 0
        self._is_paused = False
        self._pause_start = None  # Tracks when pause started

    def get_time(self):
        """
        Returns the elapsed time in nanoseconds since the clock started,
        excluding any time spent in the paused state.
        """
        if self._is_paused:
            # If paused, return the time when the clock was paused
            return self._pause_start + self._bias
        else:
            # Calculate elapsed time excluding paused duration
            return (time.time_ns() - self._start_time) - self._paused_time + self._bias

    def pause(self):
        """
        Pauses the clock. If already paused, this has no effect.
        """
        if not self._is_paused:
            self._is_paused = True
            self._pause_start = time.time_ns() - self._start_time  # Record when pause started

    def resume(self):
        """
        Resumes the clock if it is paused. If not paused, this has no effect.
        """
        if self._is_paused and self._pause_start is not None:
            self._is_paused = False
            # Add the duration of the pause to the paused time
            self._paused_time += (time.time_ns() - self._start_time) - self._pause_start
            self._pause_start = None  # Reset pause start


if __name__ == "__main__":
    clock = Clock(start_time=10**9)
    print("Clock started")
    time.sleep(1)  # Simulate some elapsed time
    print("Elapsed time (ns):", clock.get_time())
    clock.pause()
    print("Clock paused")
    time.sleep(1)  # Simulate some elapsed time while paused
    print("Elapsed time (ns):", clock.get_time())
    clock.resume()
    print("Clock resumed")
    time.sleep(1)  # Simulate some elapsed time after resuming
    print("Elapsed time (ns):", clock.get_time())
