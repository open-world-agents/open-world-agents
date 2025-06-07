import time

from owa.core import Listener
from owa.core.registry import CALLABLES, LISTENERS


# Create a named function for entry points compatibility
def time_ns_func():
    """Get current time in nanoseconds."""
    return time.time_ns()


# Register with unified naming only
CALLABLES.register("std/time_ns")(time_ns_func)

S_TO_NS = 1_000_000_000


# tick listener
@LISTENERS.register("std/tick")
class ClockTickListener(Listener):
    def on_configure(self, *, interval=1):
        self.interval = interval * S_TO_NS

    def loop(self, *, stop_event, callback):
        self._last_called = time.time()
        while not stop_event.is_set():
            callback()
            to_sleep = self.interval - (time.time() - self._last_called)
            if to_sleep > 0:
                stop_event.wait(to_sleep / S_TO_NS)
