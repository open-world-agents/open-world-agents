import threading
import time

from owa import Listener
from owa.registry import CALLABLES, LISTENERS

CALLABLES.register("clock.time_ns")(time.time_ns)


# tick listener
@LISTENERS.register("clock/tick")
class ClockTickListener(Listener):
    def configure(self, *, interval=1):
        self.interval = interval

    def loop(self):
        self._last_called = time.time()
        while not self._stop_event.is_set():
            self.callback()
            to_sleep = self.interval - (time.time() - self._last_called)
            if to_sleep > 0:
                self._stop_event.wait(to_sleep)

    def cleanup(self): ...  # nothing to clean up
