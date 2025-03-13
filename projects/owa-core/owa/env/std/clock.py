import time

from owa.core import Listener
from owa.core.registry import CALLABLES, LISTENERS

CALLABLES.register("clock.time_ns")(time.time_ns)

S_TO_NS = 1_000_000_000


# tick listener
@LISTENERS.register("clock/tick")
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
