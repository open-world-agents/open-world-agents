import threading
import time

from owa.node import Node
from owa.registry import CALLABLES, LISTENERS

CALLABLES.register("clock.time_ns")(time.time_ns)


# tick listener
@LISTENERS.register("clock/tick")
class ClockTickListener(Node):
    def on_configure(self, callback, *, interval=1):
        self.callback = callback
        self.interval = interval
        self._stop_event = threading.Event()
        return True

    def on_activate(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return True

    def _run(self):
        while not self._stop_event.is_set():
            self.callback()
            time.sleep(self.interval)

    def on_deactivate(self):
        self._stop_event.set()
        self._thread.join()
        del self._thread
        return True

    def on_shutdown(self):
        self._stop_event.set()
        self._thread.join()
        del self._thread
        return True
