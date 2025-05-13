from owa.core import Runnable


class PerceptionProvider(Runnable):
    def on_configure(self, perception_queue, clock):
        self._perception_queue = perception_queue
        self._clock = clock

    def loop(self, *, stop_event):
        # Acquire perceptions, put into perception_queue as needed
        pass
