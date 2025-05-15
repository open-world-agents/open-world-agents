import queue

from owa.agent.core import Clock, Event
from owa.agent.core.spec import PerceptionSamplingSpec
from owa.core import Runnable


class PerceptionProvider(Runnable):
    def on_configure(self, perception_queue: queue.Queue[Event], spec: PerceptionSamplingSpec, clock: Clock):
        self._perception_queue = perception_queue
        self._clock = clock

    def loop(self, *, stop_event):
        # Acquire perceptions, put into perception_queue as needed
        # Placeholder for the actual perception logic
        while not stop_event.is_set():
            perception = Event(timestamp=self._clock.get_time_ns(), topic="print", msg="I saw a dragon")
            self._perception_queue.put_nowait(perception)
            self._clock.sleep(1)
