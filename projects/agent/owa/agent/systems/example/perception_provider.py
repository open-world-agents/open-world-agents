from owa.agent.core import Clock, Event
from owa.agent.core.perception import PerceptionQueue, PerceptionSpecDict
from owa.core import Runnable

from .perception_spec import PERCEPTION_SPEC_DICT


class PerceptionProvider(Runnable):
    def on_configure(
        self, perception_queue: PerceptionQueue, clock: Clock, spec: PerceptionSpecDict = PERCEPTION_SPEC_DICT
    ):
        self._perception_queue = perception_queue
        self._clock = clock

    def loop(self, *, stop_event):
        # Acquire perceptions, put into perception_queue as needed
        # Placeholder for the actual perception logic
        while not stop_event.is_set():
            perception = Event(timestamp=self._clock.get_time_ns(), topic="print", msg="I saw a dragon")
            self._perception_queue["print"].put_nowait(perception)
            self._clock.sleep(1)
