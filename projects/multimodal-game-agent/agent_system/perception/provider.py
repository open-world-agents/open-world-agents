from owa.core import Runnable  # or your real Runnable location


class PerceptionProvider(Runnable):
    def on_configure(self, perception_queue, clock):
        self._perception_queue = perception_queue
        self._clock = clock

    def loop(self, *, stop_event):
        # Acquire perceptions, put into perception_queue as needed
        pass


class OWAMcapPerceptionReader:
    def __init__(self, file_path):
        # Read MCAP or similar initialization
        pass

    def sample(self, now):
        # Sample logic
        pass
