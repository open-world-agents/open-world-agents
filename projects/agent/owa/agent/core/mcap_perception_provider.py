from mcap_owa.highlevel import OWAMcapReader

from .event import Event
from .spec import PerceptionSamplingSpec


class OWAMcapPerceptionReader:
    def __init__(self, file_path):
        self._reader = OWAMcapReader(file_path)

    def __exit__(self, exc_type, exc_value, traceback):
        self._reader.__exit__()

    def sample(self, now, *, spec: PerceptionSamplingSpec) -> list[Event]:
        # Placeholder for the actual sampling logic
        events = [Event(timestamp=now, data="sampled_data")]
        return events
