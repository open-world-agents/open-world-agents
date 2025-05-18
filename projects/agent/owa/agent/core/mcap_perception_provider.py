from typing import List

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits

from .event import Event
from .perception import Perception, PerceptionSpec, PerceptionSpecDict


class OWAMcapPerceptionReader:
    def __init__(self, file_path):
        # deserialize_to_objects=True is required to match output of online PerceptionProvider
        self._reader = OWAMcapReader(file_path, deserialize_to_objects=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._reader.__exit__()

    def sample(self, now: int, *, spec: PerceptionSpecDict) -> Perception:
        """
        Sample events from the MCAP file based on the provided specification.

        Args:
            now: Timestamp in nanoseconds that serves as the reference point for all relative times
            spec: Sampling specification defining which events to extract and how

        Returns:
            List of Event objects containing the sampled data
        """
        # Initialize result list
        perception = Perception()

        # Process all strategies
        for channel, perception_spec in spec.items():
            # Calculate absolute time window in nanoseconds
            start_time_ns = now + int(perception_spec.window_start * TimeUnits.SECOND)
            end_time_ns = now + int(perception_spec.window_end * TimeUnits.SECOND)

            perception[channel] = self._sample(perception_spec, start_time_ns, end_time_ns, now)

        return perception

    def _sample(self, spec: PerceptionSpec, start_time_ns: int, end_time_ns: int, now: int) -> List[Event]:
        """
        Sample continuous events according to the given strategy.

        Supports regular sampling, FPS-based sampling, and interpolation.
        """
        events = []

        # Get all events in the specified window
        msgs = list(
            self._reader.iter_decoded_messages(
                topics=spec.topics,
                start_time=start_time_ns,
                end_time=end_time_ns,
                log_time_order=True,
                reverse=False,
            )
        )

        for topic, timestamp_ns, msg in msgs:
            events.append(Event(timestamp=timestamp_ns, topic=topic, msg=msg))

        return events
