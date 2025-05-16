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

    def sample(self, now, *, spec: PerceptionSpecDict) -> Perception:
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
        for channel, strategy in spec.items():
            # Calculate absolute time window in nanoseconds
            start_time_ns = now + int(strategy.window_start * TimeUnits.SECOND)
            end_time_ns = now + int(strategy.window_end * TimeUnits.SECOND)

            perception[channel] = self._sample(strategy, start_time_ns, end_time_ns, now)

        return perception

    def _sample(self, spec: PerceptionSpec, start_time_ns: int, end_time_ns: int, now: float) -> List[Event]:
        """
        Sample continuous events according to the given strategy.

        Supports regular sampling, FPS-based sampling, and interpolation.
        """
        events = []

        # Get all events in the specified window
        msgs = list(
            self._reader.iter_decoded_messages(
                topics=[spec.topic],
                start_time=start_time_ns,
                end_time=end_time_ns,
                log_time_order=True,
                reverse=False,
            )
        )

        # Filter messages if a filter is specified
        if spec.msg_filter:
            msgs = [(topic, ts, msg) for topic, ts, msg in msgs if spec.msg_filter(msg)]

        # If there are no messages in the window, return empty list
        if not msgs:
            return events

        # Apply the specified sampling mode
        if spec.mode == "all":
            selected_msgs = msgs
        elif spec.mode == "first_k":
            selected_msgs = msgs[: spec.k]
        elif spec.mode == "last_k":
            selected_msgs = msgs[-spec.k :] if msgs else []
        else:
            # Default to all messages if mode isn't recognized
            selected_msgs = msgs

        # If we need to sample at a specific FPS, do additional processing
        if spec.fps > 0:
            # Calculate time between samples in nanoseconds
            interval_ns = int(1e9 / spec.fps)

            # Sample at regular intervals
            sampled_msgs = []
            target_times = range(start_time_ns, end_time_ns, interval_ns)

            if spec.interpolate:
                # Interpolate to get samples at exact times
                for target_time in target_times:
                    # Find messages before and after target time
                    before = None
                    after = None

                    for topic, ts, msg in selected_msgs:
                        if ts <= target_time and (before is None or ts > before[1]):
                            before = (topic, ts, msg)
                        if ts >= target_time and (after is None or ts < after[1]):
                            after = (topic, ts, msg)

                    # Interpolate the message with function given by the spec
                    sampled_msgs.append(spec.interpolation_fn(before, after))
            else:
                # Without interpolation, pick the closest message to each target time
                for target_time in target_times:
                    closest = None
                    min_diff = float("inf")

                    for topic, ts, msg in selected_msgs:
                        diff = abs(ts - target_time)
                        if diff < min_diff:
                            min_diff = diff
                            closest = (topic, ts, msg)

                    if closest:
                        sampled_msgs.append(closest)

            selected_msgs = sampled_msgs

        # Convert selected messages to Event objects
        for topic, timestamp_ns, msg in selected_msgs:
            events.append(
                Event(
                    timestamp=timestamp_ns,
                    topic=topic,
                    msg=msg,
                )
            )

        return events
