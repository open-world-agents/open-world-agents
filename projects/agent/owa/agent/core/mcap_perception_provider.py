import math
import time
from typing import Any, Dict, List, Optional, Tuple

from mcap_owa.highlevel import OWAMcapReader

from .event import Event
from .spec import ContinuousSamplingStrategy, DiscreteSamplingStrategy, EventType, PerceptionSamplingSpec


class OWAMcapPerceptionReader:
    def __init__(self, file_path):
        # deserialize_to_objects=True is required to match output of online PerceptionProvider
        self._reader = OWAMcapReader(file_path, deserialize_to_objects=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._reader.__exit__()

    def sample(self, now, *, spec: PerceptionSamplingSpec) -> list[Event]:
        """
        Sample events from the MCAP file based on the provided specification.

        Args:
            now: Timestamp in nanoseconds that serves as the reference point for all relative times
            spec: Sampling specification defining which events to extract and how

        Returns:
            List of Event objects containing the sampled data
        """
        # Initialize result list
        events = []

        # Process all strategies (both inputs and outputs)
        for strategy_list in [spec.inputs, spec.outputs]:
            for strategy in strategy_list:
                # Calculate absolute time window in nanoseconds
                start_time_ns = now + int(strategy.window_start * 1e9)
                end_time_ns = now + int(strategy.window_end * 1e9)

                # Handle discrete events
                if strategy.event_type == EventType.DISCRETE:
                    events.extend(self._sample_discrete(strategy, start_time_ns, end_time_ns, now))

                # Handle continuous events
                elif strategy.event_type == EventType.CONTINUOUS:
                    events.extend(self._sample_continuous(strategy, start_time_ns, end_time_ns, now))

        return events

    def _sample_discrete(
        self, strategy: DiscreteSamplingStrategy, start_time_ns: int, end_time_ns: int, now: int
    ) -> List[Event]:
        """
        Sample discrete events according to the given strategy.

        Handles regular event sampling and also state bootstrapping for stateful inputs.
        """
        events = []

        # Check if we need to fetch a prior state
        if strategy.include_prior_state and strategy.state_topic:
            # Get the most recent state before the window starts
            state_topic = strategy.state_topic
            state_msgs = list(
                self._reader.iter_decoded_messages(
                    topics=[state_topic],
                    end_time=start_time_ns,  # Before window starts
                    log_time_order=True,
                    reverse=True,  # Most recent first
                )
            )

            # If we found a state message, add it as an event
            if state_msgs:
                topic, timestamp_ns, msg = state_msgs[0]  # Take only the most recent
                events.append(
                    Event(
                        timestamp=timestamp_ns,
                        topic=topic,
                        msg=msg,
                    )
                )

        # Get all events in the specified window
        msgs = list(
            self._reader.iter_decoded_messages(
                topics=[strategy.topic],
                start_time=start_time_ns,
                end_time=end_time_ns,
                log_time_order=True,
                reverse=False,
            )
        )

        # Filter messages if a filter is specified
        if strategy.msg_filter:
            msgs = [(topic, ts, msg) for topic, ts, msg in msgs if strategy.msg_filter(msg)]

        # Apply the specified sampling mode
        if strategy.mode == "all":
            selected_msgs = msgs
        elif strategy.mode == "first_k":
            selected_msgs = msgs[: strategy.k]
        elif strategy.mode == "last_k":
            selected_msgs = msgs[-strategy.k :] if msgs else []

        # Convert selected messages to Event objects
        for topic, timestamp_ns, msg in selected_msgs:
            # If we have a state update function, apply it
            if strategy.state_update_fn and events and events[-1].topic == strategy.state_topic:
                events[-1].msg = strategy.state_update_fn(events[-1].msg, msg)

            # Add the event
            events.append(
                Event(
                    timestamp=timestamp_ns,
                    topic=topic,
                    msg=msg,
                )
            )

        return events

    def _sample_continuous(
        self, strategy: ContinuousSamplingStrategy, start_time_ns: int, end_time_ns: int, now: float
    ) -> List[Event]:
        """
        Sample continuous events according to the given strategy.

        Supports regular sampling, FPS-based sampling, and interpolation.
        """
        events = []

        # Get all events in the specified window
        msgs = list(
            self._reader.iter_decoded_messages(
                topics=[strategy.topic],
                start_time=start_time_ns,
                end_time=end_time_ns,
                log_time_order=True,
                reverse=False,
            )
        )

        # Filter messages if a filter is specified
        if strategy.msg_filter:
            msgs = [(topic, ts, msg) for topic, ts, msg in msgs if strategy.msg_filter(msg)]

        # If there are no messages in the window, return empty list
        if not msgs:
            return events

        # Apply the specified sampling mode
        if strategy.mode == "all":
            selected_msgs = msgs
        elif strategy.mode == "first_k":
            selected_msgs = msgs[: strategy.k]
        elif strategy.mode == "last_k":
            selected_msgs = msgs[-strategy.k :] if msgs else []
        else:
            # Default to all messages if mode isn't recognized
            selected_msgs = msgs

        # If we need to sample at a specific FPS, do additional processing
        if strategy.fps > 0:
            # Calculate time between samples in nanoseconds
            interval_ns = int(1e9 / strategy.fps)

            # Sample at regular intervals
            sampled_msgs = []
            target_times = range(start_time_ns, end_time_ns, interval_ns)

            if strategy.interpolate:
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

                    # If we have both before and after, we can interpolate
                    # For now, just use the closest one (actual interpolation would depend on the message type)
                    if before and after:
                        if target_time - before[1] < after[1] - target_time:
                            sampled_msgs.append(before)
                        else:
                            sampled_msgs.append(after)
                    # If we only have one, use that
                    elif before:
                        sampled_msgs.append(before)
                    elif after:
                        sampled_msgs.append(after)
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
