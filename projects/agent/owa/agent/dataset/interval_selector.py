from abc import ABC, abstractmethod
from pathlib import Path

from mcap_owa.highlevel import OWAMcapReader
from owa.agent.dataset.interval import Intervals
from owa.core.time import TimeUnits
from owa.env.desktop.constants import VK


class IntervalExtractor(ABC):
    """Base class for different interval extraction strategies."""

    @abstractmethod
    def extract_intervals(self, file_path: Path) -> Intervals:
        """Extract valid time intervals from a recording file."""
        pass

    def filter_by_duration(self, intervals: Intervals, min_duration: int) -> Intervals:
        """Filter intervals based on minimum duration."""
        result = Intervals()
        for interval in intervals:
            if interval.length > min_duration:
                result.add((interval.start, interval.end))
        return result


class KeyPressIntervalExtractor(IntervalExtractor):
    """Extract intervals based on explicit start/stop key presses."""

    def __init__(self, start_stop_key: int = VK.F9, pause_key: int = VK.F10):
        self.start_stop_key = start_stop_key
        self.pause_key = pause_key

    def extract_intervals(self, file_path: Path) -> Intervals:
        timestamps = []

        with OWAMcapReader(file_path) as reader:
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard"]
            ):
                if msg.event_type == "release" and msg.vk == self.start_stop_key:
                    timestamps.append(timestamp)
                elif msg.vk == self.pause_key:
                    raise NotImplementedError("Pause key is not implemented")

        # Create pairs of timestamps (start, end)
        interval_pairs = list(zip(timestamps[::2], timestamps[1::2]))

        # Convert to Intervals object
        return Intervals(interval_pairs)


class InactivityBasedIntervalExtractor(IntervalExtractor):
    """Extract intervals by detecting periods of activity vs inactivity."""

    def __init__(self, inactivity_threshold: float = 5.0):
        """
        Args:
            inactivity_threshold: Time in seconds to consider as inactivity gap
        """
        self.inactivity_threshold = inactivity_threshold

    def extract_intervals(self, file_path: Path) -> Intervals:
        """
        Extract intervals of continuous activity from a recording file.

        This implementation identifies periods of activity by looking for gaps in
        input events that exceed the inactivity threshold.
        """
        activity_intervals = Intervals()
        current_interval_start = None
        last_activity_time = None

        with OWAMcapReader(file_path) as reader:
            for topic, timestamp, msg in reader.iter_decoded_messages(
                topics=["keyboard", "mouse"]
            ):
                # If this is the first activity or we're starting a new interval after inactivity
                if current_interval_start is None:
                    current_interval_start = timestamp
                    last_activity_time = timestamp
                    continue

                # If we have a gap in activity exceeding our threshold
                if timestamp - last_activity_time > int(
                    self.inactivity_threshold * TimeUnits.SECOND
                ):
                    # Close the previous interval
                    if (
                        current_interval_start is not None
                        and current_interval_start < last_activity_time
                    ):
                        activity_intervals.add(
                            (current_interval_start, last_activity_time)
                        )
                    # Start a new interval
                    current_interval_start = timestamp

                # Update the last activity time
                last_activity_time = timestamp

        # Add the final interval if there is one
        if current_interval_start is not None and last_activity_time is not None:
            activity_intervals.add((current_interval_start, last_activity_time))

        return activity_intervals


class WholeIntervalExtractor(IntervalExtractor):
    """Extract intervals without any specific logic, just return the whole file duration."""

    def extract_intervals(self, file_path: Path) -> Intervals:
        """Return a single interval covering the entire file duration."""
        with OWAMcapReader(file_path) as reader:
            start_time = reader.start_time
            end_time = reader.end_time

        return Intervals(
            [(start_time, end_time)]
        )  # Return the whole file as a single interval
