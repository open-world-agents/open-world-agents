from abc import ABC, abstractmethod
from pathlib import Path

from mcap_owa.highlevel import OWAMcapReader
from owa.core.time import TimeUnits
from owa.env.desktop.constants import VK

from .interval import Intervals


class IntervalExtractor(ABC):
    """
    Base class for interval extraction. Supports combining via &, |, and - operators.

    Subclasses must implement extract_intervals() to return an Intervals object.
    """

    @abstractmethod
    def extract_intervals(self, file_path: Path) -> Intervals:
        """
        Given a Path to an MCAP file, return an Intervals object containing
        valid (start, end) timestamp pairs according to this extractor's logic.
        """
        pass

    def filter_by_duration(self, intervals: Intervals, min_duration: int) -> Intervals:
        """
        Return only those intervals whose length is strictly greater than min_duration.

        Args:
            intervals: An Intervals object to filter.
            min_duration: An integer duration threshold (in the same time units as the Intervals).
        """
        result = Intervals()
        for interval in intervals:
            if interval.length > min_duration:
                result.add((interval.start, interval.end))
        return result

    # Operator overloads to allow syntax like (A & B) | C - D

    def __and__(self, other: "IntervalExtractor") -> "IntervalAnd":
        return IntervalAnd(self, other)

    def __or__(self, other: "IntervalExtractor") -> "IntervalOr":
        return IntervalOr(self, other)

    def __sub__(self, other: "IntervalExtractor") -> "IntervalSubtract":
        return IntervalSubtract(self, other)


class All(IntervalExtractor):
    """
    Return a single interval covering the entire file.

    Scans all messages in the MCAP to find the minimum and maximum timestamps,
    then returns an Intervals containing exactly one pair: (min_timestamp, max_timestamp).
    """

    def extract_intervals(self, file_path: Path) -> Intervals:
        min_ts = None
        max_ts = None

        with OWAMcapReader(file_path) as reader:
            min_ts = reader.start_time
            max_ts = reader.end_time

        if min_ts is None or max_ts is None:
            # No messages found => return empty intervals
            return Intervals()
        return Intervals([(min_ts, max_ts)])


class Empty(IntervalExtractor):
    """
    Always return an empty set of intervals.

    Acts as the identity element for union operations.
    """

    def extract_intervals(self, file_path: Path) -> Intervals:
        return Intervals()  # Always empty


class StartStopKeyPress(IntervalExtractor):
    """
    Extract intervals based on explicit start/stop key presses.

    By default, uses F9 as a toggle: on each 'release' of the F9 key,
    records a timestamp. Pairs of consecutive timestamps form (start, end).
    """

    def __init__(self, start_stop_key: int = VK.F9, pause_key: int = VK.F10):
        """
        Args:
            start_stop_key: Virtual key code for toggling start/end (default: F9).
            pause_key: Virtual key code for pause (currently not implemented, default: F10).
        """
        self.start_stop_key = start_stop_key
        self.pause_key = pause_key

    def extract_intervals(self, file_path: Path) -> Intervals:
        """
        Iterate over keyboard messages; whenever the specified start_stop_key is
        released, record its timestamp. Then pair off even and odd indices into
        (start, end) timestamp intervals.
        """
        timestamps: list[int] = []
        with OWAMcapReader(file_path) as reader:
            for mcap_msg in reader.iter_messages(topics=["keyboard"]):
                # Record on key release of start_stop_key
                if mcap_msg.decoded.event_type == "release" and mcap_msg.decoded.vk == self.start_stop_key:
                    timestamps.append(mcap_msg.timestamp)
                # Pause functionality not implemented
                elif mcap_msg.decoded.vk == self.pause_key:
                    raise NotImplementedError("Pause key is not implemented")

        # Pair consecutive timestamps: (timestamps[0], timestamps[1]), (timestamps[2], timestamps[3]), ...
        pairs = list(zip(timestamps[::2], timestamps[1::2]))
        return Intervals(pairs)


class InactivityFilter(IntervalExtractor):
    """
    Extract intervals by detecting periods of activity versus inactivity.

    This implementation scans both keyboard and mouse topics. If the gap
    between two consecutive events exceeds inactivity_threshold (in seconds),
    it closes the previous activity interval and starts a new one.
    """

    def __init__(self, inactivity_threshold: float = 5.0):
        """
        Args:
            inactivity_threshold: A float number of seconds that defines an inactivity gap.
        """
        self.inactivity_threshold = inactivity_threshold

    def extract_intervals(self, file_path: Path) -> Intervals:
        """
        Walk through keyboard and mouse messages in the MCAP file. Whenever the
        time difference between the current event and the last recorded activity
        exceeds inactivity_threshold, close off the current interval and start a new one.
        """
        activity_intervals = Intervals()
        current_interval_start = None
        last_activity_time = None

        with OWAMcapReader(file_path) as reader:
            for mcap_msg in reader.iter_messages(topics=["keyboard", "mouse"]):
                # If this is the first event, mark the start of the first interval
                if current_interval_start is None:
                    current_interval_start = mcap_msg.timestamp
                    last_activity_time = mcap_msg.timestamp
                    continue

                # If gap > threshold, close previous interval and begin a new one
                if mcap_msg.timestamp - last_activity_time > int(self.inactivity_threshold * TimeUnits.SECOND):
                    if current_interval_start < last_activity_time:
                        activity_intervals.add((current_interval_start, last_activity_time))
                    current_interval_start = mcap_msg.timestamp

                last_activity_time = mcap_msg.timestamp

        # After the loop, if there's an open interval, close it
        if current_interval_start is not None and last_activity_time is not None:
            activity_intervals.add((current_interval_start, last_activity_time))

        return activity_intervals


# --- Composite extractor classes for &, |, and - operations --- #


class IntervalAnd(IntervalExtractor):
    """Composite extractor that returns the intersection of two extractors' intervals."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, file_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(file_path)
        right_intervals = self.right.extract_intervals(file_path)
        return left_intervals & right_intervals


class IntervalOr(IntervalExtractor):
    """Composite extractor that returns the union of two extractors' intervals."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, file_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(file_path)
        right_intervals = self.right.extract_intervals(file_path)
        return left_intervals | right_intervals


class IntervalSubtract(IntervalExtractor):
    """Composite extractor that subtracts one extractor's intervals from another's."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, file_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(file_path)
        right_intervals = self.right.extract_intervals(file_path)
        return left_intervals - right_intervals
