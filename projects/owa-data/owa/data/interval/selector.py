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
    def extract_intervals(self, episode_path: Path) -> Intervals:
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

    def extract_intervals(self, episode_path: Path) -> Intervals:
        min_ts = None
        max_ts = None

        with OWAMcapReader(episode_path) as reader:
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

    def extract_intervals(self, episode_path: Path) -> Intervals:
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

    def extract_intervals(self, episode_path: Path) -> Intervals:
        """
        Iterate over keyboard messages; whenever the specified start_stop_key is
        released, record its timestamp. Then pair off even and odd indices into
        (start, end) timestamp intervals.
        """
        timestamps: list[int] = []
        with OWAMcapReader(episode_path) as reader:
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

    This enhanced implementation uses composite interval operations to handle different
    inactivity thresholds for different event types:
    - Screen topics: 1 second inactivity threshold
    - Input devices (keyboard, mouse/raw): 5 seconds inactivity threshold

    The activity interval spans from the first to last screen topic event, with
    inactivity gaps removed based on the appropriate thresholds.
    """

    def __init__(self, screen_inactivity_threshold: float = 1.0, input_inactivity_threshold: float = 5.0):
        """
        Args:
            screen_inactivity_threshold: Seconds of gap between screen events to consider inactivity (default: 1.0).
            input_inactivity_threshold: Seconds of gap between input events to consider inactivity (default: 5.0).
        """
        self.screen_inactivity_threshold = screen_inactivity_threshold
        self.input_inactivity_threshold = input_inactivity_threshold

    def extract_intervals(self, episode_path: Path) -> Intervals:
        """
        Extract activity intervals using composite interval operations.

        1. Find screen topic boundaries (first to last screen event)
        2. Create screen activity intervals (removing >1s gaps)
        3. Create input device activity intervals (removing >5s gaps)
        4. Use intersection to find periods where both are active
        5. Constrain result to screen boundary interval
        """
        # Step 1: Find screen topic boundaries
        screen_boundary = self._get_screen_boundary_interval(episode_path)
        if screen_boundary.is_empty:
            return Intervals()  # No screen events found

        # Step 2: Create screen activity intervals
        screen_activity = self._get_topic_activity_intervals(
            episode_path, ["screen"], self.screen_inactivity_threshold
        )

        # Step 3: Create input device activity intervals
        input_activity = self._get_topic_activity_intervals(
            episode_path, ["keyboard", "mouse/raw"], self.input_inactivity_threshold
        )

        # Step 4: Use composite interval operations
        # Use intersection of screen and input activities, within boundary
        result = screen_activity & input_activity & screen_boundary

        return result

    def _get_screen_boundary_interval(self, episode_path: Path) -> Intervals:
        """
        Find the overall boundary interval from first to last screen topic event.

        Returns:
            Intervals containing a single interval from first to last screen event,
            or empty Intervals if no screen events found.
        """
        first_screen_time = None
        last_screen_time = None

        with OWAMcapReader(episode_path) as reader:
            try:
                first_screen_time = next(reader.iter_messages(topics=["screen"])).timestamp
                last_screen_time = next(reader.iter_messages(topics=["screen"], reverse=True)).timestamp
            except StopIteration:
                pass  # No screen events, will return empty Intervals below

        if first_screen_time is None or last_screen_time is None:
            return Intervals()

        return Intervals([(first_screen_time, last_screen_time)])

    def _get_topic_activity_intervals(
        self, episode_path: Path, topics: list[str], inactivity_threshold: float
    ) -> Intervals:
        """
        Extract activity intervals for specific topics with given inactivity threshold.

        Args:
            episode_path: Path to MCAP file
            topics: List of topic names to process
            inactivity_threshold: Gap threshold in seconds to consider inactivity

        Returns:
            Intervals representing periods of activity (gaps > threshold removed)
        """
        activity_intervals = Intervals()
        threshold_ns = int(inactivity_threshold * TimeUnits.SECOND)

        current_interval_start: int | None = None
        last_activity_time: int | None = None

        # Since iter_messages() yields messages in chronological order by default,
        # we can process them in a streaming fashion without buffering all timestamps
        with OWAMcapReader(episode_path) as reader:
            for mcap_msg in reader.iter_messages(topics=topics):
                # If this is the first event, mark the start of the first interval
                if current_interval_start is None:
                    current_interval_start = mcap_msg.timestamp
                    last_activity_time = mcap_msg.timestamp
                    continue

                # If gap > threshold, close previous interval and begin a new one
                # At this point, last_activity_time is guaranteed to be not None
                assert last_activity_time is not None
                if mcap_msg.timestamp - last_activity_time > threshold_ns:
                    if current_interval_start < last_activity_time:
                        activity_intervals.add((current_interval_start, last_activity_time))
                    current_interval_start = mcap_msg.timestamp

                last_activity_time = mcap_msg.timestamp

        # After the loop, if there's an open interval, close it
        if current_interval_start is not None and last_activity_time is not None:
            if current_interval_start < last_activity_time:
                activity_intervals.add((current_interval_start, last_activity_time))

        return activity_intervals


# --- Composite extractor classes for &, |, and - operations --- #


class IntervalAnd(IntervalExtractor):
    """Composite extractor that returns the intersection of two extractors' intervals."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, episode_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(episode_path)
        right_intervals = self.right.extract_intervals(episode_path)
        return left_intervals & right_intervals


class IntervalOr(IntervalExtractor):
    """Composite extractor that returns the union of two extractors' intervals."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, episode_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(episode_path)
        right_intervals = self.right.extract_intervals(episode_path)
        return left_intervals | right_intervals


class IntervalSubtract(IntervalExtractor):
    """Composite extractor that subtracts one extractor's intervals from another's."""

    def __init__(self, left: IntervalExtractor, right: IntervalExtractor):
        self.left = left
        self.right = right

    def extract_intervals(self, episode_path: Path) -> Intervals:
        left_intervals = self.left.extract_intervals(episode_path)
        right_intervals = self.right.extract_intervals(episode_path)
        return left_intervals - right_intervals
