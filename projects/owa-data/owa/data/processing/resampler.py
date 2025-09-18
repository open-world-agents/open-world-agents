"""Event resampling implementations for OWA data processing."""

import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import List

from mcap_owa.highlevel import McapMessage
from owa.core.time import TimeUnits
from owa.msgs.desktop.mouse import RawMouseEvent


class EventResampler(ABC):
    """Abstract base class for event resampling strategies.

    Provides a common interface for different event resampling algorithms
    that control the rate and timing of event processing in the OWA pipeline.
    """

    @abstractmethod
    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add an incoming event to the resampler for processing.

        Args:
            mcap_msg: The MCAP message containing the event data to be processed.
        """
        pass

    @abstractmethod
    def step(self, now: int) -> None:
        """Advance the resampler to the specified timestamp.

        Args:
            now: Current timestamp in nanoseconds. Must be monotonically increasing.
        """
        pass

    @abstractmethod
    def pop_events(self) -> List[McapMessage]:
        """Retrieve and remove all ready events from the resampler.

        Returns:
            List of processed MCAP messages ready for output. Safe to call multiple times.
        """
        pass


class DropResampler(EventResampler):
    """Simple drop-based resampling strategy.

    Filters events by dropping those that arrive too frequently, ensuring
    a minimum time interval between consecutive events.
    """

    def __init__(self, *, min_interval_ns: int) -> None:
        self.min_interval_ns = min_interval_ns
        self.last_emitted_timestamp = 0
        self.input_queue = deque()
        self.output_queue = []

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Accept event only if sufficient time has passed since the last accepted event."""
        if (mcap_msg.timestamp - self.last_emitted_timestamp) >= self.min_interval_ns:
            self.last_emitted_timestamp = mcap_msg.timestamp
            self.input_queue.append(mcap_msg)

    def step(self, now: int) -> None:
        """Move all events with timestamps up to 'now' from input to output queue."""
        while self.input_queue and self.input_queue[0].timestamp <= now:
            self.output_queue.append(self.input_queue.popleft())

    def pop_events(self) -> List[McapMessage]:
        """Return all ready events and clear the output queue."""
        events = self.output_queue
        self.output_queue = []
        return events


class KeyboardUniformResampler(EventResampler):
    """Resample keyboard events to maintain uniform intervals during key holds.

    Simulates consistent key repeat behavior by generating synthetic press events
    at regular intervals while a key is held down. Typical keyboard behavior:
    initial press, then repeat starts after 500ms, continuing every 30ms until release.
    This resampler outputs press events at the specified min_interval_ns interval.
    """

    def __init__(self, *, min_interval_ns: int) -> None:
        self.min_interval_ns = min_interval_ns
        self.keys = {}  # Maps key -> {"pressed": bool, "last_created_timestamp": int | None}
        self.input_queue = deque()
        self.output_queue = []

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Process keyboard events and track key state for uniform resampling."""
        key = mcap_msg.decoded.vk
        if key not in self.keys:
            self.keys[key] = {"pressed": False, "last_created_timestamp": None}

        if mcap_msg.decoded.event_type == "press":
            # Only queue the first press event for each key hold
            if not self.keys[key]["pressed"]:
                self.input_queue.append(mcap_msg)
                self.keys[key]["last_created_timestamp"] = mcap_msg.timestamp
            self.keys[key]["pressed"] = True
        else:
            # Always queue release events
            self.input_queue.append(mcap_msg)
            self.keys[key]["pressed"] = False

    def step(self, now: int) -> None:
        """Process queued events and generate synthetic press events for held keys."""
        while self.input_queue and self.input_queue[0].timestamp <= now:
            self.output_queue.append(self.input_queue.popleft())
        self.output_queue.extend(self._create_events(now))
        self.output_queue.sort(key=lambda x: x.timestamp)

    def _create_events(self, until_now: int) -> List[McapMessage]:
        """Generate synthetic press events for keys that are currently held down."""
        from owa.msgs.desktop.keyboard import KeyboardEvent

        events = []
        for key, state in self.keys.items():
            if not state["pressed"]:
                continue
            if state["last_created_timestamp"] is None:
                continue

            # Handle large time gaps (e.g., system suspend/resume)
            if until_now - state["last_created_timestamp"] >= TimeUnits.SECOND:
                warnings.warn(
                    f"Large time gap for key {key}: {until_now - state['last_created_timestamp']}ns, IGNORING"
                )
                state["last_created_timestamp"] = until_now

            # Generate press events at regular intervals
            while (until_now - state["last_created_timestamp"]) >= self.min_interval_ns:
                next_timestamp = state["last_created_timestamp"] + self.min_interval_ns
                events.append(
                    McapMessage(
                        topic="keyboard",
                        timestamp=next_timestamp,
                        message=KeyboardEvent(event_type="press", vk=key, timestamp=next_timestamp).model_dump_json(),
                        message_type="desktop/KeyboardEvent",
                    )
                )
                state["last_created_timestamp"] = next_timestamp

        return events

    def pop_events(self) -> List[McapMessage]:
        """Return all processed events and clear the output queue."""
        events = self.output_queue
        self.output_queue = []
        return events


class PassThroughResampler(EventResampler):
    """Pass-through resampler that preserves all events without modification.

    Used when no resampling is desired, allowing all events to pass through
    the pipeline unchanged while maintaining the resampler interface.
    """

    def __init__(self) -> None:
        self.input_queue = deque()
        self.output_queue = []

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Accept all events without any filtering or modification."""
        self.input_queue.append(mcap_msg)

    def step(self, now: int) -> None:
        """Move all events with timestamps up to 'now' to the output queue."""
        while self.input_queue and self.input_queue[0].timestamp <= now:
            self.output_queue.append(self.input_queue.popleft())

    def pop_events(self) -> List[McapMessage]:
        """Return all ready events and clear the output queue."""
        events = self.output_queue
        self.output_queue = []
        return events


class MouseAggregationResampler(EventResampler):
    """Mouse resampler that accumulates movement deltas to reduce event frequency.

    Combines multiple small mouse movements into larger aggregated movements,
    reducing the overall event rate while preserving total movement distance.
    Non-movement events (clicks, scrolls) are passed through unchanged.
    """

    def __init__(self, *, min_interval_ns: int) -> None:
        self.min_interval_ns = min_interval_ns
        self.last_emitted_timestamp = 0
        self.accumulated_dx = 0
        self.accumulated_dy = 0
        self.input_queue = deque()
        self.output_queue = []

    def add_event(self, mcap_msg: McapMessage[RawMouseEvent]) -> None:
        """Accumulate mouse movement deltas or pass through non-movement events."""
        mouse_event = mcap_msg.decoded

        # Identify simple mouse movement events (no button interactions)
        is_simple_move = mouse_event.button_flags == RawMouseEvent.ButtonFlags.RI_MOUSE_NOP

        if is_simple_move:
            assert mouse_event.button_data == 0, "Non-zero button data in simple move event"

            # Accumulate movement deltas for later aggregation
            self.accumulated_dx += mouse_event.dx
            self.accumulated_dy += mouse_event.dy

            # Emit aggregated movement when sufficient time has elapsed
            if (mcap_msg.timestamp - self.last_emitted_timestamp) >= self.min_interval_ns:
                aggregated_event = RawMouseEvent(
                    us_flags=mouse_event.us_flags,
                    last_x=self.accumulated_dx,
                    last_y=self.accumulated_dy,
                    button_flags=mouse_event.button_flags,
                    button_data=mouse_event.button_data,
                    device_handle=mouse_event.device_handle,
                    timestamp=mcap_msg.timestamp,
                )

                self.input_queue.append(
                    McapMessage(
                        topic=mcap_msg.topic,
                        timestamp=mcap_msg.timestamp,
                        message=aggregated_event.model_dump_json().encode("utf-8"),
                        message_type=mcap_msg.message_type,
                    )
                )

                # Reset accumulation state for next interval
                self.accumulated_dx = 0
                self.accumulated_dy = 0
                self.last_emitted_timestamp = mcap_msg.timestamp
        else:
            # Pass through non-movement events immediately (clicks, scrolls, etc.)
            self.input_queue.append(mcap_msg)

    def step(self, now: int) -> None:
        """Move all events with timestamps up to 'now' from input to output queue."""
        while self.input_queue and self.input_queue[0].timestamp <= now:
            self.output_queue.append(self.input_queue.popleft())

    def pop_events(self) -> List[McapMessage]:
        """Return all ready events and clear the output queue."""
        events = self.output_queue
        self.output_queue = []
        return events


def create_resampler(topic: str, *, min_interval_ns: int = 0, **kwargs) -> EventResampler:
    """Create the most appropriate resampler for a given event topic.

    Args:
        topic: Event topic name (e.g., "mouse/raw", "keyboard", "screen").
        min_interval_ns: Minimum interval between events in nanoseconds.
                        If 0, returns PassThroughResampler.
        **kwargs: Additional arguments passed to the resampler constructor.

    Returns:
        EventResampler instance optimized for the specified topic.
    """
    if min_interval_ns == 0:
        return PassThroughResampler()

    resampler_map = {
        "mouse/raw": MouseAggregationResampler,
        "keyboard": KeyboardUniformResampler,
    }
    resampler_class = resampler_map.get(topic, DropResampler)
    return resampler_class(min_interval_ns=min_interval_ns, **kwargs)


class EventResamplerDict(dict):
    """Dictionary of resamplers for multiple topics."""

    def __init__(self, rate_settings: dict[str, float], **kwargs) -> None:
        super().__init__()
        for topic, rate_hz in rate_settings.items():
            min_interval_ns = 0 if rate_hz == 0 else int((1.0 / rate_hz) * 1e9)
            self[topic] = create_resampler(topic, min_interval_ns=min_interval_ns, **kwargs)

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add an incoming event to the appropriate resampler."""
        # Lazily create resampler if topic is new
        if mcap_msg.topic not in self:
            self[mcap_msg.topic] = create_resampler(mcap_msg.topic, min_interval_ns=0)

        # Add event to resampler
        self[mcap_msg.topic].add_event(mcap_msg)

    def step(self, now: int) -> None:
        """Advance all resamplers to the specified timestamp."""
        for resampler in self.values():
            resampler.step(now)

    def pop_events(self) -> List[McapMessage]:
        """Return all ready events from all resamplers."""
        events = []
        for resampler in self.values():
            events.extend(resampler.pop_events())
        return sorted(events, key=lambda x: x.timestamp)
