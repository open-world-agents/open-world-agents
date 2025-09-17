"""Event resampling implementations for OWA data processing."""

from abc import ABC, abstractmethod
from typing import List

from mcap_owa.highlevel import McapMessage
from owa.msgs.desktop.mouse import RawMouseEvent


class EventResampler(ABC):
    """Abstract base class for event resampling strategies."""

    @abstractmethod
    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add an event to the resampler."""
        pass

    @abstractmethod
    def pop_event(self, now: int) -> List[McapMessage]:
        """Pop ready events from the resampler."""
        pass


class DropResampler(EventResampler):
    """Simple drop-based resampling."""

    def __init__(self, *, min_interval_ns: int):
        self.min_interval_ns = min_interval_ns
        self.last_emitted_timestamp = 0
        self.ready_events = []

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add event if enough time has passed."""
        if (mcap_msg.timestamp - self.last_emitted_timestamp) >= self.min_interval_ns:
            self.last_emitted_timestamp = mcap_msg.timestamp
            self.ready_events.append(mcap_msg)

    def pop_event(self, now: int) -> List[McapMessage]:
        """Pop all ready events."""
        events = self.ready_events
        self.ready_events = []
        return events


class KeyboardUniformResampler(EventResampler):
    """Resample keyboard events to a uniform interval.

    Common keypress: if we press a key, event repeat starts at 500ms, then every 30ms.
    Output of resampler: press events with timestamp at min_interval_ns interval
    """

    def __init__(self, *, min_interval_ns: int):
        self.min_interval_ns = min_interval_ns
        self.keys = {}  # key -> (is_pressed,  last_popped_timestamp)

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add event if enough time has passed."""
        key = mcap_msg.decoded.vk
        if key not in self.keys:
            self.keys[key] = [False, None]
        if mcap_msg.decoded.event_type == "press":
            self.keys[key][0] = "press"
        else:
            self.keys[key][0] = "release"
            self.keys[key][1] = None  # reset popped timestamp

    def pop_event(self, now: int) -> List[McapMessage]:
        """Pop all ready events."""
        ready_events = []
        for key in self.keys:
            # we must repeat event between self.min_interval_ns after last_popped_timestamp
            if self.keys[key][0] != "press":
                continue
            while self.keys[key][1] is None or (now - self.keys[key][1]) >= self.min_interval_ns:
                from owa.msgs.desktop.keyboard import KeyboardEvent

                new_timestamp = self.keys[key][1] + self.min_interval_ns if self.keys[key][1] is not None else now

                ready_events.append(
                    McapMessage(
                        topic="keyboard",
                        timestamp=new_timestamp,
                        message=KeyboardEvent(event_type="press", vk=key, timestamp=new_timestamp).model_dump_json(),
                        message_type="desktop/KeyboardEvent",
                    )
                )
                self.keys[key][1] = new_timestamp
        ready_events.sort(key=lambda e: e.timestamp)
        return ready_events


class PassThroughResampler(EventResampler):
    """All-passing resampler that doesn't filter any events."""

    def __init__(self):
        self.ready_events = []

    def add_event(self, mcap_msg: McapMessage) -> None:
        """Add all events without filtering."""
        self.ready_events.append(mcap_msg)

    def pop_event(self, now: int) -> List[McapMessage]:
        """Pop all ready events."""
        events = self.ready_events
        self.ready_events = []
        return events


class MouseAggregationResampler(EventResampler):
    """Mouse resampler that accumulates dx/dy values."""

    def __init__(self, *, min_interval_ns: int, **kwargs):
        self.min_interval_ns = min_interval_ns
        self.last_emitted_timestamp = 0
        self.accumulated_dx = 0
        self.accumulated_dy = 0
        self.ready_events = []

    def add_event(self, mcap_msg: McapMessage[RawMouseEvent]) -> None:
        """Accumulate mouse movement or pass through non-movement events."""
        mouse_event = mcap_msg.decoded

        # Check if this is a simple mouse move (no buttons)
        is_simple_move = mouse_event.button_flags == RawMouseEvent.ButtonFlags.RI_MOUSE_NOP

        if is_simple_move:
            assert mouse_event.button_data == 0, "Non-zero button data in simple move event"

            # Accumulate movement
            self.accumulated_dx += mouse_event.dx
            self.accumulated_dy += mouse_event.dy

            # Emit aggregated movement if enough time passed
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

                self.ready_events.append(
                    McapMessage(
                        topic=mcap_msg.topic,
                        timestamp=mcap_msg.timestamp,
                        message=aggregated_event.model_dump_json().encode("utf-8"),
                        message_type=mcap_msg.message_type,
                    )
                )

                # Reset state
                self.accumulated_dx = 0
                self.accumulated_dy = 0
                self.last_emitted_timestamp = mcap_msg.timestamp
        else:
            # Pass through non-movement events (clicks, etc.)
            self.ready_events.append(mcap_msg)

    def pop_event(self, now: int) -> List[McapMessage]:
        """Pop all ready events."""
        events = self.ready_events
        self.ready_events = []
        return events


def create_resampler(topic: str, *, min_interval_ns: int = 0, **kwargs) -> EventResampler:
    """Create appropriate resampler for a given topic."""
    if min_interval_ns == 0:
        return PassThroughResampler()

    resampler_map = {
        "mouse/raw": MouseAggregationResampler,
        "keyboard": KeyboardUniformResampler,
    }
    resampler_class = resampler_map.get(topic, DropResampler)
    return resampler_class(min_interval_ns=min_interval_ns, **kwargs)
