"""
Desktop keyboard message definitions.

This module contains message types for keyboard events and state,
following the domain-based message naming convention for better organization.
"""

from typing import Annotated, Literal

from annotated_types import Ge, Lt

from owa.core.message import OWAMessage

# Matches definition of Windows Virtual Key Codes
# https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
UInt8 = Annotated[int, Ge(0), Lt(256)]


class KeyboardEvent(OWAMessage):
    """
    Represents a keyboard key press or release event.

    This message captures individual keyboard events with timing information,
    suitable for recording user interactions and replaying them.

    Attributes:
        event_type: Type of event - "press" or "release"
        vk: Virtual key code (Windows VK codes)
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/KeyboardEvent"

    event_type: Literal["press", "release"]
    vk: int
    timestamp: int | None = None


class KeyboardState(OWAMessage):
    """
    Represents the current state of all keyboard keys.

    This message captures the complete keyboard state at a point in time,
    useful for state synchronization and debugging.

    Attributes:
        buttons: Set of virtual key codes currently pressed
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/KeyboardState"

    buttons: set[UInt8]
    timestamp: int | None = None
