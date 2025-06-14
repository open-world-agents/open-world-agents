"""
Desktop mouse message definitions.

This module contains message types for mouse events and state,
following the domain-based message naming convention for better organization.
"""

from typing import Literal, TypeAlias

from owa.core.message import OWAMessage

# Matches definition of https://github.com/moses-palmer/pynput/blob/master/lib/pynput/mouse/_win32.py#L48
MouseButton: TypeAlias = Literal["unknown", "left", "middle", "right", "x1", "x2"]


class MouseEvent(OWAMessage):
    """
    Represents a mouse event (movement, click, or scroll).

    This message captures mouse interactions with detailed event information,
    suitable for recording user interactions and replaying them.

    Attributes:
        event_type: Type of event - "move", "click", or "scroll"
        x: X coordinate on screen
        y: Y coordinate on screen
        button: Mouse button involved (for click events)
        pressed: Whether button was pressed (True) or released (False)
        dx: Horizontal scroll delta (for scroll events)
        dy: Vertical scroll delta (for scroll events)
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/MouseEvent"

    event_type: Literal["move", "click", "scroll"]
    x: int
    y: int
    button: MouseButton | None = None
    pressed: bool | None = None
    dx: int | None = None
    dy: int | None = None
    timestamp: int | None = None


class MouseState(OWAMessage):
    """
    Represents the current state of the mouse.

    This message captures the complete mouse state at a point in time,
    useful for state synchronization and debugging.

    Attributes:
        x: Current X coordinate on screen
        y: Current Y coordinate on screen
        buttons: Set of currently pressed mouse buttons
        timestamp: Optional timestamp in nanoseconds since epoch
    """

    _type = "desktop/MouseState"

    x: int
    y: int
    buttons: set[MouseButton]
    timestamp: int | None = None
