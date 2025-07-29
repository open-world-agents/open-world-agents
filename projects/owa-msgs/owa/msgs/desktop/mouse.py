"""
Desktop mouse message definitions.

This module contains message types for mouse events and state,
following the domain-based message naming convention for better organization.
"""

from enum import IntFlag
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

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


class RawMouseEvent(OWAMessage):
    """
    Represents raw mouse input data from Windows WM_INPUT messages.
    Ref: https://learn.microsoft.com/ko-kr/windows/win32/api/winuser/ns-winuser-rawmouse

    This message captures high-definition mouse movement data directly from the HID stack,
    bypassing Windows pointer acceleration and screen resolution limits. Provides sub-pixel
    precision and unfiltered input data essential for gaming and precision applications.

    Attributes:
        us_flags: mouse state flags, containing movement data type (relative/absolute). Default is relative.
        last_x: can be relative or absolute, depends on us_flags
        last_y: can be relative or absolute, depends on us_flags
        button_flags: Raw button state flags from Windows RAWMOUSE structure
        button_data: Additional button data (wheel delta, etc.)
        device_handle: Raw input device handle (optional)
        timestamp: Optional timestamp in nanoseconds since epoch
    Properties:
        dx: Horizontal movement delta (derived from last_x)
        dy: Vertical movement delta (derived from last_y)
    """

    _type = "desktop/RawMouseEvent"

    last_x: int  # can be relative or absolute, depends on us_flags
    last_y: int  # can be relative or absolute, depends on us_flags

    class UsFlags(IntFlag):
        MOUSE_MOVE_RELATIVE = 0x0000
        MOUSE_MOVE_ABSOLUTE = 0x0001
        MOUSE_VIRTUAL_DESKTOP = 0x0002  # the coordinates are mapped to the virtual desktop
        MOUSE_ATTRIBUTES_CHANGED = 0x0004  # the mouse button flags or mouse attributes have changed
        MOUSE_MOVE_NOCOALESCE = 0x0008  # the message should not be coalesced

    us_flags: UsFlags = (
        UsFlags.MOUSE_MOVE_RELATIVE
    )  # mouse state flags, containing movement data type (relative/absolute). Default is relative.

    class ButtonFlags(IntFlag):
        RI_MOUSE_NOP = 0x0000
        RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
        RI_MOUSE_LEFT_BUTTON_UP = 0x0002
        RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
        RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
        RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
        RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020
        RI_MOUSE_BUTTON_4_DOWN = 0x0040
        RI_MOUSE_BUTTON_4_UP = 0x0080
        RI_MOUSE_BUTTON_5_DOWN = 0x0100
        RI_MOUSE_BUTTON_5_UP = 0x0200
        RI_MOUSE_WHEEL = 0x0400
        RI_MOUSE_HWHEEL = 0x0800

    # Raw button information from Windows RAWMOUSE structure
    button_flags: ButtonFlags = ButtonFlags.RI_MOUSE_NOP  # RI_MOUSE_* flags (button press/release, wheel)
    button_data: int = 0  # Additional data (wheel delta, x-button info)

    # Device information
    device_handle: int | None = None  # HANDLE to raw input device

    # Timing
    timestamp: int | None = None

    @property
    def dx(self) -> int:
        """Get raw horizontal movement delta."""
        if self.us_flags == self.UsFlags.MOUSE_MOVE_RELATIVE:
            return self.last_x
        else:
            raise NotImplementedError("Absolute mouse movement not implemented")

    @property
    def dy(self) -> int:
        """Get raw vertical movement delta."""
        if self.us_flags == self.UsFlags.MOUSE_MOVE_RELATIVE:
            return self.last_y
        else:
            raise NotImplementedError("Absolute mouse movement not implemented")


class PointerBallisticsConfig(OWAMessage):
    """Windows pointer ballistics configuration for WM_MOUSEMOVE reconstruction."""

    model_config = ConfigDict(populate_by_name=True)
    _type = "desktop/PointerBallisticsConfig"

    mouse_threshold1: int = Field(alias="MouseThreshold1")
    mouse_threshold2: int = Field(alias="MouseThreshold2")
    mouse_speed: int = Field(alias="MouseSpeed")
    mouse_sensitivity: int = Field(alias="MouseSensitivity")
    smooth_mouse_x_curve: str | None = Field(default=None, alias="SmoothMouseXCurve")
    smooth_mouse_y_curve: str | None = Field(default=None, alias="SmoothMouseYCurve")
