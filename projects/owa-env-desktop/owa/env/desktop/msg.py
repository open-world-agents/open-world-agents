"""
Legacy message definitions for backward compatibility.

DEPRECATED: These message definitions have been moved to the owa-msgs package
for better organization and centralized management. Please use the new imports:

    from owa.msgs.desktop.keyboard import KeyboardEvent, KeyboardState
    from owa.msgs.desktop.mouse import MouseEvent, MouseState

Or access via the message registry:

    from owa.core import MESSAGES
    KeyboardEvent = MESSAGES['desktop/KeyboardEvent']

This module provides compatibility imports and will be removed in a future version.
"""

import warnings
from typing import Annotated, Literal, TypeAlias

from annotated_types import Ge, Lt

# Import new message classes for compatibility
try:
    from owa.msgs.desktop.keyboard import KeyboardEvent as _NewKeyboardEvent
    from owa.msgs.desktop.keyboard import KeyboardState as _NewKeyboardState
    from owa.msgs.desktop.mouse import MouseEvent as _NewMouseEvent
    from owa.msgs.desktop.mouse import MouseState as _NewMouseState
    from owa.msgs.desktop.window import WindowInfo as _NewWindowInfo
except ImportError:
    # Fallback if owa-msgs is not installed
    _NewKeyboardEvent = None
    _NewKeyboardState = None
    _NewMouseEvent = None
    _NewMouseState = None
    _NewWindowInfo = None

from owa.core.message import OWAMessage

# Type aliases for backward compatibility
UInt8 = Annotated[int, Ge(0), Lt(256)]
MouseButton: TypeAlias = Literal["unknown", "left", "middle", "right", "x1", "x2"]


def _deprecation_warning(old_name: str, new_import: str) -> None:
    """Issue deprecation warning for legacy message usage."""
    warnings.warn(
        f"Using {old_name} from owa.env.desktop.msg is deprecated. Use: {new_import}", DeprecationWarning, stacklevel=3
    )


# Compatibility classes that redirect to new implementations
class KeyboardEvent:
    """Legacy KeyboardEvent - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning(
            "owa.env.desktop.msg.KeyboardEvent", "from owa.msgs.desktop.keyboard import KeyboardEvent"
        )
        if _NewKeyboardEvent is not None:
            return _NewKeyboardEvent(*args, **kwargs)
        else:
            # Fallback implementation
            return _LegacyKeyboardEvent(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewKeyboardEvent is not None:
            return _NewKeyboardEvent.deserialize(buffer)
        else:
            return _LegacyKeyboardEvent.deserialize(buffer)


class KeyboardState:
    """Legacy KeyboardState - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning(
            "owa.env.desktop.msg.KeyboardState", "from owa.msgs.desktop.keyboard import KeyboardState"
        )
        if _NewKeyboardState is not None:
            return _NewKeyboardState(*args, **kwargs)
        else:
            return _LegacyKeyboardState(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewKeyboardState is not None:
            return _NewKeyboardState.deserialize(buffer)
        else:
            return _LegacyKeyboardState.deserialize(buffer)


class MouseEvent:
    """Legacy MouseEvent - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning("owa.env.desktop.msg.MouseEvent", "from owa.msgs.desktop.mouse import MouseEvent")
        if _NewMouseEvent is not None:
            return _NewMouseEvent(*args, **kwargs)
        else:
            return _LegacyMouseEvent(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewMouseEvent is not None:
            return _NewMouseEvent.deserialize(buffer)
        else:
            return _LegacyMouseEvent.deserialize(buffer)


class MouseState:
    """Legacy MouseState - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning("owa.env.desktop.msg.MouseState", "from owa.msgs.desktop.mouse import MouseState")
        if _NewMouseState is not None:
            return _NewMouseState(*args, **kwargs)
        else:
            return _LegacyMouseState(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewMouseState is not None:
            return _NewMouseState.deserialize(buffer)
        else:
            return _LegacyMouseState.deserialize(buffer)


# Fallback implementations for when owa-msgs is not available
class _LegacyKeyboardEvent(OWAMessage):
    _type = "owa.env.desktop.msg.KeyboardEvent"
    event_type: Literal["press", "release"]
    vk: int


class _LegacyKeyboardState(OWAMessage):
    _type = "owa.env.desktop.msg.KeyboardState"
    buttons: set[UInt8]


class _LegacyMouseEvent(OWAMessage):
    _type = "owa.env.desktop.msg.MouseEvent"
    event_type: Literal["move", "click", "scroll"]
    x: int
    y: int
    button: MouseButton | None = None
    pressed: bool | None = None
    dx: int | None = None
    dy: int | None = None


class _LegacyMouseState(OWAMessage):
    _type = "owa.env.desktop.msg.MouseState"
    x: int
    y: int
    buttons: set[MouseButton]


class WindowInfo:
    """Legacy WindowInfo - redirects to new implementation."""

    def __new__(cls, *args, **kwargs):
        _deprecation_warning("owa.env.desktop.msg.WindowInfo", "from owa.msgs.desktop.window import WindowInfo")
        if _NewWindowInfo is not None:
            return _NewWindowInfo(*args, **kwargs)
        else:
            return _LegacyWindowInfo(*args, **kwargs)

    @classmethod
    def deserialize(cls, buffer):
        """Deserialize method for legacy compatibility."""
        if _NewWindowInfo is not None:
            return _NewWindowInfo.deserialize(buffer)
        else:
            return _LegacyWindowInfo.deserialize(buffer)


# Fallback implementation for when owa-msgs is not available
class _LegacyWindowInfo(OWAMessage):
    _type = "owa.env.desktop.msg.WindowInfo"

    title: str
    # rect has (left, top, right, bottom) format
    # normally,
    # 0 <= left < right <= screen_width
    # 0 <= top < bottom <= screen_height
    rect: tuple[int, int, int, int]
    hWnd: int

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]
